"""Dispider (CVPR 2025) with persistent per-clip perception memory.

Upstream: github.com/Mark12Ding/Dispider @ 803f6b4634b3232293e89a38bcce9f5815c33ed4 (vendored under
external/baselines/dispider), weights Mar2Ding/Dispider @ 7c5517eaaa4dc3788602af40e077063792c75f6c.

Matches upstream
- Clip segmentation: upstream `StreamingSession` (non-overlapping 16 s windows, 16 frames sampled nearest to
  uniformly spaced targets), driven with the 1 FPS evaluator frames. Each closed window is perceived once.
- Perception memory: CLIP tower + pool projector + compact Qwen2-1.5B decoder over
  "This contains a clip sampled in S to E seconds<image>", reading the memory (100) and time (200) token states.
  This part of the Decision forward is question-independent, so it is computed once per clip and cached.
- Answering: the question-dependent part of `VideoStream.react` (used for OVO-Bench MCQs): `_streaming_selection`
  with insert_position=0, ans_position=[] (global memory + top-4 query-selected clips), the Reaction projectors,
  and the Qwen2-7B Reaction LM with the upstream `qwen` conversation template.

Deviations
- The Decision trigger ("when to respond") is not run: the question is asked at t_q and must be answered now.
- The tail window that has not closed at t_q is perceived ephemerally at each query (upstream `flush`, i.e.
  complete=False, end_s = last frame time) and discarded; frames there may be encoded more than once.
- Decoding is a single step restricted to the option-letter tokens after "Best option: (".
- Decision decoder uses SDPA instead of the hardcoded FlashAttention-2 (flash_attn is not installable for the
  node's torch 2.11); the attention is the same exact computation.
- Question/answer never enter the cached state.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from scripts.eval.streaming_adapters import REPO_ROOT, StreamingSpec

UPSTREAM = REPO_ROOT / "external/baselines/dispider"

SPEC = StreamingSpec(
    key="dispider_7b",
    checkpoint="Dispider",
    runtime=".venv-dispider/bin/python",
    description="Dispider (Qwen2-7B Reaction), per-clip perception memory persisted (Mark12Ding/Dispider 803f6b4)",
)


def _force_sdpa_decision() -> None:
    from dispider.model.decision_backbone import adapter

    base_init = adapter.TransformersQwen2Model.__init__

    def __init__(self, config):
        config._attn_implementation = "sdpa"
        base_init(self, config)

    adapter.Qwen2Model.__init__ = __init__


class DispiderAdapter:
    def __init__(self, model_path: Path):
        import torch

        if str(UPSTREAM) not in sys.path:
            sys.path.insert(0, str(UPSTREAM))
        _force_sdpa_decision()
        from dispider.model.builder import load_pretrained_model
        from dispider.streaming.dispider_adapter import ReleasedPerceptionDecisionReactionBackend
        from dispider.streaming.session import StreamingSession

        if not torch.cuda.is_available():
            raise RuntimeError("Dispider requires CUDA")
        self.torch = torch
        tokenizer, model, processors, _ = load_pretrained_model(str(model_path), None, "Dispider", device_map={"": "cuda"})
        self.backend = ReleasedPerceptionDecisionReactionBackend(model, tokenizer, *processors)
        self.model = model.eval()
        self.decision = self.backend.perception_decision_model
        self.tokenizer = tokenizer
        # Separate hook object: StreamingSession.reset() calls each component's reset(), so passing the adapter
        # itself would recurse into adapter.reset() -> session.reset().
        hooks = _SessionHooks(self)
        self.session = StreamingSession(hooks, hooks, hooks)
        self.reset()

    # StreamingSession hooks: perceive closed windows, never trigger.
    def perceive(self, window):
        self.clips.append(self._encode(window))
        return None

    def should_respond(self, perception, *, window, history) -> bool:
        return False

    def respond(self, perception, *, window, history) -> str:
        raise RuntimeError("Dispider trigger is disabled in RevisitQA evaluation")

    def _encode(self, window) -> dict:
        from dispider.model.decision_backbone.memory import extract_perception_memory

        payload = self.backend.perceive_clip(window)
        device = self.decision.device
        images = payload.pixel_values.to(device=device, dtype=self.decision.dtype)
        seqs = payload.time_ids.unsqueeze(0).to(device)
        mask = seqs.ne(self.backend.time_tokenizer.pad_token_id)
        ans, todo = self.backend._delimiter_tokens(device)
        qs = self.backend._question_token_ids("").unsqueeze(0).to(device)  # unused by the perception pass
        with self.torch.inference_mode():
            prep = self.decision.prepare_inference_inputs(seqs, None, mask, qs, qs.ne(-1), None, images, ans, todo)
            memory = extract_perception_memory(self.decision, None, prep[4], prep[2], prep[8], 100)
        return {"clip": memory.clip, "time": memory.time, "embeds": prep[5][0]}

    def reset(self) -> None:
        self.clips = []
        self.session.reset()
        self.frame_count = 0

    def observe(self, frames) -> None:
        if frames:
            self.session.push_frames([f.image for f in frames], [f.timestamp for f in frames])
            self.frame_count += len(frames)

    def _tail_clip(self):
        from dispider.streaming.types import SampledWindow

        cands = self.session._candidates
        if not cands or cands[0] is None:
            return None
        window = SampledWindow(
            start_s=self.session._window_start_s, end_s=self.session._window_start_s + self.session.window_seconds,
            frames=tuple(c.frame for c in cands), timestamps_s=tuple(c.timestamp_s for c in cands), complete=False,
        )
        return self._encode(window)

    def _option_token_ids(self, prompt: str) -> list[int]:
        labels = []
        for line in prompt.splitlines():
            match = re.match(r"\s*\(([A-E])\)\s+", line)
            if match and match.group(1) not in labels:
                labels.append(match.group(1))
        ids = []
        for label in labels:
            encoded = self.tokenizer(label, add_special_tokens=False).input_ids
            if len(encoded) != 1:
                raise ValueError(f"option label {label!r} is not a single token")
            ids.append(encoded[0])
        return ids

    def answer(self, prompt: str, option_count: int) -> str:
        from dispider.model.decision_backbone.memory import PerceptionMemory

        torch = self.torch
        clips = list(self.clips)
        tail = self._tail_clip()
        if tail is not None:
            clips.append(tail)
        if not clips:
            raise RuntimeError("Dispider memory is empty before answering")
        device = self.decision.device
        clip = torch.cat([c["clip"] for c in clips])
        time = torch.cat([c["time"] for c in clips])
        memory = PerceptionMemory(clip=clip, time=time, per_clip=clip.view(len(clips), -1, clip.shape[-1]))
        embeds = torch.stack([c["embeds"] for c in clips])
        qs = self.backend._question_token_ids(prompt).unsqueeze(0).to(device)
        ans, todo = self.backend._delimiter_tokens(device)
        emb = self.decision.get_model().embed_tokens
        rmodel = self.model.get_model()
        with torch.inference_mode():
            sel_mem, sel_clips, glob, _, _ = self.decision._streaming_selection(
                None, memory, embeds, emb(qs), 0, [], emb(ans), emb(todo), 100)
            feats = torch.cat((rmodel.clip_projector(sel_clips[0]), rmodel.compress_projector(sel_mem[0])), dim=-2)
            video = torch.cat((feats.reshape(-1, feats.shape[-1]), rmodel.compress_projector(glob[0])), dim=0)
            ids = self.backend._prompt_token_ids(prompt).to(device)
            suffix = self.tokenizer(" Best option: (", add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
            inputs = self.model._interleave_image_features([torch.cat((ids, suffix))], [video])[0].unsqueeze(0)
            logits = self.model(inputs_embeds=inputs.to(self.model.dtype), use_cache=False).logits[0, -1]
        allowed = self._option_token_ids(prompt)
        best = allowed[int(torch.argmax(logits[allowed]))]
        return self.tokenizer.decode([best]).strip()

    def state_info(self) -> dict:
        nbytes = sum(t.numel() * t.element_size() for c in self.clips for t in c.values())
        return {"state_frames": self.frame_count, "state_bytes": int(nbytes)}


class _SessionHooks:
    """StreamingSession perception/decision/reaction hooks without a reset() of their own."""

    def __init__(self, adapter):
        self.adapter = adapter

    def perceive(self, window):
        return self.adapter.perceive(window)

    def should_respond(self, perception, *, window, history) -> bool:
        return False

    def respond(self, perception, *, window, history) -> str:
        raise RuntimeError("Dispider trigger is disabled in RevisitQA evaluation")


def build(model_path: Path, args) -> DispiderAdapter:
    return DispiderAdapter(model_path)
