"""TimeChat-Online-7B (Qwen2.5-VL-7B) with Differential Token Drop, cached incrementally per frame pair.

Upstream (github.com/yaolinli/TimeChat-Online @ 25a1578, copy in external/baselines/timechat_online):
- Benchmark evaluation (`eval/scripts/ovobench.py`, `streamingbench.py`) cuts the video at t_q, re-encodes the
  whole prefix at 1 FPS (min_pixels = max_pixels = 448*448) and runs `generate` with DTD
  (`drop_method="feature"`, `drop_threshold=0.5`, absolute). There is no persistent LLM KV cache.
- DTD (`eval/qwen2_5_vl/modeling_qwen2_5_vl_DTD.py::token_drop`) runs on the merged visual embeddings inside the
  LLM input. Visual tokens are grouped by temporal grid step (2 frames, `temporal_patch_size=2`); a token is
  dropped when its cosine similarity to the same spatial token of the previous step's *full* (undropped)
  embedding exceeds the threshold. The first step is kept whole. Kept tokens keep the M-RoPE position ids
  computed for the undropped sequence.
- The web demo's streaming mode (`demo/web_demo.py`) caches visual embeddings and keep masks per incoming chunk
  and re-runs the LLM over the cached kept tokens plus the question at each query.

This adapter follows the demo's streaming structure with the benchmark settings:
- `observe` resizes each frame like qwen_vl_utils' video path (smart_resize with min = max = args.max_pixels
  (200704 = 448*448), bicubic antialias), groups frames into consecutive pairs, encodes each pair once with the
  Qwen2.5-VL vision tower and applies DTD against the previous pair's full embeddings. Only the kept embeddings
  (and the previous pair's full embeddings needed for the next comparison) are stored. The Qwen2.5-VL vision
  tower attends within one temporal grid step only, so pairwise encoding gives the same features as encoding
  the whole prefix, and DTD only compares neighbouring steps, so the kept set equals the upstream one for the
  same frames.
- `answer` builds `system + user(<video> + prompt) + assistant`, computes M-RoPE ids for the undropped sequence
  (second_per_grid_t = 1.0, as upstream: its processor call passes no fps and falls back to fps = 2.0), keeps the
  positions of the retained tokens, runs one LLM forward without KV reuse and takes the argmax over the option
  letter tokens. Nothing from the question is stored.

Deviations:
- Frame sampling uses the evaluator's 1 FPS clock (frames with timestamp <= t_q) instead of qwen_vl_utils'
  `smart_nframes` over a cut clip, which rounds the frame count to an even number and resamples with linspace.
- With an odd number of visible frames the last frame is paired with itself for the current query (what the
  Qwen2.5-VL processor does for odd-length videos); that provisional step is not stored and is replaced by the
  real pair when the next frame arrives, so the unpaired frame can be encoded twice.
- The demo additionally truncates to the last 8192 visual tokens and uses threshold 0.25; the benchmark
  scripts use neither the cap nor 0.25, and this adapter follows the benchmark scripts (no cap, 0.5).
- The answer is the constrained first token (option letters only) instead of free generation + parsing.
- transformers 4.57 stock Qwen2.5-VL is used for the vision tower, LLM and processor; the checkpoint is a
  standard `Qwen2_5_VLForConditionalGeneration` (DTD is inference-only), and DTD is reimplemented here.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

from scripts.eval.streaming_adapters import StreamingSpec

SPEC = StreamingSpec(
    key="timechat_online_7b",
    checkpoint="TimeChat-Online-7B",
    runtime=".venv-revisitqa-vlm/bin/python",
    description="TimeChat-Online-7B (wyccccc/TimeChatOnline-7B @ 4aa3f6f), DTD feature/0.5 cached per frame pair "
                "(yaolinli/TimeChat-Online 25a1578)",
)

DROP_THRESHOLD = 0.5  # upstream eval default: drop_method="feature", absolute cosine-similarity threshold
SECOND_PER_GRID_T = 1.0  # upstream processor default fps=2.0 -> temporal_patch_size / fps


def _smart_resize(height: int, width: int, min_pixels: int, max_pixels: int, factor: int = 28) -> tuple[int, int]:
    """qwen_vl_utils 0.0.8 smart_resize."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class TimeChatOnlineAdapter:
    def __init__(self, model_path: Path, max_pixels: int, threshold: float = DROP_THRESHOLD):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("TimeChat-Online requires CUDA")
        self.torch = torch
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, attn_implementation=attn, output_loading_info=True,
        )
        if info["missing_keys"] or info["unexpected_keys"] or info.get("mismatched_keys"):
            raise RuntimeError(f"checkpoint mismatch: {info}")
        self._check_against_checkpoint(model_path, model)
        self.model = model.to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.tokenizer = self.processor.tokenizer
        self.video_token_id = model.config.video_token_id
        self.max_pixels = int(max_pixels)
        self.min_pixels = min(448 * 448, self.max_pixels)  # upstream eval: min_pixels = max_pixels = 448*448
        self.threshold = float(threshold)
        self.reset()

    @staticmethod
    def _check_against_checkpoint(model_path: Path, model) -> None:
        """Fail if loaded tensors differ from the checkpoint file (guards against silent random init)."""
        import json

        import torch
        from safetensors import safe_open

        index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
        pairs = {
            "visual.patch_embed.proj.weight": model.model.visual.patch_embed.proj.weight,
            "visual.merger.mlp.2.weight": model.model.visual.merger.mlp[2].weight,
            "model.layers.27.mlp.down_proj.weight": model.model.language_model.layers[27].mlp.down_proj.weight,
            "lm_head.weight": model.lm_head.weight,
        }
        for key, loaded in pairs.items():
            with safe_open(str(Path(model_path) / index[key]), framework="pt") as handle:
                stored = handle.get_tensor(key)
            if not torch.equal(stored.to(loaded.dtype), loaded.detach().cpu()):
                raise RuntimeError(f"{key} does not match the checkpoint")

    def reset(self) -> None:
        self.steps = []  # per temporal step: (kept embeddings [k, D], keep mask [n] bool)
        self.prev_full = None  # full embeddings of the last committed step (DTD reference)
        self.pending = []  # resized frame tensor waiting for its pair
        self.provisional = None  # (frame_count, (kept, mask)) for an unpaired last frame
        self.grid_hw = None
        self.frame_count = 0

    def _frame_tensor(self, image):
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        tensor = TF.pil_to_tensor(image.convert("RGB"))
        h, w = _smart_resize(tensor.shape[1], tensor.shape[2], self.min_pixels, self.max_pixels)
        return TF.resize(tensor, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True).float()

    def _encode(self, pair):
        torch = self.torch
        video = torch.stack(pair)  # [2, C, H, W], 0..255 float like qwen_vl_utils fetch_video
        inputs = self.processor.video_processor(videos=[video], do_resize=False, do_sample_frames=False, return_tensors="pt")
        grid = inputs["video_grid_thw"]
        if int(grid[0, 0]) != 1:
            raise RuntimeError(f"expected one temporal step per frame pair, got grid {grid.tolist()}")
        hw = (int(grid[0, 1]), int(grid[0, 2]))
        if self.grid_hw is None:
            self.grid_hw = hw
        elif hw != self.grid_hw:
            raise RuntimeError(f"frame grid changed within an episode: {self.grid_hw} -> {hw}")
        with torch.inference_mode():
            pixels = inputs["pixel_values_videos"].to("cuda", dtype=self.model.model.visual.dtype)
            return self.model.model.visual(pixels, grid_thw=grid.to("cuda"))  # [h*w/4, D]

    def _drop(self, full, prev_full):
        torch = self.torch
        if prev_full is None:
            mask = torch.ones(full.shape[0], dtype=torch.bool, device=full.device)
        else:
            similarity = torch.nn.functional.cosine_similarity(prev_full, full, dim=1)
            mask = ~(similarity > self.threshold)
        return full[mask], mask

    def observe(self, frames) -> None:
        for frame in frames:
            self.pending.append(self._frame_tensor(frame.image))
            self.frame_count += 1
            if len(self.pending) == 2:
                full = self._encode(self.pending)
                self.steps.append(self._drop(full, self.prev_full))
                self.prev_full = full
                self.pending = []
                self.provisional = None

    def _visual_steps(self):
        steps = list(self.steps)
        if self.pending:
            if self.provisional is None or self.provisional[0] != self.frame_count:
                full = self._encode([self.pending[0], self.pending[0]])
                self.provisional = (self.frame_count, self._drop(full, self.prev_full))
            steps.append(self.provisional[1])
        return steps

    def _option_token_ids(self, prompt: str) -> tuple[list[str], list[int]]:
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
        return labels, ids

    def answer(self, prompt: str, option_count: int) -> str:
        torch = self.torch
        steps = self._visual_steps()
        content = ([{"type": "video"}] if steps else []) + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=True)
        embed = self.model.model.get_input_embeddings()
        if steps:
            prefix, suffix = text.split("<|video_pad|>")
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
            suffix_ids = self.tokenizer(suffix, add_special_tokens=False).input_ids
            h, w = self.grid_hw
            per_step = h * w // 4
            ids = torch.tensor([prefix_ids + [self.video_token_id] * (per_step * len(steps)) + suffix_ids], device="cuda")
            position_ids, _ = self.model.model.get_rope_index(
                ids, None, torch.tensor([[len(steps), h, w]], device="cuda"),
                second_per_grid_ts=torch.tensor([SECOND_PER_GRID_T], device="cuda"),
                attention_mask=torch.ones_like(ids),
            )
            ones = lambda n: torch.ones(n, dtype=torch.bool, device="cuda")  # noqa: E731
            keep = torch.cat([ones(len(prefix_ids))] + [mask for _, mask in steps] + [ones(len(suffix_ids))])
            with torch.inference_mode():
                prefix_emb = embed(torch.tensor(prefix_ids, device="cuda"))
                suffix_emb = embed(torch.tensor(suffix_ids, device="cuda"))
                visual = torch.cat([kept for kept, _ in steps]).to(prefix_emb.dtype)
                inputs_embeds = torch.cat([prefix_emb, visual, suffix_emb])[None]
            position_ids = position_ids[:, :, keep]
            if position_ids.shape[-1] != inputs_embeds.shape[1]:
                raise RuntimeError("position ids and embeddings disagree")
        else:
            ids = torch.tensor([self.tokenizer(text, add_special_tokens=False).input_ids], device="cuda")
            with torch.inference_mode():
                inputs_embeds = embed(ids)
            position_ids = torch.arange(ids.shape[1], device="cuda").view(1, 1, -1).expand(3, 1, -1)
        labels, allowed = self._option_token_ids(prompt)
        with torch.inference_mode():
            hidden = self.model.model.language_model(
                inputs_embeds=inputs_embeds, position_ids=position_ids, use_cache=False,
            ).last_hidden_state[:, -1]
            logits = self.model.lm_head(hidden)[0, allowed]
        self.last_tokens = int(inputs_embeds.shape[1])
        return labels[int(logits.argmax())]

    def state_info(self) -> dict:
        tensors = [kept for kept, _ in self.steps] + ([self.prev_full] if self.prev_full is not None else []) + self.pending
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        kept = sum(int(k.shape[0]) for k, _ in self.steps)
        total = sum(int(m.numel()) for _, m in self.steps)
        return {"state_frames": self.frame_count, "state_bytes": int(nbytes), "kept_visual_tokens": kept,
                "total_visual_tokens": total, "drop_ratio": (1 - kept / total) if total else 0.0}


def build(model_path: Path, args) -> TimeChatOnlineAdapter:
    return TimeChatOnlineAdapter(model_path, args.max_pixels)
