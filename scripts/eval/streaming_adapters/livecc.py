"""LiveCC-7B-Instruct (Qwen2-VL-7B) with a persistent interleaved KV cache.

Upstream: showlab/livecc fdaa91e (`inference.md` LiveCCDemoInfer.live_cc, `data/lmm_dataset.py`
preprocess_conversation_stream, `evaluation/ovobench/distributed_evaluate_ovobench.py`, livecc-utils 0.0.2);
weights chenjoya/LiveCC-7B-Instruct dbe7415c5d2024c3e2d7f898ecbc58fd9fc8da2f.

Matches upstream:
- Stream format: one Qwen2-VL chat turn per clip, user content `Time={start:.1f}-{stop:.1f}s` followed by the clip
  (`<|vision_start|><|video_pad|><|vision_end|>`), stop = last frame time + one frame interval; the first clip holds
  `initial_fps_frames=6` frames and later clips `streaming_fps_frames=2` frames; the system prompt is emitted once
  and every later turn continues the same token sequence and KV cache (upstream `system_prompt_offset` splice).
- Resize: one (h, w) per episode from `smart_resize(factor=28, min_pixels=100*28*28, max_pixels)` as in
  `get_smart_resized_video_reader`, torchvision BICUBIC antialias resize of uint8 frames as in
  `get_smart_resized_clip`; odd clips are padded by repeating the last frame (FRAME_FACTOR=2).
- Answer: OVO-Bench MCQ readout, answer prefix `The answer is:\n` after the assistant header and argmax over the
  option-letter logits with letter ids `tokenizer('\\n' + letter).input_ids[-1]`.

Deviations:
- Clock 1 FPS (evaluator protocol) instead of 2 FPS; clip sizes are kept in frames (6 then 2), so each Qwen2-VL
  temporal token spans 2 s instead of 1 s and the `Time=` text states the true seconds.
- The assistant slot after each clip is filled with ` ...`, the trained "stream continues, nothing to say" target
  (lmm_dataset.py), instead of greedily generating commentary for a default query. This keeps the state purely
  visual, deterministic and query-independent; no user query is attached to the first clip.
- A query that arrives mid-clip commits the visible remainder as a shorter clip (padded to 2 frames) so every
  visible frame is in the state and each frame is encoded exactly once.
- Positions: new tokens get the exact 3D M-RoPE positions of a single full-conversation forward (as in training and
  in the offline evaluators), computed per chunk with `get_rope_index` plus the running offset. The upstream demo
  instead lets `generate` extend later clips with 1D `rope_deltas` positions.
- The question turn (`<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\nThe answer is:\\n`) is
  prefilled on the cache, read out, then cropped away, so no question or answer persists.
- No liger_kernel (not installed; its Qwen2-VL patches are fused RMSNorm/RoPE/SwiGLU with the same math) and SDPA
  instead of flash_attention_2.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from scripts.eval.streaming_adapters import StreamingSpec

SPEC = StreamingSpec(
    key="livecc_7b",
    checkpoint="LiveCC-7B-Instruct",
    runtime=".venv-revisitqa-vlm/bin/python",
    description="LiveCC-7B-Instruct, interleaved Time=/clip turns in one persistent KV cache (showlab/livecc fdaa91e)",
)

INITIAL_FPS_FRAMES = 6
STREAMING_FPS_FRAMES = 2
FRAME_FACTOR = 2
LIVECC_VIDEO_MIN_PIXELS = 100 * 28 * 28  # livecc_utils overrides qwen_vl_utils VIDEO_MIN_PIXELS
STREAM_CONTINUE = " ..."
ANSWER_PREFIX = "The answer is:\n"
ABCD_PREVIOUS_STR = "\n"


class LiveCCAdapter:
    def __init__(self, model_path: Path, max_pixels: int, fps: float):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError("LiveCC requires CUDA")
        self.torch = torch
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"
        model, info = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, attn_implementation=attn, output_loading_info=True,
        )
        bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys")}
        if bad:
            raise RuntimeError(f"LiveCC checkpoint did not load cleanly: {bad}")
        self.model = model.to("cuda").eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path), use_fast=False)
        self.tokenizer = self.processor.tokenizer
        self.video_token = "<|video_pad|>"
        self.merge = self.processor.video_processor.merge_size ** 2
        self.max_pixels = int(max_pixels)
        self.frame_interval = 1.0 / float(fps)
        probe = self.processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "livecc"}]}], tokenize=False)
        offset = probe.index("<|im_start|>user")
        self.system_text = probe[:offset]
        self.system_offset = offset
        self.reset()

    # ---- state ----
    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.seq_len = 0
        self.next_pos = 0
        self.frame_count = 0
        self.clip_count = 0
        self.resized_hw = None
        self._prefill(self.system_text, video=None)

    def _prefill(self, text: str, video):
        """Append `text` (with its video placeholder expanded) to the cache; return last-position logits."""
        torch = self.torch
        pixel_values_videos = video_grid_thw = None
        if video is not None:
            vin = self.processor.video_processor(videos=[video], do_resize=False, return_tensors="pt")
            pixel_values_videos = vin["pixel_values_videos"].to("cuda", torch.bfloat16)
            video_grid_thw = vin["video_grid_thw"]
            n = int(video_grid_thw.prod().item()) // self.merge
            text = text.replace(self.video_token, self.video_token * n, 1)
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        pos, _ = self.model.model.get_rope_index(ids, None, video_grid_thw, None)
        pos = pos + self.next_pos
        length = ids.shape[1]
        with torch.inference_mode():
            out = self.model(
                input_ids=ids.to("cuda"), pixel_values_videos=pixel_values_videos,
                video_grid_thw=None if video_grid_thw is None else video_grid_thw.to("cuda"),
                position_ids=pos.to("cuda"), past_key_values=self.cache, use_cache=True,
                cache_position=torch.arange(self.seq_len, self.seq_len + length, device="cuda"),
                logits_to_keep=1,
            )
        self.seq_len += length
        self.next_pos = int(pos.max().item()) + 1
        return out.logits[0, -1].float()

    # ---- streaming ----
    def _resize(self, frames):
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        from qwen_vl_utils.vision_process import smart_resize

        clip = torch.stack([TF.pil_to_tensor(f.image.convert("RGB")) for f in frames])  # T C H W uint8
        if self.resized_hw is None:
            h, w = clip.shape[2:]
            self.resized_hw = smart_resize(h, w, factor=28, min_pixels=LIVECC_VIDEO_MIN_PIXELS, max_pixels=self.max_pixels)
        clip = TF.resize(clip, list(self.resized_hw), interpolation=InterpolationMode.BICUBIC, antialias=True)
        while clip.shape[0] % FRAME_FACTOR:
            clip = torch.cat([clip, clip[-1:]])
        return clip

    def _commit_clip(self, frames) -> None:
        start = frames[0].timestamp
        stop = frames[-1].timestamp + self.frame_interval
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": f"Time={start:.1f}-{stop:.1f}s"}, {"type": "video", "video": None}]},
            {"role": "assistant", "content": [{"type": "text", "text": STREAM_CONTINUE}]},
        ]
        text = self.processor.apply_chat_template(conversation, tokenize=False)[self.system_offset:]
        self._prefill(text, video=self._resize(frames))
        self.frame_count += len(frames)
        self.clip_count += 1

    def observe(self, frames) -> None:
        frames = list(frames)
        while frames:
            size = INITIAL_FPS_FRAMES if self.clip_count == 0 else STREAMING_FPS_FRAMES
            self._commit_clip(frames[:size])
            frames = frames[size:]

    def _option_token_ids(self, prompt: str) -> tuple[list[str], list[int]]:
        labels = []
        for line in prompt.splitlines():
            match = re.match(r"\s*\(([A-E])\)\s+", line)
            if match and match.group(1) not in labels:
                labels.append(match.group(1))
        return labels, [self.tokenizer(ABCD_PREVIOUS_STR + label).input_ids[-1] for label in labels]

    def answer(self, prompt: str, option_count: int) -> str:
        labels, ids = self._option_token_ids(prompt)
        if not labels:
            raise ValueError("no option labels found in prompt")
        keep = self.seq_len, self.next_pos
        message = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)[self.system_offset:]
        try:
            logits = self._prefill(text + ANSWER_PREFIX, video=None)
        finally:
            self.cache.crop(keep[0])
            self.seq_len, self.next_pos = keep
        return labels[int(logits[ids].argmax().item())]

    def state_info(self) -> dict:
        nbytes = 0
        for layer in self.cache.layers:
            for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if self.torch.is_tensor(t):
                    nbytes += t.numel() * t.element_size()
        return {"state_frames": self.frame_count, "state_bytes": int(nbytes), "state_tokens": self.seq_len}


def build(model_path: Path, args) -> LiveCCAdapter:
    return LiveCCAdapter(model_path, args.max_pixels, args.fps)
