"""Flash-VStream (Qwen2-VL-7B) with incremental Flash Memory.

Upstream realtime mode embeds each new clip into Flash Memory with `embed_new_video_clip` and answers from
the memory. The earlier PReM evaluator cleared the memory and re-embedded the whole prefix at every query;
here each frame is embedded once, when it first becomes visible, and the memory is cleared only at an
episode boundary. Frames are resized per image with the same `fetch_image` path the upstream list-of-frames
video loader uses, and embedded one frame per clip (upstream `chunk_frames=1`).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from scripts.eval.streaming_adapters import StreamingSpec

SPEC = StreamingSpec(
    key="flash_vstream_7b",
    checkpoint="Flash-VStream-Qwen-7b",
    runtime=".venv-revisitqa-vlm/bin/python",
    description="Flash-VStream-Qwen-7B, Flash Memory updated one frame at a time (IVGSZ/Flash-VStream 8f6bde2)",
    # Qwen2-VL-7B-Instruct tokenizer/preprocessor files (huggingface Qwen/Qwen2-VL-7B-Instruct); the preprocessor
    # config shipped with the Flash-VStream checkpoint uses a format current transformers rejects.
    extra_checkpoints={"processor": "Qwen2-VL-7B-Instruct-processor"},
)


def _check_against_checkpoint(model_path: Path, tensors: dict) -> None:
    """Fail if a loaded tensor differs from the checkpoint file (guards the tower re-wiring against random init)."""
    import json

    import torch
    from safetensors import safe_open

    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())["weight_map"]
    for key, loaded in tensors.items():
        with safe_open(str(Path(model_path) / index[key]), framework="pt") as handle:
            stored = handle.get_tensor(key)
        if not torch.equal(stored.to(loaded.dtype), loaded.detach().cpu()):
            raise RuntimeError(f"{key} does not match the checkpoint")


class FlashVStreamAdapter:
    def __init__(self, model_path: Path, processor_path: Path, max_pixels: int):
        import torch

        from models.flash_memory_constants import DEFAULT_FLASH_MEMORY_CONFIG
        from models.vstream_qwen2vl_model import FlashVStreamQwen2VLConfig
        from models.vstream_qwen2vl_processor import FlashVStreamQwen2VLProcessor
        from models.vstream_qwen2vl_realtime import FlashVStreamQwen2VLModel

        if not torch.cuda.is_available():
            raise RuntimeError("Flash-VStream requires CUDA")
        self.torch = torch
        config = FlashVStreamQwen2VLConfig.from_pretrained(str(model_path), trust_remote_code=True)
        config.vision_config.flash_memory_config = DEFAULT_FLASH_MEMORY_CONFIG.copy()
        # Without flash_attn use SDPA: the text model then builds its own causal mask (the eager path relies on a
        # mask helper that newer transformers removed; a local copy is kept in vstream_qwen2vl_realtime.py).
        attn = "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"

        model = FlashVStreamQwen2VLModel.from_pretrained(
            str(model_path), config=config, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation=attn,
        )
        # transformers >= 4.52 maps checkpoint keys `visual.*` into the inner standard tower `model.model.visual`
        # (and `model.visual` is a property returning it), while Flash-VStream registers its own streaming tower
        # as the submodule `visual`, which then stays on the meta device. Both towers have identical parameters,
        # so build the Flash tower normally (buffers such as rotary inv_freq initialized) and load the inner
        # tower's weights into it strictly, then use it in both slots.
        inner = model.model.visual
        tower_cls = type(model._modules["visual"])
        tower = tower_cls._from_config(config.vision_config, torch_dtype=torch.bfloat16)
        tower.load_state_dict(inner.state_dict(), strict=True)
        model._modules["visual"] = tower
        model.model.visual = tower
        still_meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        if still_meta:
            raise RuntimeError(f"unloaded tensors remain: {still_meta[:5]}")
        _check_against_checkpoint(model_path, {
            "visual.patch_embed.proj.weight": tower.patch_embed.proj.weight,
            "visual.blocks.31.mlp.fc2.weight": tower.blocks[31].mlp.fc2.weight,
            "lm_head.weight": model.lm_head.weight,
        })
        self.model = model.to("cuda").eval()
        self.model.use_video_streaming_mode = True
        self.processor = FlashVStreamQwen2VLProcessor.from_pretrained(str(processor_path))
        self.memory_config = config.vision_config.flash_memory_config
        self.max_pixels = int(max_pixels)
        self.reset()

    def reset(self) -> None:
        self.model.video_embedding_memory = []
        self.frame_count = 0

    def observe(self, frames) -> None:
        from qwen_vl_utils.vision_process import VIDEO_MIN_PIXELS, fetch_image

        for frame in frames:
            image = fetch_image({"image": frame.image, "min_pixels": VIDEO_MIN_PIXELS, "max_pixels": self.max_pixels})
            inputs = self.processor.image_processor(
                images=None, videos=[image], return_tensors="pt",
                additional_pool_size=self.memory_config["flash_memory_temporal_poolsize"],
            )
            with self.torch.inference_mode():
                self.model.embed_new_video_clip(**inputs, start_idx=self.frame_count)
            self.frame_count += 1

    def _option_token_ids(self, prompt: str) -> list[int]:
        labels = []
        for line in prompt.splitlines():
            match = re.match(r"\s*\(([A-E])\)\s+", line)
            if match and match.group(1) not in labels:
                labels.append(match.group(1))
        ids = []
        for label in labels:
            encoded = self.processor.tokenizer(label, add_special_tokens=False).input_ids
            if len(encoded) != 1:
                raise ValueError(f"option label {label!r} is not a single token")
            ids.append(encoded[0])
        return ids

    def answer(self, prompt: str, option_count: int) -> str:
        memory = self.model.get_video_embedding_memory_cuda_list()
        if not memory:
            raise RuntimeError("Flash memory is empty before answering")
        messages = [{"role": "user", "content": [{"type": "text", "text": "<|vision_start|><|video_pad|><|vision_end|>" + prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "Best option: ("
        tem_thw, spa_thw = memory[1], memory[5]
        video_tokens = int((tem_thw.prod() // 4 + spa_thw.prod() // 4).item())
        inputs = self.processor(
            text=[text], images=None, videos=None, padding=True, return_tensors="pt",
            flash_memory_config=self.memory_config, dummy_video_tokens=video_tokens * 4,
        ).to("cuda")
        allowed = self._option_token_ids(prompt)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=1, do_sample=False, use_cache=False,
                prefix_allowed_tokens_fn=lambda _b, _ids: allowed,
            )
        return self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

    def state_info(self) -> dict:
        memory = self.model.video_embedding_memory
        nbytes = sum(t.numel() * t.element_size() for t in memory if self.torch.is_tensor(t))
        return {"state_frames": self.frame_count, "state_bytes": int(nbytes)}


def build(model_path: Path, args) -> FlashVStreamAdapter:
    processor = Path(args.checkpoint_root) / SPEC.extra_checkpoints["processor"]
    return FlashVStreamAdapter(model_path, processor, args.max_pixels)
