"""VideoLLM-online-8B-v1+ (Llama-3-8B-Instruct + SigLIP-L/384) with a persistent per-frame KV cache.

Upstream: showlab/videollm-online 3da632217258b830a9a594bdabd7a40ae99646dc (`demo/inference.py` LiveInfer,
`models/tokenization_live.py` chat template, `models/vision_live.py`, `models/arguments_live.py` live1+ defaults,
`data/utils.py` ffmpeg_once), vendored in `external/baselines/videollm_online`.
Weights: chenjoya/videollm-online-8b-v1plus b6541f5208f887690856ebb705e2c6c3b71d0095 (PEFT LoRA r=128 + connector),
meta-llama/Meta-Llama-3-8B-Instruct 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2,
google/siglip-large-patch16-384 ce005573a40965dfd21fd937fbdeeebf2439fc35.

Matches upstream:
- Model built by the vendored `build_live_llama` (LiveLlamaConfig with the live1+ frame settings, PeftModel adapter
  applied unmerged as in LiveInfer, connector from the adapter's `modules_to_save`); vision tower is the SigLIP
  `vision_model` in fp32 under fp16 autocast with the upstream `_siglip_vision_encode` (CLS + 3x3 pooled = 10 tokens
  per frame). Loaded adapter tensors are compared with the adapter file and base tensors with the Llama shards.
- Token stream: `<|begin_of_text|>{system}\\n[` + frame, then `,` (frame_token_interval) + frame for every later frame,
  each frame appended to the persistent KV cache in its own forward, exactly the LiveInfer frame path.
- Query turn: `]\\nUser: {prompt}\\nAssistant:` (chat template with add_stream_query_prompt + add_generation_prompt)
  appended after the last frame, as LiveInfer does when a query arrives.
- Frames: upstream ffmpeg preprocessing (bicubic scale of the long side to 384, other side to an even size, centered
  black pad to 384x384), uint8 RGB fed to the vision encoder.

Deviations:
- Clock 1 FPS (evaluator protocol) instead of the 2 FPS used in training and the demo.
- Silence is forced: after every frame upstream emits `,` if P(`,`) >= 0.725 and otherwise starts a response. Here the
  `,` interval path (the trained "stay silent" continuation) is always taken, so the state stays visual-only and
  query-independent; no narration query is attached. The number of frames where upstream would have spoken is
  reported as `would_speak_frames` (a diagnostic, not used for answering).
- Answer: the first answer token is restricted to the option letters (` A`.. with the leading space the template
  puts after `Assistant:`) and the argmax letter is returned; no free generation.
- The query turn is prefilled on the cache, read out, then cropped away; upstream would keep the question, the
  answer and `<|eot_id|>\\n[` in the stream.
- The system prompt is prefilled at reset() instead of together with the first frame (same tokens and positions).
- SDPA instead of flash_attention_2 (flash_attn not installed); PIL bicubic instead of ffmpeg swscale bicubic.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path

from scripts.eval.streaming_adapters import REPO_ROOT, StreamingSpec

SPEC = StreamingSpec(
    key="videollm_online_8b",
    checkpoint="videollm-online-8b-v1plus",
    runtime=".venv-revisitqa-vlm/bin/python",
    description="VideoLLM-online-8B-v1+, per-frame KV cache with forced `,` silence (showlab/videollm-online 3da6322)",
    extra_checkpoints={"llm": "Meta-Llama-3-8B-Instruct", "vision": "siglip-large-patch16-384"},
)

UPSTREAM_ROOT = REPO_ROOT / "external/baselines/videollm_online"
UPSTREAM_PACKAGE = "videollm_online_upstream"
UPSTREAM_VISION_ID = "google/siglip-large-patch16-384"
FRAME_TOKEN_INTERVAL_THRESHOLD = 0.725  # demo/inference.py


def _upstream():
    """Import the vendored upstream `models` package under a private name (the repo has its own `models`)."""
    if UPSTREAM_PACKAGE in sys.modules:
        return sys.modules[UPSTREAM_PACKAGE]
    root = UPSTREAM_ROOT / "models"
    spec = importlib.util.spec_from_file_location(UPSTREAM_PACKAGE, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[UPSTREAM_PACKAGE] = module
    spec.loader.exec_module(module)
    return module


def _live1plus_defaults(up) -> dict:
    cls = up.arguments_live.LiveOnePlusTrainingArguments
    values = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            values[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            values[f.name] = f.default_factory()
    return values


def letterbox(image, resolution: int):
    """ffmpeg_once: scale long side to `resolution` (other side -2, i.e. nearest even), pad centered with black."""
    from PIL import Image

    image = image.convert("RGB")
    w, h = image.size
    if w > h:
        nw, nh = resolution, max(2, int(round(h * resolution / (w * 2))) * 2)
    else:
        nw, nh = max(2, int(round(w * resolution / (h * 2))) * 2), resolution
    image = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
    canvas.paste(image, (((resolution - nw) // 2) & ~1, ((resolution - nh) // 2) & ~1))
    return canvas


class VideoLLMOnlineAdapter:
    def __init__(self, adapter_path: Path, llm_path: Path, vision_path: Path):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("VideoLLM-online requires CUDA")
        self.torch = torch
        self.up = _upstream()
        self.args = _live1plus_defaults(self.up)
        self._load(Path(adapter_path), Path(llm_path), Path(vision_path))
        cfg = self.model.config
        self.hidden_size = cfg.hidden_size
        self.frame_resolution = int(cfg.frame_resolution)
        self.frame_num_tokens = int(cfg.frame_num_tokens)
        self.interval_id = int(cfg.frame_token_interval_id)
        tok = self.tokenizer
        start = tok.apply_chat_template([{"role": "system", "content": self.args["system_prompt"]}], add_stream_prompt=True, tokenize=False)
        self.start_ids = tok(start, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        if self.start_ids[0, 0].item() != tok.bos_token_id:
            raise RuntimeError("system prompt does not start with BOS")
        self.reset()

    def _load(self, adapter_path: Path, llm_path: Path, vision_path: Path) -> None:
        import importlib.util as iu
        from functools import partial

        torch = self.torch
        from transformers import AutoModel

        args = self.args
        attn = "flash_attention_2" if iu.find_spec("flash_attn") else "sdpa"
        model, tokenizer = self.up.build_model_and_tokenizer(
            is_training=False, set_vision_inside=False, resume_from_checkpoint=str(adapter_path),
            llm_pretrained=str(llm_path), attn_implementation=attn, torch_dtype=torch.bfloat16,
            vision_pretrained=UPSTREAM_VISION_ID, frame_resolution=args["frame_resolution"],
            frame_token_cls=args["frame_token_cls"], frame_token_pooled=args["frame_token_pooled"],
            frame_num_tokens=args["frame_num_tokens"], frame_token_interval=args["frame_token_interval"],
        )
        self._check_weights(model, adapter_path, llm_path)
        base = model.get_base_model()
        # set_vision_inside() resolves the hub id; load the same tower from the local copy instead.
        vision = AutoModel.from_pretrained(str(vision_path), torch_dtype=torch.float32).vision_model
        base.vision_encoder = vision
        base.vision_encode = partial(
            self.up.vision_live._siglip_vision_encode,
            frame_token_cls=base.config.frame_token_cls, frame_token_pooled=base.config.frame_token_pooled,
        )
        self.model = model.to("cuda").eval()
        self.tokenizer = tokenizer

    def _check_weights(self, model, adapter_path: Path, llm_path: Path) -> None:
        """Every adapter tensor (LoRA + connector) and sampled base tensors must equal the checkpoint files."""
        from peft import get_peft_model_state_dict
        from safetensors import safe_open

        torch = self.torch
        loaded = get_peft_model_state_dict(model)
        with safe_open(str(adapter_path / "adapter_model.safetensors"), framework="pt") as handle:
            stored_keys = set(handle.keys())
            if set(loaded) != stored_keys:
                raise RuntimeError(f"adapter keys differ: {sorted(set(loaded) ^ stored_keys)[:5]}")
            for key in stored_keys:
                if not torch.equal(handle.get_tensor(key).to(loaded[key].dtype), loaded[key].detach().cpu()):
                    raise RuntimeError(f"adapter tensor {key} does not match the checkpoint")
        if not any("connector" in k for k in stored_keys):
            raise RuntimeError("adapter has no connector weights")
        index = json.loads((llm_path / "model.safetensors.index.json").read_text())["weight_map"]
        base = model.get_base_model()
        params = dict(base.named_parameters())
        last = base.config.num_hidden_layers - 1
        for key in ("model.embed_tokens.weight", f"model.layers.{last}.mlp.down_proj.base_layer.weight", "lm_head.base_layer.weight"):
            ckpt_key = key.replace(".base_layer", "")
            if key not in params:
                key = ckpt_key
            with safe_open(str(llm_path / index[ckpt_key]), framework="pt") as handle:
                if not torch.equal(handle.get_tensor(ckpt_key).to(params[key].dtype), params[key].detach().cpu()):
                    raise RuntimeError(f"base tensor {ckpt_key} does not match the checkpoint")

    # ---- state ----
    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache()
        self.frame_count = 0
        self.would_speak = 0
        self._forward(self.model.get_input_embeddings()(self.start_ids))

    def _forward(self, embeds):
        with self.torch.inference_mode():
            out = self.model(inputs_embeds=embeds, past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values
        return out.logits[0, -1].float()

    def seq_len(self) -> int:
        return int(self.cache.get_seq_length())

    # ---- streaming ----
    def observe(self, frames) -> None:
        import numpy as np

        torch = self.torch
        frames = list(frames)
        if not frames:
            return
        pixels = torch.stack([
            torch.from_numpy(np.array(letterbox(f.image, self.frame_resolution))).permute(2, 0, 1)
            for f in frames
        ]).to("cuda")
        with torch.inference_mode():
            embeds = self.model.visual_embed(pixels).split(self.frame_num_tokens)
            interval = self.model.get_input_embeddings()(torch.tensor([[self.interval_id]], device="cuda"))
        for frame_embeds in embeds:
            parts = [frame_embeds.view(1, -1, self.hidden_size)]
            if self.frame_count > 0:  # LiveInfer: last_ids is the `,` the model chose (here: forced) after the previous frame
                parts.insert(0, interval.view(1, -1, self.hidden_size))
            logits = self._forward(torch.cat(parts, dim=1))
            if logits.softmax(-1)[self.interval_id].item() < FRAME_TOKEN_INTERVAL_THRESHOLD:
                self.would_speak += 1
            self.frame_count += 1

    def _option_token_ids(self, prompt: str) -> tuple[list[str], list[int]]:
        labels = []
        for line in prompt.splitlines():
            match = re.match(r"\s*\(([A-E])\)\s+", line)
            if match and match.group(1) not in labels:
                labels.append(match.group(1))
        ids = []
        for label in labels:
            encoded = self.tokenizer(" " + label, add_special_tokens=False).input_ids
            if len(encoded) != 1:
                raise ValueError(f"option label {label!r} is not a single token")
            ids.append(encoded[0])
        return labels, ids

    def answer(self, prompt: str, option_count: int) -> str:
        labels, ids = self._option_token_ids(prompt)
        if not labels:
            raise ValueError("no option labels found in prompt")
        if self.frame_count == 0:
            raise RuntimeError("no frame observed before answering")
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_stream_query_prompt=True, add_generation_prompt=True, tokenize=False,
        )
        query = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
        keep = self.seq_len()
        try:
            logits = self._forward(self.model.get_input_embeddings()(query))
        finally:
            self.cache.crop(keep)
        if self.seq_len() != keep:
            raise RuntimeError("cache was not restored after answering")
        return labels[int(logits[ids].argmax().item())]

    def state_info(self) -> dict:
        nbytes = 0
        for layer in self.cache.layers:
            for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if self.torch.is_tensor(t):
                    nbytes += t.numel() * t.element_size()
        return {"state_frames": self.frame_count, "state_bytes": int(nbytes), "state_tokens": self.seq_len(),
                "would_speak_frames": self.would_speak}


def build(model_path: Path, args) -> VideoLLMOnlineAdapter:
    root = Path(args.checkpoint_root)
    return VideoLLMOnlineAdapter(model_path, root / SPEC.extra_checkpoints["llm"], root / SPEC.extra_checkpoints["vision"])
