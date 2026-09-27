"""Inference adapters for the heterogeneous STORM-Bench model set."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.eval.storm_streaming import (
    CLOCK_FPS,
    frames_to_images,
    frames_to_timestamps,
    keep_last_frames,
    pad_images_to_multiple,
)


MCQ_INSTRUCTION = (
    "Answer the following four-choice question based on the video. Output only the "
    "letter of the correct option (A, B, C, or D); do not explain.\n\n"
)

NATIVE_FRAME_CAPS = {
    "qwen": 768,
    "molmo2": 384,
    "internvl": 240,
    "internvideo": 512,
    "minicpm": 180,
    "eagle": 256,
    "videollama3": 128,
    "llava_next_video": 32,
    "glm4v": 240,
}


def _model_device(model):
    try:
        return model.device
    except AttributeError:
        return next(model.parameters()).device


def _prepare_protocol_frames(
    frames: Sequence,
    timestamps: Sequence[float] | None,
    cap: int,
    multiple: int,
    clock_fps: float = CLOCK_FPS,
) -> tuple[list, list[float], bool]:
    images = frames_to_images(frames)
    times = list(timestamps) if timestamps is not None else frames_to_timestamps(frames, clock_fps)
    if len(times) != len(images):
        times = frames_to_timestamps(images, clock_fps)
    images, times, capped = keep_last_frames(images, times, cap)
    images, times = pad_images_to_multiple(images, times, multiple)
    return images, times, capped


# HuggingFace VideoMetadata accepts these constructor fields. `timestamps` is a
# derived property (`frame_idx / fps`) and must not be passed through **kwargs.
_VIDEO_METADATA_FIELDS = {
    "total_num_frames",
    "fps",
    "width",
    "height",
    "duration",
    "video_backend",
    "frames_indices",
}


def _clock_video_metadata(
    num_frames: int,
    fps: float,
    timestamps: Sequence[float] | None = None,
) -> dict:
    times = [
        float(index) / max(float(fps), 1e-6)
        for index in range(num_frames)
    ] if timestamps is None else [float(value) for value in timestamps]
    if len(times) != num_frames:
        raise ValueError(f"Expected {num_frames} timestamps, got {len(times)}")
    frame_indices = [int(round(value * fps)) for value in times]
    payload = {
        "total_num_frames": max(int(num_frames), max(frame_indices, default=-1) + 1),
        "fps": float(fps),
        "frames_indices": frame_indices,
        "video_backend": "frames",
    }
    return {key: value for key, value in payload.items() if key in _VIDEO_METADATA_FIELDS}


def _images_to_numpy(images: Sequence) -> np.ndarray:
    return np.stack([np.asarray(image.convert("RGB")) for image in images], axis=0)


def _require_flash_attention_2(purpose: str) -> str:
    """Official VideoLLaMA3 inference uses FlashAttention-2, including the vision encoder."""
    if importlib.util.find_spec("flash_attn") is None:
        raise ImportError(
            f"{purpose} requires the flash_attn package. Official VideoLLaMA3 loads with "
            'attn_implementation="flash_attention_2" so the vision encoder can call '
            "flash_attn_varlen_func; SDPA materializes a dense [S, S] mask and OOMs on "
            "128-frame videos. Install it into .venv-storm-videollama3, for example:\n"
            "  python -m pip install 'flash-attn==2.8.3.post1+cu.12.8.torch.2.11' "
            "--index-url https://wheels.astral.sh/simple/cu128/"
        )
    try:
        from transformers.utils import is_flash_attn_2_available
    except ImportError:
        return "flash_attention_2"
    if not is_flash_attn_2_available():
        raise ImportError(
            f"{purpose} found flash_attn, but transformers reports FlashAttention-2 is "
            "unavailable. Rebuild/install a CUDA-matching flash-attn wheel."
        )
    return "flash_attention_2"


def _assert_videollama3_flash_vision(model) -> None:
    vision = model.get_vision_encoder()
    attn = vision.encoder.layers[0].self_attn
    name = type(attn).__name__
    impl = getattr(getattr(vision, "config", None), "_attn_implementation", None)
    if name != "VisionFlashAttention2":
        raise RuntimeError(
            "VideoLLaMA3 vision encoder is not using official FlashAttention-2 "
            f"(got {name}, config._attn_implementation={impl}). "
            "The vision path must be VisionFlashAttention2 / flash_attn_varlen_func; "
            "VisionSdpaAttention builds a full-video [S, S] mask and OOMs."
        )


class QwenAdapter:
    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        attn = (
            "flash_attention_2"
            if torch.cuda.is_available() and importlib.util.find_spec("flash_attn")
            else "sdpa"
        )
        self.torch = torch
        config = AutoConfig.from_pretrained(model_path)
        self.model_type = config.model_type
        if self.model_type == "qwen3_5":
            from transformers import AutoModelForMultimodalLM
            model_class = AutoModelForMultimodalLM
        else:
            model_class = AutoModelForImageTextToText
        self.model = model_class.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn,
        ).eval()
        processor_kwargs = {} if self.model_type in {"qwen3_vl", "qwen3_5"} else {"use_fast": False}
        self.processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        self.max_frames = NATIVE_FRAME_CAPS["qwen"]
        self.fps = fps
        self.max_pixels = max_pixels

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        from qwen_vl_utils import process_vision_info, qwen3_video_metadata

        images, times, _ = _prepare_protocol_frames(frames, timestamps, self.max_frames, 2, self.fps)
        content_video = {
            "type": "video",
            "video": images,
            "max_frames": len(images),
            "max_pixels": self.max_pixels,
        }
        messages = [{"role": "user", "content": [content_video, {"type": "text", "text": MCQ_INSTRUCTION + question}]}]
        if self.model_type == "qwen3_5":
            template_kwargs = dict(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={
                    "do_sample_frames": False,
                    "fps": self.fps,
                    "video_metadata": [_clock_video_metadata(len(images), self.fps, times)],
                },
            )
            try:
                inputs = self.processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
            except TypeError:
                template_kwargs.pop("processor_kwargs", None)
                try:
                    inputs = self.processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
                except TypeError:
                    inputs = self.processor.apply_chat_template(messages, **template_kwargs)
            inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
            generated = generated[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
        except TypeError:
            text = self.processor.apply_chat_template(messages, **template_kwargs)
        image_inputs, video_inputs = process_vision_info(messages)
        video_kwargs = {"do_sample_frames": False, "fps": [self.fps]}
        if self.model_type == "qwen3_vl":
            frame_indices = [int(round(value * self.fps)) for value in times]
            video_kwargs["video_metadata"] = [qwen3_video_metadata(
                {**content_video, "fps": self.fps},
                frame_indices=frame_indices,
                total_num_frames=max(frame_indices, default=-1) + 1,
            )]
            video_kwargs["video_metadata"][0]["fps"] = float(self.fps)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def answer_text(self, question: str) -> str:
        """Closed-book MCQ: same prompt and options, no video or image tokens."""
        messages = [{"role": "user", "content": [{"type": "text", "text": MCQ_INSTRUCTION + question}]}]
        if self.model_type == "qwen3_5":
            template_kwargs = dict(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            try:
                inputs = self.processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
            except TypeError:
                inputs = self.processor.apply_chat_template(messages, **template_kwargs)
            inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
            generated = generated[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.processor.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
        except TypeError:
            text = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


class Molmo2Adapter:
    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.max_frames = NATIVE_FRAME_CAPS["molmo2"]
        self.fps = fps

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        images, times, _ = _prepare_protocol_frames(frames, timestamps, self.max_frames, 1, self.fps)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": MCQ_INSTRUCTION + question},
            {"type": "video", "video": images},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=text,
            videos=[_images_to_numpy(images)],
            padding=True,
            return_tensors="pt",
            do_sample_frames=False,
            video_metadata=[_clock_video_metadata(len(images), self.fps, times)],
        )
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def answer_text(self, question: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": MCQ_INSTRUCTION + question},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, padding=True, return_tensors="pt")
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


def _build_transform(input_size: int = 448):
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose([
        transforms.Lambda(lambda image: image.convert("RGB")),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class OpenGVLabAdapter:
    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=bool(importlib.util.find_spec("flash_attn")),
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.frame_multiple = int(getattr(self.model, "local_num_frames", 1) or 1)
        cap_key = "internvideo" if self.frame_multiple > 1 else "internvl"
        self.max_frames = NATIVE_FRAME_CAPS[cap_key]
        self.fps = fps

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        images, times, _ = _prepare_protocol_frames(
            frames, timestamps, self.max_frames, max(self.frame_multiple, 1), self.fps
        )
        transform = _build_transform()
        pixel_values = self.torch.stack([transform(image) for image in images])
        pixel_values = pixel_values.to(device=_model_device(self.model), dtype=self.torch.bfloat16)
        num_patches = [1] * len(images)
        prefix = "".join(
            f"Frame{i + 1} ({time:.1f}s): <image>\n"
            for i, time in enumerate(times)
        )
        generation_config = dict(do_sample=False, max_new_tokens=8, num_beams=1)
        with self.torch.inference_mode():
            output = self.model.chat(
                self.tokenizer,
                pixel_values,
                prefix + MCQ_INSTRUCTION + question,
                generation_config,
                num_patches_list=num_patches,
                history=None,
                return_history=False,
            )
        return output[0] if isinstance(output, tuple) else str(output)

    def answer_text(self, question: str) -> str:
        """Closed-book MCQ: same prompt and options, no video or image tokens."""
        generation_config = dict(do_sample=False, max_new_tokens=8, num_beams=1)
        with self.torch.inference_mode():
            output = self.model.chat(
                self.tokenizer,
                None,
                MCQ_INSTRUCTION + question,
                generation_config,
                num_patches_list=[],
                history=None,
                return_history=False,
            )
        return output[0] if isinstance(output, tuple) else str(output)


class MiniCPMAdapter:
    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.max_frames = NATIVE_FRAME_CAPS["minicpm"]
        self.fps = fps

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        images, times, _ = _prepare_protocol_frames(frames, timestamps, self.max_frames, 1, self.fps)
        packing = max(1, min(3, math.ceil(len(images) / 180)))
        temporal_ids = np.rint(np.asarray(times, dtype=np.float64) / 0.1).astype(np.int32).tolist()
        grouped_ids = [temporal_ids[start:start + packing] for start in range(0, len(temporal_ids), packing)]
        messages = [{"role": "user", "content": images + [MCQ_INSTRUCTION + question]}]
        with self.torch.inference_mode():
            return str(self.model.chat(
                msgs=messages,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_slice_nums=1,
                temporal_ids=grouped_ids,
                max_new_tokens=8,
                sampling=False,
            ))

    def answer_text(self, question: str) -> str:
        """Closed-book MCQ: same prompt and options, no video or image tokens."""
        messages = [{"role": "user", "content": MCQ_INSTRUCTION + question}]
        with self.torch.inference_mode():
            return str(self.model.chat(
                msgs=messages,
                tokenizer=self.tokenizer,
                use_image_id=False,
                max_new_tokens=8,
                sampling=False,
            ))


class EagleAdapter:
    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        attn = _require_flash_attention_2("Eagle2.5-8B")
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation=attn,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )
        self.processor.tokenizer.padding_side = "left"
        self.max_frames = NATIVE_FRAME_CAPS["eagle"]
        self.fps = fps

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        images, _, _ = _prepare_protocol_frames(frames, timestamps, self.max_frames, 2, self.fps)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": images},
            {"type": "text", "text": MCQ_INSTRUCTION + question},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = self.processor.process_vision_info(
            messages, return_video_kwargs=True
        )
        if isinstance(video_kwargs, dict):
            video_kwargs["do_sample_frames"] = False
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            videos_kwargs=video_kwargs,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def answer_text(self, question: str) -> str:
        """Closed-book MCQ: same prompt and options, no video or image tokens."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": MCQ_INSTRUCTION + question},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


class VideoLLaMA3Adapter:
    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        attn = _require_flash_attention_2("VideoLLaMA3-7B")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
        ).eval()
        _assert_videollama3_flash_vision(self.model)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.max_frames = NATIVE_FRAME_CAPS["videollama3"]
        self.fps = fps

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        images, times, _ = _prepare_protocol_frames(frames, timestamps, self.max_frames, 2, self.fps)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {
                    "type": "video",
                    "video": images,
                    "num_frames": len(images),
                    "timestamps": times,
                },
                {"type": "text", "text": MCQ_INSTRUCTION + question},
            ]},
        ]
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(_model_device(self.model)) if isinstance(value, self.torch.Tensor) else value
            for key, value in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.torch.bfloat16)
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def answer_text(self, question: str) -> str:
        """Closed-book MCQ: same prompt and options, no video or image tokens."""
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": MCQ_INSTRUCTION + question},
            ]},
        ]
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(_model_device(self.model)) if isinstance(value, self.torch.Tensor) else value
            for key, value in inputs.items()
        }
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


class LlavaNextVideoAdapter:
    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoProcessor, LlavaNextVideoForConditionalGeneration

        self.torch = torch
        attn = (
            "flash_attention_2"
            if torch.cuda.is_available() and importlib.util.find_spec("flash_attn")
            else "sdpa"
        )
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation=attn,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.max_frames = NATIVE_FRAME_CAPS["llava_next_video"]
        self.fps = fps

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        images, _, _ = _prepare_protocol_frames(frames, timestamps, self.max_frames, 1, self.fps)
        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": MCQ_INSTRUCTION + question},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt, videos=_images_to_numpy(images), return_tensors="pt"
        )
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def answer_text(self, question: str) -> str:
        """Closed-book MCQ: same prompt and options, no video or image tokens."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": MCQ_INSTRUCTION + question},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, return_tensors="pt")
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        generated = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


class Glm4vAdapter:
    THINKING_BUDGET = 8192
    FORCE_ANSWER_BUDGET = 64

    def __init__(self, model_path: Path, fps: float, max_pixels: int):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.torch = torch
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.max_frames = NATIVE_FRAME_CAPS["glm4v"]
        self.fps = fps
        self.max_pixels = max_pixels

    def answer(self, frames: Sequence, question: str, timestamps: Sequence[float] | None = None) -> str:
        images, times, _ = _prepare_protocol_frames(frames, timestamps, self.max_frames, 2, self.fps)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": images},
            {"type": "text", "text": MCQ_INSTRUCTION + question},
        ]}]
        template_kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "do_sample_frames": False,
                "fps": self.fps,
                "video_metadata": [_clock_video_metadata(len(images), self.fps, times)],
                "size": {
                    "shortest_edge": 12544,
                    "longest_edge": self.max_pixels * len(images),
                },
            },
        )
        inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.THINKING_BUDGET,
                do_sample=False,
            )
        output_ids = generated[0, input_length:]

        tokenizer = self.processor.tokenizer
        think_end = tokenizer.convert_tokens_to_ids("</think>")
        answer_start = tokenizer.convert_tokens_to_ids("<answer>")
        answer_end = tokenizer.convert_tokens_to_ids("</answer>")
        output_tokens = output_ids.tolist()
        needs_forced_answer = answer_end not in output_tokens
        if needs_forced_answer:
            suffix = []
            if think_end not in output_tokens:
                suffix.append(think_end)
            if answer_start not in output_tokens:
                suffix.append(answer_start)
            suffix_ids = self.torch.tensor(
                suffix, device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype
            )
            force_input_ids = self.torch.cat(
                [inputs["input_ids"], output_ids.unsqueeze(0), suffix_ids.unsqueeze(0)], dim=1
            )
            force_inputs = {
                "input_ids": force_input_ids,
                "attention_mask": self.torch.ones_like(force_input_ids),
            }
            for key in ("pixel_values", "video_metadata"):
                if key in inputs:
                    force_inputs[key] = inputs[key]
            with self.torch.inference_mode():
                forced = self.model.generate(
                    **force_inputs,
                    max_new_tokens=self.FORCE_ANSWER_BUDGET,
                    do_sample=False,
                )
            forced_output = forced[0, force_input_ids.shape[1]:]
            output_ids = self.torch.cat([output_ids, suffix_ids, forced_output])

        return self.processor.decode(output_ids, skip_special_tokens=False).strip()

    def answer_text(self, question: str) -> str:
        """Closed-book MCQ: same prompt and options, no video or image tokens."""
        messages = [{"role": "user", "content": [
            {"type": "text", "text": MCQ_INSTRUCTION + question},
        ]}]
        template_kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self.processor.apply_chat_template(messages, **template_kwargs)
        inputs = {key: value.to(_model_device(self.model)) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.THINKING_BUDGET,
                do_sample=False,
            )
        output_ids = generated[0, input_length:]

        tokenizer = self.processor.tokenizer
        think_end = tokenizer.convert_tokens_to_ids("</think>")
        answer_start = tokenizer.convert_tokens_to_ids("<answer>")
        answer_end = tokenizer.convert_tokens_to_ids("</answer>")
        output_tokens = output_ids.tolist()
        needs_forced_answer = answer_end not in output_tokens
        if needs_forced_answer:
            suffix = []
            if think_end not in output_tokens:
                suffix.append(think_end)
            if answer_start not in output_tokens:
                suffix.append(answer_start)
            suffix_ids = self.torch.tensor(
                suffix, device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype
            )
            force_input_ids = self.torch.cat(
                [inputs["input_ids"], output_ids.unsqueeze(0), suffix_ids.unsqueeze(0)], dim=1
            )
            force_inputs = {
                "input_ids": force_input_ids,
                "attention_mask": self.torch.ones_like(force_input_ids),
            }
            for key in ("pixel_values", "video_metadata"):
                if key in inputs:
                    force_inputs[key] = inputs[key]
            with self.torch.inference_mode():
                forced = self.model.generate(
                    **force_inputs,
                    max_new_tokens=self.FORCE_ANSWER_BUDGET,
                    do_sample=False,
                )
            forced_output = forced[0, force_input_ids.shape[1]:]
            output_ids = self.torch.cat([output_ids, suffix_ids, forced_output])

        return self.processor.decode(output_ids, skip_special_tokens=False).strip()


def load_adapter(spec, model_path: Path, fps: float, max_pixels: int):
    if spec.family == "qwen":
        return QwenAdapter(model_path, fps, max_pixels)
    if spec.family == "molmo2":
        return Molmo2Adapter(model_path, fps, max_pixels)
    if spec.family in {"internvl", "internvideo"}:
        return OpenGVLabAdapter(model_path, fps, max_pixels)
    if spec.family == "minicpm":
        return MiniCPMAdapter(model_path, fps, max_pixels)
    if spec.family == "eagle":
        return EagleAdapter(model_path, fps, max_pixels)
    if spec.family == "videollama3":
        return VideoLLaMA3Adapter(model_path, fps, max_pixels)
    if spec.family == "llava_next_video":
        return LlavaNextVideoAdapter(model_path, fps, max_pixels)
    if spec.family == "glm4v":
        return Glm4vAdapter(model_path, fps, max_pixels)
    raise ValueError(f"Unsupported adapter family: {spec.family}")
