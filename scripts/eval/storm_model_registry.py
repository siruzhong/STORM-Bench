"""Model registry for the STORM-Bench cross-model benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StormModelSpec:
    key: str
    repo_id: str
    local_name: str
    family: str
    runtime: str
    trust_remote_code: bool = False

    def local_path(self, checkpoint_root: str | Path = "ckpt") -> Path:
        return Path(checkpoint_root) / self.local_name


MODEL_SPECS = {
    spec.key: spec
    for spec in (
        StormModelSpec(
            "qwen25_vl_3b",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "Qwen2.5-VL-3B-Instruct",
            "qwen",
            "vlm",
        ),
        StormModelSpec(
            "qwen25_vl_7b",
            "Qwen/Qwen2.5-VL-7B-Instruct",
            "Qwen2.5-VL-7B-Instruct",
            "qwen",
            "vlm",
        ),
        StormModelSpec(
            "qwen3_vl_4b",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen3-VL-4B-Instruct",
            "qwen",
            "vlm",
        ),
        StormModelSpec(
            "qwen3_vl_8b",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen3-VL-8B-Instruct",
            "qwen",
            "vlm",
        ),
        StormModelSpec(
            "qwen35_4b",
            "Qwen/Qwen3.5-4B",
            "Qwen3.5-4B",
            "qwen",
            "modern",
        ),
        StormModelSpec(
            "qwen35_9b",
            "Qwen/Qwen3.5-9B",
            "Qwen3.5-9B",
            "qwen",
            "modern",
        ),
        StormModelSpec(
            "internvl35_8b",
            "OpenGVLab/InternVL3_5-8B",
            "InternVL3_5-8B",
            "internvl",
            "vlm",
            trust_remote_code=True,
        ),
        StormModelSpec(
            "molmo2_8b",
            "allenai/Molmo2-8B",
            "Molmo2-8B",
            "molmo2",
            "vlm",
            trust_remote_code=True,
        ),
        StormModelSpec(
            "minicpm_v45",
            "openbmb/MiniCPM-V-4_5",
            "MiniCPM-V-4_5",
            "minicpm",
            "vlm",
            trust_remote_code=True,
        ),
        StormModelSpec(
            "internvideo25_8b",
            "OpenGVLab/InternVideo2_5_Chat_8B",
            "InternVideo2_5_Chat_8B",
            "internvideo",
            "legacy",
            trust_remote_code=True,
        ),
        StormModelSpec(
            "eagle25_8b",
            "nvidia/Eagle2.5-8B",
            "Eagle2.5-8B",
            "eagle",
            "vlm",
            trust_remote_code=True,
        ),
        StormModelSpec(
            "videollama3_7b",
            "DAMO-NLP-SG/VideoLLaMA3-7B",
            "VideoLLaMA3-7B",
            "videollama3",
            "videollama3",
            trust_remote_code=True,
        ),
        StormModelSpec(
            "llava_next_video_7b",
            "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf",
            "LLaVA-NeXT-Video-7B-DPO-hf",
            "llava_next_video",
            "vlm",
        ),
        StormModelSpec(
            "glm41v_9b",
            "zai-org/GLM-4.1V-9B-Thinking",
            "GLM-4.1V-9B-Thinking",
            "glm4v",
            "modern",
        ),
    )
}

DEFAULT_MODEL_KEYS = tuple(MODEL_SPECS)


def parse_model_keys(value: str | None) -> list[str]:
    if not value or value.strip().lower() == "all":
        return list(DEFAULT_MODEL_KEYS)
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys {unknown}; available={list(MODEL_SPECS)}")
    return keys
