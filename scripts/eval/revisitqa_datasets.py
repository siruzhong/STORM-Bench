"""Canonical paths and selectors for the seven released RevisitQA subsets."""

from __future__ import annotations

from pathlib import Path
import os
import re


REVISITQA_ROOT = Path(os.environ.get("REVISITQA_ROOT", "data/eval_video/RevisitQA"))
REVISITQA_OUTPUT_ROOT = Path(os.environ.get("REVISITQA_OUTPUT_ROOT", "outputs/revisitqa/released"))
REVISITQA_SUBSETS = {
    "cook": ("revisitqa_cook", "cook_question_answer.jsonl"),
    "bike": ("revisitqa_bike", "bike_question_answer.jsonl"),
    "healthy": ("revisitqa_healthy", "health_question_answer.jsonl"),
    "music": ("revisitqa_music", "music_question_answer.jsonl"),
    "sports": ("revisitqa_sports", "sports_question_answer.jsonl"),
    "thor": ("revisitqa_thor", "thor_question_answer.jsonl"),
    "vhome": ("revisitqa_vhome", "vhome_question_answer.jsonl"),
}
REVISITQA_ALIASES = {"real": "cook", "sim": "thor", "health": "healthy"}
RELEASED_EPISODE = re.compile(
    r"^RevisitQA_(cook|bike|health|music|sports|thor|vhome)_[A-Za-z0-9_]+$"
)


def revisitqa_paths(subset: str) -> tuple[str, str]:
    subset = REVISITQA_ALIASES.get(subset, subset)
    directory, filename = REVISITQA_SUBSETS[subset]
    root = REVISITQA_ROOT / directory
    return str(root / filename), str(root / "video")


def revisitqa_output_dir(subset: str, protocol: str | None = None, model_key: str | None = None) -> str:
    """Output directory for one subset; omit protocol for the models root."""
    subset = REVISITQA_ALIASES.get(subset, subset)
    path = REVISITQA_OUTPUT_ROOT / subset / "models"
    if protocol is not None:
        path = path / protocol
    return str(path / model_key if model_key else path)


def expand_revisitqa_datasets(names: list[str]) -> list[str]:
    """`revisitqa` selects all seven subsets; `revisitqa_<subset>` selects one."""
    expanded = []
    for name in names:
        if name == "revisitqa":
            candidates = [f"revisitqa_{subset}" for subset in REVISITQA_SUBSETS]
        elif name.startswith("revisitqa_"):
            subset = name.removeprefix("revisitqa_")
            candidates = [f"revisitqa_{REVISITQA_ALIASES.get(subset, subset)}"]
        else:
            candidates = [name]
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded

