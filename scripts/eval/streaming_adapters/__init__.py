"""Native streaming adapters for RevisitQA.

A streaming adapter keeps visual state across the questions of one episode. The evaluator
(`scripts/eval/eval_revisitqa_streaming.py`) drives it as:

    adapter.reset()                         # new episode: drop all visual state
    adapter.observe(frames)                 # frames newly visible since the last call (1 FPS, timestamps <= t_q)
    text = adapter.answer(prompt, 4)        # answer from current state only; must not keep the question/answer

Each adapter module in this package defines

    SPEC = StreamingSpec(key=..., checkpoint=..., runtime=..., description=...)
    def build(model_path: Path, args) -> StreamingAdapter

and is discovered automatically, so adding a model touches only its own module.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class StreamingSpec:
    key: str
    checkpoint: str  # directory under the checkpoint root
    runtime: str  # python interpreter, relative to the repo root or absolute
    description: str
    extra_checkpoints: dict[str, str] = field(default_factory=dict)

    def runtime_path(self) -> Path:
        path = Path(self.runtime)
        return path if path.is_absolute() else REPO_ROOT / path


class StreamingAdapter(Protocol):
    def reset(self) -> None: ...

    def observe(self, frames: Sequence) -> None: ...

    def answer(self, prompt: str, option_count: int) -> str: ...

    def state_info(self) -> dict: ...


def _modules():
    for info in pkgutil.iter_modules(__path__):
        if not info.name.startswith("_"):
            yield info.name


def specs() -> dict[str, StreamingSpec]:
    """Specs only; importing a spec must not import heavy model code."""
    found = {}
    for name in _modules():
        module = importlib.import_module(f"{__name__}.{name}")
        spec = getattr(module, "SPEC", None)
        if spec is not None:
            found[spec.key] = spec
    return found


def build(key: str, model_path: Path, args) -> StreamingAdapter:
    for name in _modules():
        module = importlib.import_module(f"{__name__}.{name}")
        spec = getattr(module, "SPEC", None)
        if spec is not None and spec.key == key:
            return module.build(model_path, args)
    raise KeyError(f"No streaming adapter registered for {key}")
