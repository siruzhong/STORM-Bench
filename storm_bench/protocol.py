"""Model-facing online protocol helpers."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable


def observed_frames(frames: Iterable[dict[str, Any]], query_time: float) -> list[dict[str, Any]]:
    """Return a copy of the causal video prefix; future frames are never exposed."""
    return [frame for frame in frames if float(frame["timestamp"]) <= query_time]


def permute_options(options: list[str], sample_id: str) -> tuple[list[str], list[int]]:
    """Deterministically permute options. Returns (permuted options, new->old indices)."""
    order = list(range(len(options)))
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    order.sort(key=lambda index: digest[index])
    return [options[index] for index in order], order


def restore_index(permuted_index: int, new_to_old: list[int]) -> int:
    if not 0 <= permuted_index < len(new_to_old):
        raise ValueError("prediction index is outside the option range")
    return new_to_old[permuted_index]
