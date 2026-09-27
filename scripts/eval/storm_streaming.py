"""Shared 1 FPS clock and sliding-window state for STORM-Bench Offline / Online."""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


CLOCK_FPS = 1.0
PROTOCOL_NAME = "clock_1fps_sliding_window"
NATIVE_RECURRENT_STATE = False
MODEL_INPUT = "frame_list"


@dataclass(frozen=True)
class StreamFrame:
    stream_index: int
    timestamp: float
    image: Image.Image


def _volatility_bucket(row: dict) -> str:
    intensity = row.get("change_intensity")
    if intensity is None:
        intensity = (row.get("diagnostics") or {}).get("change_intensity")
    value = int(intensity)
    if value <= 3:
        return "1-3"
    if value <= 6:
        return "4-6"
    return "7-10"


def diagnostic_cell(row: dict) -> tuple[str, str]:
    return _volatility_bucket(row), str(row["diagnostics"]["epistemic_status"])


def select_episode_subset(rows: Sequence[dict], limit: int | None, seed: int) -> list[dict]:
    """Greedily preserve diagnostic cells and multiple query-time updates."""
    if limit is None:
        return list(rows)
    if limit <= 0:
        raise ValueError("episode limit must be positive")
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row["episode_id"]), []).append(row)
    if limit >= len(groups):
        return list(rows)

    target_cells = {diagnostic_cell(row) for row in rows}
    uncovered = set(target_cells)
    selected: list[str] = []
    remaining = set(groups)
    while remaining and len(selected) < limit:
        def priority(episode_id: str) -> tuple:
            group = groups[episode_id]
            cells = {diagnostic_cell(row) for row in group}
            query_times = {float(row["query_time"]) for row in group}
            tie = hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()
            return (
                len(cells & uncovered),
                len(cells),
                len(query_times),
                len(group),
                tie,
            )

        chosen = max(remaining, key=priority)
        selected.append(chosen)
        uncovered -= {diagnostic_cell(row) for row in groups[chosen]}
        remaining.remove(chosen)
    selected_set = set(selected)
    return [row for row in rows if str(row["episode_id"]) in selected_set]


def frames_to_images(frames: Sequence) -> list[Image.Image]:
    images: list[Image.Image] = []
    for frame in frames:
        image = frame.image if isinstance(frame, StreamFrame) else frame
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image)).convert("RGB")
        images.append(image.convert("RGB"))
    if not images:
        raise ValueError("Protocol produced an empty frame list")
    return images


def frames_to_timestamps(frames: Sequence, fallback_fps: float = CLOCK_FPS) -> list[float]:
    timestamps: list[float] = []
    for index, frame in enumerate(frames):
        if isinstance(frame, StreamFrame):
            timestamps.append(float(frame.timestamp))
        else:
            timestamps.append(index / max(float(fallback_fps), 1e-6))
    return timestamps


def pad_images_to_multiple(
    images: Sequence[Image.Image],
    timestamps: Sequence[float],
    multiple: int,
) -> tuple[list[Image.Image], list[float]]:
    images = list(images)
    timestamps = list(timestamps)
    if multiple <= 1 or not images:
        return images, timestamps
    remainder = len(images) % multiple
    if remainder == 0:
        return images, timestamps
    pad = multiple - remainder
    images.extend([images[-1]] * pad)
    timestamps.extend([timestamps[-1]] * pad)
    return images, timestamps


def shuffle_visible_frames(
    frames: Sequence[StreamFrame],
    seed: int,
    sample_id: str,
    fps: float = CLOCK_FPS,
) -> list[StreamFrame]:
    """Permute the causal prefix; restamp so processors cannot recover order from metadata."""
    import hashlib
    import random

    items = list(frames)
    if len(items) <= 1:
        return items
    rng = random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()[:8], "big"))
    rng.shuffle(items)
    clock = max(float(fps), 1e-6)
    return [
        StreamFrame(stream_index=index, timestamp=index / clock, image=frame.image)
        for index, frame in enumerate(items)
    ]


def keep_last_frames(
    images: Sequence[Image.Image],
    timestamps: Sequence[float],
    cap: int,
) -> tuple[list[Image.Image], list[float], bool]:
    images = list(images)
    timestamps = list(timestamps)
    if cap <= 0 or len(images) <= cap:
        return images, timestamps, False
    return images[-cap:], timestamps[-cap:], True


class IncrementalVideoSource:
    """Decode fixed-rate observations only when their timestamps become visible."""

    def __init__(self, video_path: str | Path, fps: float):
        from decord import VideoReader, cpu

        if fps <= 0:
            raise ValueError("fps must be positive")
        self.video_path = str(video_path)
        self._reader = VideoReader(self.video_path, ctx=cpu(0), num_threads=1)
        total = len(self._reader)
        if total == 0:
            raise ValueError(f"Video has no frames: {video_path}")
        source_fps = max(float(self._reader.get_avg_fps()), 1e-6)
        requested = max(1, int(np.floor((total - 1) / source_fps * fps)) + 1)
        indices = np.rint(np.arange(requested, dtype=np.float64) * source_fps / fps).astype(int)
        self._indices = np.unique(np.clip(indices, 0, total - 1)).tolist()
        self._timestamps = [index / source_fps for index in self._indices]
        self._cursor = 0

    @property
    def visible_count(self) -> int:
        return self._cursor

    def take_until(self, query_time: float) -> list[StreamFrame]:
        cutoff = bisect.bisect_right(self._timestamps, float(query_time) + 1e-6)
        if cutoff <= self._cursor:
            return []
        start = self._cursor
        arrays = self._reader.get_batch(self._indices[start:cutoff]).asnumpy()
        result = [
            StreamFrame(
                stream_index=index,
                timestamp=self._timestamps[index],
                image=Image.fromarray(array).convert("RGB"),
            )
            for index, array in zip(range(start, cutoff), arrays)
        ]
        self._cursor = cutoff
        return result

    def take_all(self) -> list[StreamFrame]:
        return self.take_until(float("inf"))


class SlidingWindow:
    """Causal recency window. capacity<=0 keeps the whole clock stream."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.limit = None if self.capacity <= 0 else self.capacity
        self._items: list[StreamFrame] = []
        self.seen = 0
        self.num_evictions = 0

    def update(self, frames: Sequence[StreamFrame]) -> None:
        for frame in frames:
            self._items.append(frame)
            self.seen += 1
            if self.limit is not None:
                while len(self._items) > self.limit:
                    self._items.pop(0)
                    self.num_evictions += 1

    def snapshot(self) -> list[StreamFrame]:
        return list(self._items)


class EpisodeClock:
    """One forward-only 1 FPS stream plus a sliding window per video."""

    def __init__(self, video_path: str | Path, fps: float, buffer_frames: int):
        self.source = IncrementalVideoSource(video_path, fps)
        self.window = SlidingWindow(buffer_frames)

    def observe(self, protocol: str, query_time: float | None) -> list[StreamFrame]:
        if protocol == "offline":
            if self.window.seen == 0:
                self.window.update(self.source.take_all())
            return self.window.snapshot()
        if query_time is None:
            raise ValueError("Online evaluation requires query_time")
        self.window.update(self.source.take_until(float(query_time)))
        snapshot = self.window.snapshot()
        if not snapshot:
            raise RuntimeError(f"No causal frames visible at query_time={query_time}")
        return snapshot
