"""Segment timestamped video frames into region-aware activity intervals."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import re
from pathlib import Path

from .client import VLMClient, load_prompt
from .domains import get_domain
from .io_utils import file_sha256, write_json

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

WINDOW_DURATION_SEC = 300.0
MIN_TAIL_WINDOW_SEC = 45.0
SEGMENT_SCHEMA_VERSION = "stream_eqa_activity_segments_v6_en"
INTERVAL_CONVENTION = "[start_sec, end_sec)"
SCHEMA_RETRY_ATTEMPTS = 3
TIME_TOLERANCE_SEC = 0.05
DESCRIPTION_MAX_CHARS = 150

REGIONS = {
    "sink_area", "food_prep_area", "stove_area", "refrigerator_area",
    "cabinet_area", "other_area",
}
SEGMENT_ROLES = {"observation", "interaction", "navigation", "unusable"}
ACTIVITY_TYPES = {
    "observe", "navigate", "retrieve", "place", "clean", "prepare", "cook",
    "open_close", "organize", "other_interaction", "unusable",
}
CHANGE_TYPES = {"position_change", "state_change", "identity_replacement"}
SEGMENT_FIELDS = {
    "start_sec", "end_sec", "region", "segment_role", "activity_type",
    "change_types", "description",
}


class SegmentSchemaError(ValueError):
    """A model response violates the segment schema or timeline."""


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _check_sampling(sample_fps: float, jpeg_quality: int, blur_ksize: int) -> None:
    if _finite_number(sample_fps, "sample_fps") <= 0:
        raise ValueError("sample_fps must be positive")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer in [1, 100]")
    if type(blur_ksize) is not int or blur_ksize < 1 or blur_ksize % 2 == 0:
        raise ValueError("blur_ksize must be a positive odd integer")


def _open_video(video_path: Path):
    if cv2 is None:
        raise RuntimeError("Video sampling requires opencv-python-headless")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise OSError("Cannot open the source video")
    return cap


def _capture_metadata(cap) -> tuple[float, int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count < 1:
        raise OSError("Cannot determine a positive video duration")
    return fps, int(count)


def _video_metadata(video_path: Path) -> tuple[float, int]:
    cap = _open_video(video_path)
    try:
        return _capture_metadata(cap)
    finally:
        cap.release()


def video_duration(video_path: Path) -> float:
    """Return the source duration in seconds, rounded to two decimals."""
    fps, count = _video_metadata(video_path)
    return round(count / fps, 2)


def build_processing_windows(
    duration_sec: float,
    window_sec: float = WINDOW_DURATION_SEC,
    min_tail_sec: float = MIN_TAIL_WINDOW_SEC,
) -> list[tuple[float, float]]:
    """Cover the source with windows, merging short tails into the preceding one."""
    duration_sec = round(_finite_number(duration_sec, "duration_sec"), 2)
    window_sec = round(_finite_number(window_sec, "window_sec"), 2)
    min_tail_sec = round(_finite_number(min_tail_sec, "min_tail_sec"), 2)
    if duration_sec <= 0 or window_sec <= 0:
        raise ValueError("duration_sec and window_sec must be positive")
    if not 0 <= min_tail_sec <= window_sec:
        raise ValueError("min_tail_sec must be within [0, window_sec]")
    if duration_sec <= window_sec:
        return [(0.0, duration_sec)]
    count = int(duration_sec // window_sec)
    windows = [
        (round(i * window_sec, 2), round((i + 1) * window_sec, 2))
        for i in range(count)
    ]
    tail_start = windows[-1][1]
    tail = round(duration_sec - tail_start, 2)
    if tail > 0 and tail >= min_tail_sec:
        windows.append((tail_start, duration_sec))
    elif tail > 0:
        windows[-1] = (windows[-1][0], duration_sec)
    return windows


def _sampling_grid(
    fps: float, count: int, sample_fps: float, start_sec: float, end_sec: float,
) -> tuple[list[tuple[int, float]], float]:
    start = round(max(0.0, _finite_number(start_sec, "start_sec")), 2)
    end = round(min(count / fps, _finite_number(end_sec, "end_sec")), 2)
    if end <= start:
        raise ValueError("The sampling interval must have positive duration")
    first = max(0, round(start * fps))
    stop = min(count, math.ceil(end * fps))
    step = max(1, round(fps / sample_fps))
    grid = [(i, round(i / fps, 2)) for i in range(first, stop, step)]
    grid = [(i, timestamp) for i, timestamp in grid if timestamp < end]
    if not grid or abs(grid[0][1] - start) > TIME_TOLERANCE_SEC:
        raise ValueError("The window start does not align with a source frame")
    # Canonicalize small frame-rounding offsets to the requested window boundary.
    grid[0] = (grid[0][0], start)
    return grid, end


def _blur_and_encode(frame_bgr, jpeg_quality: int, blur_ksize: int = 5) -> str:
    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    ok, buffer = cv2.imencode(".jpg", blurred, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def sample_frames(
    video_path: Path,
    sample_fps: float,
    frames_dir: Path | None = None,
    jpeg_quality: int = 75,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    blur_ksize: int = 5,
) -> tuple[list[tuple[float, str]], float]:
    """Sample blurred JPEGs in memory using absolute source timestamps.

    ``frames_dir`` remains accepted for API compatibility; no frame files are written.
    """
    _check_sampling(sample_fps, jpeg_quality, blur_ksize)
    cap = _open_video(video_path)
    try:
        fps, count = _capture_metadata(cap)
        grid, end = _sampling_grid(
            fps, count, sample_fps, start_sec,
            count / fps if end_sec is None else end_sec,
        )
        frames = []
        for index, timestamp in grid:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"Cannot decode the sampled frame at {timestamp:.2f}s")
            frames.append((timestamp, _blur_and_encode(frame, jpeg_quality, blur_ksize)))
        return frames, end
    finally:
        cap.release()


def _canonical_boundary(value: object, allowed: list[float], field: str) -> float:
    try:
        number = _finite_number(value, field)
    except ValueError as error:
        raise SegmentSchemaError(str(error)) from error
    closest = min(allowed, key=lambda boundary: abs(boundary - number))
    if abs(closest - number) > TIME_TOLERANCE_SEC:
        raise SegmentSchemaError(f"{field}={number} is not aligned to a sampled boundary")
    return closest


def validate_segments(
    segments: object,
    interval_start: float,
    interval_end: float,
    allowed_boundaries: list[float],
    *,
    domain: str = "cook",
) -> list[dict]:
    """Validate and canonicalize a complete, contiguous half-open timeline."""
    spec = get_domain(domain)
    if not isinstance(segments, list) or not segments:
        raise SegmentSchemaError("Output must be a non-empty JSON array")
    try:
        start = round(_finite_number(interval_start, "interval_start"), 2)
        end = round(_finite_number(interval_end, "interval_end"), 2)
        allowed = sorted({round(_finite_number(x, "boundary"), 2) for x in allowed_boundaries})
    except (TypeError, ValueError) as error:
        raise SegmentSchemaError(str(error)) from error
    if not allowed or end <= start or any(x < start or x > end for x in allowed):
        raise SegmentSchemaError("Invalid interval or allowed boundaries")
    normalized = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or set(segment) != SEGMENT_FIELDS:
            raise SegmentSchemaError(f"segment[{index}] must contain exactly the schema fields")
        item = dict(segment)
        item["start_sec"] = _canonical_boundary(item["start_sec"], allowed, "start_sec")
        item["end_sec"] = _canonical_boundary(item["end_sec"], allowed, "end_sec")
        if item["end_sec"] <= item["start_sec"]:
            raise SegmentSchemaError(f"segment[{index}] must have positive duration")
        for field, values in (
            ("region", spec.regions), ("segment_role", SEGMENT_ROLES),
            ("activity_type", spec.activities),
        ):
            if not isinstance(item[field], str) or item[field] not in values:
                raise SegmentSchemaError(f"segment[{index}] has invalid {field}")
        role, activity = item["segment_role"], item["activity_type"]
        expected = {"observation": "observe", "navigation": "navigate", "unusable": "unusable"}
        if role in expected and activity != expected[role]:
            raise SegmentSchemaError(f"segment[{index}] has incompatible role and activity")
        if role == "interaction" and activity in expected.values():
            raise SegmentSchemaError(f"segment[{index}] has incompatible interaction activity")
        changes = item["change_types"]
        if not isinstance(changes, list) or any(
            not isinstance(change, str) or change not in CHANGE_TYPES for change in changes
        ):
            raise SegmentSchemaError(f"segment[{index}] has invalid change_types")
        item["change_types"] = list(dict.fromkeys(changes))
        if role != "interaction" and changes:
            raise SegmentSchemaError(f"segment[{index}] non-interaction changes must be empty")
        description = item["description"]
        if not isinstance(description, str) or not 0 < len(description.strip()) <= DESCRIPTION_MAX_CHARS:
            raise SegmentSchemaError(f"segment[{index}] needs a description of 1-150 characters")
        item["description"] = description.strip()
        normalized.append(item)
    if normalized[0]["start_sec"] != start or normalized[-1]["end_sec"] != end:
        raise SegmentSchemaError("Segments must cover both interval endpoints")
    if any(left["end_sec"] != right["start_sec"] for left, right in zip(normalized, normalized[1:])):
        raise SegmentSchemaError("Segments have a gap, overlap, or ordering error")
    return normalized


def _build_batch_contents(
    frames_batch: list[tuple[float, str]],
    interval_start: float,
    interval_end: float,
    validation_error: str | None = None,
) -> list:
    contents = [
        f"Cover the complete half-open interval [{interval_start}, {interval_end}). "
        f"The first start_sec must be {interval_start}, and the final end_sec must be {interval_end}."
    ]
    if validation_error:
        contents.append(
            "The previous response failed schema validation. Return a corrected, complete JSON array. "
            f"Validation error: {validation_error}"
        )
    for timestamp, image in frames_batch:
        contents.extend([f"[t={timestamp}s]", {"type": "image_base64", "data": image}])
    return contents


def segment_video(
    client: VLMClient,
    video_path: Path,
    window_start_sec: float,
    window_end_sec: float,
    frames_dir: Path | None = None,
    sample_fps: float = 1.0,
    max_frames_per_call: int = 64,
    jpeg_quality: int = 75,
    blur_ksize: int = 5,
    thinking: str | None = None,
    *,
    domain: str = "cook",
) -> list[dict]:
    """Segment one window with the selected domain's prompt and retry policy."""
    spec = get_domain(domain)
    if type(max_frames_per_call) is not int or max_frames_per_call < 1:
        raise ValueError("max_frames_per_call must be a positive integer")
    frames, end = sample_frames(
        video_path, sample_fps, frames_dir, jpeg_quality,
        window_start_sec, window_end_sec, blur_ksize,
    )
    if not frames or end <= frames[0][0]:
        raise ValueError("No usable frames or positive-duration interval was sampled")
    prompt = load_prompt("segmentation.txt", domain=spec.name)
    segments = []
    for offset in range(0, len(frames), max_frames_per_call):
        batch = frames[offset:offset + max_frames_per_call]
        next_index = offset + len(batch)
        start = batch[0][0]
        batch_end = frames[next_index][0] if next_index < len(frames) else end
        allowed = [timestamp for timestamp, _ in batch] + [batch_end]
        validation_error = None
        logger.info("Segmenting [%0.2f, %0.2f): %d frames", start, batch_end, len(batch))
        for attempt in range(1, spec.segmentation_retries + 1):
            raw = client.call(
                prompt, _build_batch_contents(batch, start, batch_end, validation_error),
                thinking=thinking,
            )
            text = raw.strip()
            fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
            try:
                parsed = json.loads(fence.group(1) if fence else text)
                segments.extend(validate_segments(
                    parsed, start, batch_end, allowed, domain=spec.name,
                ))
                break
            except (json.JSONDecodeError, SegmentSchemaError) as error:
                validation_error = str(error)
                logger.warning(
                    "Segment schema attempt %d/%d failed: %s",
                    attempt, spec.segmentation_retries, error,
                )
        else:
            raise SegmentSchemaError(
                f"Batch [{start}, {batch_end}) failed after "
                f"{spec.segmentation_retries} attempts: {validation_error}"
            )
    return validate_segments(
        segments, window_start_sec, end, [timestamp for timestamp, _ in frames] + [end],
        domain=spec.name,
    )


def _validate_cached_result(
    path: Path, metadata: dict, allowed: list[float], *, domain: str = "cook"
) -> None:
    spec = get_domain(domain)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or set(result) != set(metadata) | {"segments"}:
            raise ValueError("Output fields do not match the current schema")
        if any(result.get(key) != value for key, value in metadata.items()):
            raise ValueError("Saved window metadata or segmentation parameters do not match")
        start, end = metadata["window_interval"]
        if result.get("domain") != spec.name:
            raise ValueError("Saved segmentation domain does not match")
        if result.get("schema_version") != spec.segmentation_schema:
            raise ValueError("Saved segmentation schema does not match the domain")
        validate_segments(result["segments"], start, end, allowed, domain=spec.name)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"Cannot reuse {path.name}: {error}. Use overwrite to regenerate.") from error


def segment_source(
    video_path: Path,
    output_dir: Path,
    client: VLMClient,
    *,
    video_id: str,
    domain: str = "cook",
    window_sec: float = 300.0,
    min_tail_sec: float = 45.0,
    sample_fps: float = 1.0,
    max_frames_per_call: int = 64,
    jpeg_quality: int = 75,
    blur_ksize: int = 5,
    thinking: str | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Write anonymous per-window segmentation JSONs, validating any cached output."""
    spec = get_domain(domain)
    if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", video_id):
        raise ValueError("video_id must be an anonymous identifier containing letters, digits, '_' or '-'")
    if thinking not in {None, "enabled", "disabled"}:
        raise ValueError("thinking must be enabled, disabled, or None")
    _check_sampling(sample_fps, jpeg_quality, blur_ksize)
    if type(max_frames_per_call) is not int or max_frames_per_call < 1:
        raise ValueError("max_frames_per_call must be a positive integer")
    fps, count = _video_metadata(video_path)
    duration = round(count / fps, 2)
    windows = build_processing_windows(duration, window_sec, min_tail_sec)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = {
        "window_sec": window_sec, "min_tail_sec": min_tail_sec,
        "sample_fps": sample_fps, "max_frames_per_call": max_frames_per_call,
        "jpeg_quality": jpeg_quality, "blur_ksize": blur_ksize, "thinking": thinking,
        "model": getattr(client, "model", None),
        "domain": spec.name,
        "prompt_sha256": hashlib.sha256(
            load_prompt("segmentation.txt", domain=spec.name).encode("utf-8")
        ).hexdigest(),
    }
    source_sha256 = file_sha256(video_path)
    paths = []
    for index, (start, end) in enumerate(windows, start=1):
        episode_id = f"{video_id}_window{index:03d}"
        path = output_dir / f"{episode_id}.json"
        grid, _ = _sampling_grid(fps, count, sample_fps, start, end)
        timestamps = [timestamp for _, timestamp in grid]
        metadata = {
            "video_id": episode_id, "source_video_id": video_id,
            "domain": spec.name,
            "source_sha256": source_sha256,
            "source_duration_sec": duration, "schema_version": spec.segmentation_schema,
            "interval_convention": INTERVAL_CONVENTION, "window_index": index,
            "window_count": len(windows), "window_interval": [start, end],
            "processed_interval": [start, end], "sample_fps": sample_fps,
            "sample_timestamps": timestamps, "parameters": parameters,
        }
        if path.exists() and not overwrite:
            _validate_cached_result(
                path, metadata, timestamps + [end], domain=spec.name,
            )
            logger.info("Reusing validated segmentation: %s", path.name)
        else:
            segments = segment_video(
                client, video_path, start, end, sample_fps=sample_fps,
                max_frames_per_call=max_frames_per_call, jpeg_quality=jpeg_quality,
                blur_ksize=blur_ksize, thinking=thinking,
                domain=spec.name,
            )
            write_json(path, {**metadata, "segments": segments})
            logger.info("Saved %s: %d segments", path.name, len(segments))
        paths.append(path)
    return paths
