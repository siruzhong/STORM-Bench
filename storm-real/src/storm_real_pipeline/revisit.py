"""Select same-region revisits and assemble a silent, temporally indexed episode."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from .client import load_prompt
from .domains import get_domain
from .io_utils import write_json

if TYPE_CHECKING:
    from .client import VLMClient

logger = logging.getLogger(__name__)
GENERATOR_VERSION = "storm_real_revisit_v1"
SEGMENT_SCHEMA_VERSION = "stream_eqa_activity_segments_v6_en"
MIN_VISITS, MAX_VISITS = 3, 10
MIN_VISIT_SEC, MAX_VISIT_SEC = 3.0, 20.0
TIME_TOLERANCE_SEC = 0.05
REGIONS = {
    "sink_area",
    "food_prep_area",
    "stove_area",
    "refrigerator_area",
    "cabinet_area",
    "other_area",
}
ROLES = {"observation", "interaction", "navigation", "unusable"}
ACTIVITIES = {
    "observe",
    "navigate",
    "retrieve",
    "place",
    "clean",
    "prepare",
    "cook",
    "open_close",
    "organize",
    "other_interaction",
    "unusable",
}
CHANGE_TYPES = {"position_change", "state_change", "identity_replacement"}


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}"
        )
    return result


def validate_segmentation_meta(
    meta: dict, *, domain: str = "cook"
) -> tuple[list[dict], float, float]:
    """Validate the English region schema and its contiguous half-open timeline."""
    spec = get_domain(domain)
    if not isinstance(meta, dict) or meta.get("schema_version") != spec.segmentation_schema:
        raise ValueError(f"Segmentation schema must be {spec.segmentation_schema}")
    if meta.get("domain", "cook") != spec.name:
        raise ValueError("Segmentation domain does not match the requested domain")
    if meta.get("interval_convention") != "[start_sec, end_sec)":
        raise ValueError("Segmentation must use [start_sec, end_sec) intervals")
    segments = meta.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments must be a nonempty list")
    interval = meta.get("processed_interval")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("processed_interval must contain two numbers")
    start = _number(interval[0], "processed_interval start")
    end = _number(interval[1], "processed_interval end", positive=True)
    if end <= start:
        raise ValueError("processed_interval must have positive duration")
    previous_end = start
    normalized = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index} must be an object")
        st = _number(segment.get("start_sec"), f"segment {index} start")
        ed = _number(segment.get("end_sec"), f"segment {index} end", positive=True)
        if ed <= st or abs(st - previous_end) > TIME_TOLERANCE_SEC:
            raise ValueError(f"segment {index} must be positive and contiguous")
        for field, allowed in (
            ("region", spec.regions),
            ("segment_role", ROLES),
            ("activity_type", spec.activities),
        ):
            if not isinstance(segment.get(field), str) or segment[field] not in allowed:
                raise ValueError(f"segment {index} has an invalid {field}")
        changes = segment.get("change_types")
        if not isinstance(changes, list) or any(
            not isinstance(item, str) or item not in CHANGE_TYPES
            for item in changes
        ):
            raise ValueError(f"segment {index} has invalid change_types")
        if not isinstance(segment.get("description"), str):
            raise ValueError(f"segment {index} must have a description string")
        normalized.append({**segment, "start_sec": st, "end_sec": ed, "_index": index})
        previous_end = ed
    if abs(previous_end - end) > TIME_TOLERANCE_SEC:
        raise ValueError("The last segment must end at processed_interval end")
    return normalized, start, end


def _duration(segment: dict) -> float:
    return segment["end_sec"] - segment["start_sec"]


def _region_of(segment: dict, *, domain: str = "cook") -> str:
    spec = get_domain(domain)
    raw = segment["region"]
    return spec.region_aliases.get(raw, raw)


def _candidate(segment: dict, *, domain: str = "cook") -> bool:
    spec = get_domain(domain)
    return (
        segment["segment_role"] in {"observation", "interaction"}
        and _region_of(segment, domain=spec.name) in spec.regions - {"other_area"}
        and _duration(segment) >= MIN_VISIT_SEC
    )


def _separate(left: dict, right: dict) -> bool:
    return (
        left["end_sec"] + TIME_TOLERANCE_SEC < right["start_sec"]
        or right["end_sec"] + TIME_TOLERANCE_SEC < left["start_sec"]
    )


def choose_region(
    segments: list[dict], target_sec: float, speed: float, blank_sec: float,
    *, domain: str = "cook",
) -> str:
    """Choose a region with enough eligible segments and useful source duration."""
    spec = get_domain(domain)
    region_order = (
        sorted(spec.regions - {"other_area"})
        if spec.name == "cook"
        else dict.fromkeys(_region_of(seg, domain=spec.name) for seg in segments)
    )
    groups = {
        region: [
            seg for seg in segments
            if _region_of(seg, domain=spec.name) == region
            and _candidate(seg, domain=spec.name)
        ]
        for region in region_order if region in spec.regions - {"other_area"}
    }
    eligible = [
        region for region, values in groups.items()
        if len(values) >= spec.min_visits
    ]
    if not eligible:
        raise ValueError(
            f"No region contains at least {spec.min_visits} eligible visit candidates"
        )
    required = (
        max(0.0, (target_sec - (spec.min_visits - 1) * blank_sec) * speed)
        if spec.name == "cook" else target_sec * speed
    )

    def score(region: str) -> tuple:
        values = groups[region]
        durations = [min(_duration(seg), MAX_VISIT_SEC) for seg in values]
        usable = sum(
            sorted(durations, reverse=True)[:MAX_VISITS]
            if spec.name == "cook" else durations
        )
        transitions = sum(_separate(a, b) for a, b in zip(values, values[1:]))
        common = (usable >= required, min(usable, required))
        if spec.name == "cook":
            return *common, transitions, len(values)
        return *common, spec.region_priority.get(region, 0), transitions, len(values)

    return max(eligible, key=score)


def estimate_output_duration(
    visits: list[dict], speed: float, blank_sec: float
) -> float:
    return (
        sum(_duration(visit) for visit in visits) / speed
        + max(0, len(visits) - 1) * blank_sec
    )


def _plan_score(
    visits: list[dict], target_sec: float, speed: float, blank_sec: float
) -> float:
    return abs(estimate_output_duration(visits, speed, blank_sec) - target_sec)


def pick_visits_greedy(
    segments: list[dict],
    target_sec: float,
    blank_sec: float,
    speed: float = 1.5,
    max_total_sec: float = 60.0,
    *,
    domain: str = "cook",
) -> list[dict]:
    """Scan candidates chronologically using the original source-duration budget."""
    spec = get_domain(domain)
    candidates = [
        segment for segment in segments if _candidate(segment, domain=spec.name)
    ]
    if len({_region_of(seg, domain=spec.name) for seg in candidates}) > 1:
        raise ValueError("Visit selection requires candidates from one region")
    candidates.sort(key=lambda segment: segment["start_sec"])
    visits = []
    last_end = -1.0
    used_source = 0.0
    budget_source = max(
        (target_sec - MAX_VISITS * blank_sec) * speed,
        spec.min_visits * MIN_VISIT_SEC,
    )
    for segment in candidates:
        if len(visits) >= MAX_VISITS:
            break
        projected = (
            (used_source + _duration(segment)) / speed
            + (len(visits) + 1) * blank_sec
        )
        if projected > max_total_sec + 0.5 and len(visits) >= spec.min_visits:
            break
        if used_source >= budget_source + 0.5 and len(visits) >= spec.min_visits:
            break
        if _duration(segment) < MIN_VISIT_SEC:
            continue
        if last_end >= 0 and segment["start_sec"] <= last_end + TIME_TOLERANCE_SEC:
            continue
        remaining_source = budget_source - used_source
        duration = min(_duration(segment), MAX_VISIT_SEC)
        if duration > remaining_source and remaining_source >= MIN_VISIT_SEC:
            duration = remaining_source
        if duration < MIN_VISIT_SEC:
            continue
        visits.append({
            "start_sec": segment["start_sec"],
            "end_sec": segment["start_sec"] + duration,
        })
        used_source += duration
        last_end = segment["end_sec"]
    return visits


def validate_visits(
    raw_visits: object,
    segments: list[dict],
    speed: float,
    blank_sec: float,
    max_total_sec: float,
    *,
    domain: str = "cook",
) -> list[dict]:
    """Discard unsupported intervals; never allow duplicate visits to one candidate."""
    spec = get_domain(domain)
    if not isinstance(raw_visits, list):
        return []
    candidates = [
        segment for segment in segments if _candidate(segment, domain=spec.name)
    ]
    if len({_region_of(seg, domain=spec.name) for seg in candidates}) > 1:
        raise ValueError("Visit validation requires candidates from one region")
    cleaned = []
    for raw in raw_visits:
        if not isinstance(raw, dict):
            continue
        try:
            st, ed = (
                _number(raw.get("start_sec"), "visit start"),
                _number(raw.get("end_sec"), "visit end"),
            )
        except ValueError:
            continue
        if ed - st < MIN_VISIT_SEC - 1e-8 or ed - st > MAX_VISIT_SEC + 1e-8:
            continue
        source = next(
            (
                seg for seg in candidates
                if st >= seg["start_sec"] - 1e-8 and ed <= seg["end_sec"] + 1e-8
            ),
            None,
        )
        if source is not None:
            cleaned.append({
                "start_sec": st,
                "end_sec": ed,
                "candidate_role": source["segment_role"],
                "source_segment_index": source["_index"],
            })
    cleaned.sort(key=lambda visit: visit["start_sec"])
    visits, seen = [], set()
    for visit in cleaned:
        if visit["source_segment_index"] in seen or (
            visits
            and visit["start_sec"] <= visits[-1]["end_sec"] + TIME_TOLERANCE_SEC
        ):
            continue
        trial = visits + [visit]
        if estimate_output_duration(trial, speed, blank_sec) > max_total_sec + 1e-8:
            if len(visits) >= spec.min_visits:
                break
            remaining_source = (
                max_total_sec
                - sum(_duration(item) for item in visits) / speed
                - len(visits) * blank_sec
            ) * speed
            if remaining_source >= MIN_VISIT_SEC:
                visits.append({
                    **visit,
                    "end_sec": visit["start_sec"] + remaining_source,
                })
            break
        visits.append(visit)
        seen.add(visit["source_segment_index"])
        if len(visits) == MAX_VISITS:
            break
    return visits


def _llm_visits(
    client: VLMClient,
    segments: list[dict],
    region: str,
    parameters: dict,
    duration_sec: float,
    *,
    domain: str = "cook",
) -> object:
    spec = get_domain(domain)
    prompt = Template(load_prompt("revisit.txt", domain=spec.name)).substitute(
        region=region,
        min_visits=spec.min_visits,
        max_visits=MAX_VISITS,
        min_visit_sec=MIN_VISIT_SEC,
        max_visit_sec=MAX_VISIT_SEC,
        src_sum_lo=(
            parameters["target_sec"] - MAX_VISITS * parameters["blank_sec"]
        ) * parameters["speed"],
        src_sum_hi=parameters["target_sec"] * parameters["speed"],
        target_lower_sec=max(0.0, parameters["target_sec"] - 3.0),
        target_upper_sec=min(
            parameters["max_total_sec"], parameters["target_sec"] + 3.0
        ),
        **{
            name: parameters[name]
            for name in ("target_sec", "max_total_sec", "speed", "blank_sec")
        },
    )
    data = [
        {
            key: segment[key]
            for key in (
                "start_sec",
                "end_sec",
                "region",
                "segment_role",
                "activity_type",
                "change_types",
                "description",
            )
        } | {"visit_candidate": _candidate(segment, domain=spec.name)}
        for segment in segments
    ]
    raw = client.call(
        prompt,
        [json.dumps(
            {"source_duration_sec": duration_sec, "segments": data},
            ensure_ascii=False,
        )],
        temperature=1.0,
        max_tokens=8192,
    )
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("The selection response must be a JSON object")
    return parsed.get("visits")


def _probe_video(path: Path, *, decode: bool = False) -> dict:
    """Read metadata and optionally verify every output frame can be decoded."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Video cannot be opened")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Video has invalid metadata")
        if decode:
            decoded = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.shape[:2] != (height, width):
                    raise ValueError("Video contains an invalid frame")
                decoded += 1
            if decoded != count:
                raise ValueError("Video is truncated or contains undecodable frames")
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": count,
            "duration_sec": count / fps,
        }
    finally:
        cap.release()


def _ffmpeg() -> str:
    configured = os.environ.get("STORM_FFMPEG")
    executable = shutil.which(configured) if configured else shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("FFmpeg is unavailable; set STORM_FFMPEG or add ffmpeg to PATH")
    return executable


def _run_ffmpeg(executable: str, args: list[str]) -> None:
    result = subprocess.run(
        [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        # FFmpeg diagnostics can contain input paths; do not publish those strings.
        raise RuntimeError(f"FFmpeg failed with exit code {result.returncode}")


def _timeline(
    visits: list[dict], fps: float, speed: float, blank_sec: float
) -> tuple[list[dict], int, int]:
    """Floor durations to whole output frames so rounding cannot exceed the cap."""
    blank_frames = math.floor(blank_sec * fps + 1e-7)
    if blank_sec > 0 and blank_frames < 1:
        raise ValueError(
            "blank_sec is shorter than one output frame; use zero or a longer gap"
        )
    cursor, indexed = 0, []
    for i, visit in enumerate(visits):
        frames = math.floor(_duration(visit) / speed * fps + 1e-7)
        if frames < 1:
            raise ValueError(
                "A visit is shorter than one output frame at the requested speed"
            )
        indexed.append({
            **visit,
            "visit_index": i + 1,
            "output_start_sec": cursor / fps,
            "output_end_sec": (cursor + frames) / fps,
            "output_frame_count": frames,
        })
        cursor += frames + (blank_frames if i < len(visits) - 1 else 0)
    return indexed, blank_frames, cursor


def _assemble(
    source: Path,
    visits: list[dict],
    info: dict,
    blank_frames: int,
    parameters: dict,
    temporary: Path,
    executable: str,
) -> tuple[Path, dict]:
    fps, width, height = info["fps"], info["width"], info["height"]
    width, height = width + width % 2, height + height % 2
    fps_text = f"{fps:.12g}"
    encoder = [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
    ]
    clips = []
    for i, visit in enumerate(visits):
        clip = temporary / f"visit_{i:02d}.mp4"
        frames = visit["output_frame_count"]
        out_duration = frames / fps
        filters = [
            f"setpts=(PTS-STARTPTS)/{parameters['speed']:.12g}",
            f"fps={fps_text}",
            f"pad={width}:{height}:0:0",
            "setsar=1",
            f"tpad=stop_mode=clone:stop_duration={2 / fps:.12g}",
            f"trim=end_frame={frames}",
            f"setpts=N/({fps_text}*TB)",
        ]
        fade = min(parameters["fade_sec"], out_duration / 2)
        if fade > 0 and i:
            filters.append(f"fade=t=in:st=0:d={fade:.12g}")
        if fade > 0 and i < len(visits) - 1:
            filters.append(f"fade=t=out:st={out_duration - fade:.12g}:d={fade:.12g}")
        _run_ffmpeg(executable, [
            "-ss", f"{visit['start_sec']:.12g}",
            "-t", f"{_duration(visit):.12g}",
            "-i", str(source),
            "-map", "0:v:0",
            "-vf", ",".join(filters),
            "-frames:v", str(frames),
            "-r", fps_text,
            *encoder,
            str(clip),
        ])
        clips.append(clip)
        if i < len(visits) - 1 and blank_frames:
            blank = temporary / "blank.mp4"
            if not blank.exists():
                _run_ffmpeg(executable, [
                    "-f", "lavfi",
                    "-i", f"color=c=black:s={width}x{height}:r={fps_text}",
                    "-frames:v", str(blank_frames),
                    *encoder,
                    str(blank),
                ])
            clips.append(blank)
    manifest = temporary / "concat.txt"
    manifest.write_text("".join(f"file '{clip.name}'\n" for clip in clips), encoding="utf-8")
    combined = temporary / "episode.mp4"
    expected_frames = (
        sum(visit["output_frame_count"] for visit in visits)
        + blank_frames * (len(visits) - 1)
    )
    # Rebuild a constant-rate timeline after concat. Copying MP4 packets can
    # accumulate container-duration rounding and shift the average frame rate.
    _run_ffmpeg(executable, [
        "-f", "concat",
        "-safe", "1",
        "-i", str(manifest),
        "-map", "0:v:0",
        "-vf", f"setpts=N/({fps_text}*TB)",
        "-r", fps_text,
        "-frames:v", str(expected_frames),
        *encoder,
        "-movflags", "+faststart",
        str(combined),
    ])
    return combined, _probe_video(combined, decode=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hidden_changes(visits: list[dict], segments: list[dict]) -> list[dict]:
    results = []
    for left, right in zip(visits, visits[1:]):
        changes = [
            seg for seg in segments
            if seg["change_types"]
            and seg["start_sec"] >= left["end_sec"] - 1e-8
            and seg["end_sec"] <= right["start_sec"] + 1e-8
        ]
        results.append({
            "source_gap_start_sec": left["end_sec"],
            "source_gap_end_sec": right["start_sec"],
            "change_segment_count": len(changes),
            "change_types": sorted({
                kind for seg in changes for kind in seg["change_types"]
            }),
            "source_segment_indices": [seg["_index"] for seg in changes],
        })
    return results


def construct_episode(
    segmentation_path: Path,
    source_video: Path,
    output_path: Path,
    client: VLMClient | None,
    *,
    domain: str = "cook",
    region: str | None = None,
    target_sec: float = 45.0,
    max_total_sec: float = 60.0,
    speed: float = 1.5,
    blank_sec: float = 0.2,
    fade_sec: float = 0.2,
    overwrite: bool = False,
) -> dict:
    """Construct an episode and atomic plan sidecar, or resume an identical valid output.

    Both selection methods run when a client is provided. Invalid or failed LLM
    selection falls back to the deterministic plan. An existing stale output
    requires ``overwrite=True``; no source metadata path is used as an input.
    """
    spec = get_domain(domain)
    parameters = {
        name: _number(value, name, positive=name not in {"blank_sec", "fade_sec"})
        for name, value in {
            "target_sec": target_sec,
            "max_total_sec": max_total_sec,
            "speed": speed,
            "blank_sec": blank_sec,
            "fade_sec": fade_sec,
        }.items()
    }
    target_sec, max_total_sec, speed, blank_sec, fade_sec = (
        parameters[name]
        for name in ("target_sec", "max_total_sec", "speed", "blank_sec", "fade_sec")
    )
    if target_sec > max_total_sec:
        raise ValueError("target_sec cannot exceed max_total_sec")
    if (
        spec.min_visits * MIN_VISIT_SEC / speed
        + (spec.min_visits - 1) * blank_sec
        > max_total_sec + 1e-8
    ):
        raise ValueError(
            f"max_total_sec cannot accommodate {spec.min_visits} "
            "minimum-length visits"
        )
    if isinstance(region, str):
        region = spec.region_aliases.get(region, region)
    if region is not None and (
        not isinstance(region, str) or region not in spec.regions - {"other_area"}
    ):
        raise ValueError("region must be an eligible English region label")
    segmentation_path, source_video, output_path = map(
        Path, (segmentation_path, source_video, output_path)
    )
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("output_path must have the .mp4 extension")
    plan_path = output_path.with_suffix(".plan.json")
    if source_video.resolve() in {
        output_path.resolve(), plan_path.resolve()
    } or segmentation_path.resolve() in {
        output_path.resolve(), plan_path.resolve()
    }:
        raise ValueError("Output paths must differ from the inputs")
    segmentation_bytes = segmentation_path.read_bytes()
    metadata = json.loads(segmentation_bytes)
    segments, interval_start, interval_end = validate_segmentation_meta(
        metadata, domain=spec.name,
    )
    segmentation_hash = hashlib.sha256(segmentation_bytes).hexdigest()
    for field in ("source_video_id", "video_id"):
        value = metadata.get(field)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value) is None
        ):
            raise ValueError(f"Segmentation {field} must be a safe anonymous identifier")
    source_video_id = metadata["source_video_id"]
    episode_id = metadata["video_id"] + "_revisit"
    source_hash = _sha256(source_video)
    if "source_sha256" in metadata and metadata["source_sha256"] != source_hash:
        raise ValueError(
            "Source video content does not match the segmentation source_sha256"
        )
    info = _probe_video(source_video)
    if interval_end > info["duration_sec"] + 1 / info["fps"]:
        raise ValueError("The processed interval extends beyond the source video")
    model = getattr(client, "model", None) if client is not None else None
    model_hash = (
        hashlib.sha256(str(model).encode()).hexdigest()
        if model is not None else None
    )
    prompt_hash = hashlib.sha256(
        load_prompt("revisit.txt", domain=spec.name).encode()
    ).hexdigest()
    config = {
        **parameters,
        "domain": spec.name,
        "min_visits": spec.min_visits,
        "requested_region": region,
        "selection_mode": "llm_and_greedy" if client is not None else "greedy",
        "selection_model_sha256": model_hash,
        "prompt_sha256": prompt_hash,
        "generator_version": GENERATOR_VERSION,
        "segmentation_sha256": segmentation_hash,
        "source_sha256": source_hash,
        "output_basename": output_path.name,
    }
    config_hash = hashlib.sha256(
        json.dumps(
            config, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if (output_path.exists() or plan_path.exists()) and not overwrite:
        try:
            previous = json.loads(plan_path.read_text(encoding="utf-8"))
            if (
                previous.get("domain") == spec.name
                and previous.get("source_schema_version") == spec.segmentation_schema
                and previous.get("config_sha256") == config_hash
                and previous.get("output_sha256") == _sha256(output_path)
            ):
                actual = _probe_video(output_path, decode=True)
                if (
                    actual["frame_count"] == previous["output_frame_count"]
                    and abs(actual["duration_sec"] - previous["total_sec"])
                    < 1 / actual["fps"]
                    and actual["duration_sec"] <= max_total_sec + 1e-7
                ):
                    return previous
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise FileExistsError(
            "Existing episode or plan is stale or invalid; "
            "use overwrite=True to regenerate"
        )
    selected_region = region or choose_region(
        segments, target_sec, speed, blank_sec, domain=spec.name,
    )
    scoped = [
        segment for segment in segments
        if _region_of(segment, domain=spec.name) == selected_region
    ]
    greedy = validate_visits(
        pick_visits_greedy(
            scoped, target_sec, blank_sec, speed, max_total_sec, domain=spec.name,
        ),
        scoped,
        speed,
        blank_sec,
        max_total_sec,
        domain=spec.name,
    )
    options = [("greedy", greedy)] if len(greedy) >= spec.min_visits else []
    llm_status, failure_reason = "disabled", None
    if client is not None:
        try:
            llm = validate_visits(
                _llm_visits(
                    client, scoped, selected_region, parameters,
                    info["duration_sec"],
                    domain=spec.name,
                ),
                scoped,
                speed,
                blank_sec,
                max_total_sec,
                domain=spec.name,
            )
            if len(llm) < spec.min_visits:
                llm_status = "invalid"
                failure_reason = f"fewer_than_{spec.min_visits}_valid_visits"
            else:
                options.insert(0, ("llm", llm))
                llm_status = "valid"
        except Exception as error:
            llm_status, failure_reason = "failed", type(error).__name__
            logger.warning(
                "LLM visit selection failed (%s); considering the greedy plan",
                failure_reason,
            )
    if not options:
        raise ValueError(
            f"No plan contains {spec.min_visits} valid, separated visits "
            "in the selected region"
        )
    method, visits = min(
        options,
        key=lambda item: _plan_score(item[1], target_sec, speed, blank_sec),
    )
    estimated = estimate_output_duration(visits, speed, blank_sec)
    visits, blank_frames, expected_frames = _timeline(visits, info["fps"], speed, blank_sec)
    executable = _ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".storm-revisit-", dir=output_path.parent
    ) as directory:
        temporary = Path(directory)
        combined, actual = _assemble(
            source_video, visits, info, blank_frames, parameters,
            temporary, executable,
        )
        if (
            actual["frame_count"] != expected_frames
            or abs(actual["fps"] - info["fps"]) > 1e-3
        ):
            raise RuntimeError(
                "Encoded frame count or frame rate differs from the planned timeline"
            )
        if actual["duration_sec"] > max_total_sec + 1e-7:
            raise RuntimeError("Encoded episode exceeds max_total_sec")
        hidden = _hidden_changes(visits, segments)
        plan = {
            "generator_version": GENERATOR_VERSION,
            "domain": spec.name,
            "source_schema_version": spec.segmentation_schema,
            "source_video_id": source_video_id,
            "episode_id": episode_id,
            "source_sha256": source_hash,
            "output_video": output_path.name,
            "segmentation_sha256": segmentation_hash,
            "config_sha256": config_hash,
            "output_sha256": _sha256(combined),
            "parameters": parameters,
            "region": selected_region,
            "processed_interval": [interval_start, interval_end],
            "interval_convention": "[start_sec, end_sec)",
            "total_sec": actual["duration_sec"],
            "estimated_total_sec": estimated,
            "duration_error_sec": actual["duration_sec"] - estimated,
            "frame_rounding": "floor_each_visit_and_gap",
            "fps": actual["fps"],
            "output_frame_count": actual["frame_count"],
            "blank_sec": blank_sec,
            "blank_output_sec": blank_frames / info["fps"],
            "speed": speed,
            "fade_sec": fade_sec,
            "visit_count": len(visits),
            "transition_count": len(visits) - 1,
            "assembly_segment_count": (
                len(visits) + (len(visits) - 1 if blank_frames else 0)
            ),
            "hidden_change_segment_count": sum(
                item["change_segment_count"] for item in hidden
            ),
            "selection_source": method,
            "llm_status": llm_status,
            "llm_failure_reason": failure_reason,
            "selection_candidates": [
                {
                    "method": name,
                    "visit_count": len(candidate),
                    "estimated_total_sec": estimate_output_duration(
                        candidate, speed, blank_sec
                    ),
                }
                for name, candidate in options
            ],
            "visits": visits,
            "hidden_transitions": hidden,
        }
        staged_plan = temporary / "episode.plan.json"
        write_json(staged_plan, plan)
        # Each replacement is atomic. A crash between them leaves a detectable
        # output hash/configuration mismatch rather than a silently resumed pair.
        os.replace(combined, output_path)
        os.replace(staged_plan, plan_path)
    logger.info(
        "Assembled %d visits using %s selection (%.3f seconds)",
        len(visits), method, actual["duration_sec"],
    )
    return plan
