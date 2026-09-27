"""Export one full Benchgen Floor as multiple StreamEQA key clips."""
from __future__ import annotations

import json
import math
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .clip_questions import build_clip_questions
from .contracts import EPISODE_KEYS, ContractError, validate_episode_document
from .event_adapter import build_events, public_events
from .exporter import (
    _episode_id,
    _load_accepted_episode,
    _read_json,
    _safe_component,
    _write_json,
)
from .questions import _surface


CLIP_PLAN_KEYS = {
    "video_path", "plan_path", "fps", "speed", "blank_sec", "region", "clips",
}

CLIP_KEYS = {
    "visit_index", "src_start_sec", "src_end_sec", "out_start_sec",
    "out_end_sec", "label", "clip_frames", "clip_path",
}


@dataclass(frozen=True)
class ClipSpec:
    """One contiguous source window and the key events it contains."""

    visit_index: int
    event_indices: tuple[int, ...]
    src_start_sec: float
    src_end_sec: float
    label: str


@dataclass(frozen=True)
class VideoInfo:
    """Properties of one rendered key-clip video."""

    frames: int
    duration_sec: float
    codec_name: str = ""
    width: int = 0
    height: int = 0
    fps: str = ""


ClipRenderer = Callable[[Path, Path, float, float, float, int], VideoInfo]


def _partition_sizes(event_count: int, clip_count: int) -> list[int]:
    """Spread larger groups across the episode instead of clustering them."""
    if event_count < clip_count or clip_count <= 0:
        raise ValueError("invalid event/clip counts")
    base, extra = divmod(event_count, clip_count)
    sizes = [base] * clip_count
    for index in range(extra):
        position = math.floor((index + 0.5) * clip_count / extra)
        sizes[min(position, clip_count - 1)] += 1
    return sizes


def _event_label(event: dict[str, Any]) -> str:
    subject = event["subject"]
    surface = _surface(event)
    if event["event_type"] == "disappearance":
        return f"{subject}从{surface}的原位置消失"
    return f"{subject}重新出现在{surface}的原位置"


def build_clip_specs(
    events: list[dict[str, Any]],
    *,
    source_frame_count: int,
    fps: int,
    target_clip_count: int = 6,
    context_sec: float = 0.75,
) -> list[ClipSpec]:
    """Partition ordered events into 4–8 reference-style key clips."""
    if fps <= 0 or source_frame_count <= 0:
        raise ValueError("source timing must be positive")
    if context_sec < 0:
        raise ValueError("context_sec must be non-negative")
    ordered = sorted(events, key=lambda item: (item["start_sec"], item["end_sec"]))
    if len(ordered) < 4:
        raise ValueError("at least four events are required for key-clip export")
    clip_count = min(max(4, target_clip_count), 8, len(ordered))
    sizes = _partition_sizes(len(ordered), clip_count)
    specs = []
    cursor = 0
    for visit_index, size in enumerate(sizes, start=1):
        group = ordered[cursor:cursor + size]
        cursor += size
        start_frame = max(
            0,
            math.floor((float(group[0]["start_sec"]) - context_sec) * fps),
        )
        end_frame = min(
            source_frame_count,
            math.ceil((float(group[-1]["end_sec"]) + context_sec) * fps),
        )
        if end_frame <= start_frame:
            raise ValueError("key clip has an empty source window")
        specs.append(ClipSpec(
            visit_index=visit_index,
            event_indices=tuple(int(event["_index"]) for event in group),
            src_start_sec=round(start_frame / fps, 6),
            src_end_sec=round(end_frame / fps, 6),
            label="；".join(_event_label(event) for event in group),
        ))
    if cursor != len(ordered):
        raise AssertionError("not every event was assigned to a key clip")
    return specs


def localize_events(
    events: list[dict[str, Any]], spec: ClipSpec, speed: float,
) -> list[dict[str, Any]]:
    """Convert full-video event spans to clip-local, speed-adjusted time."""
    if speed <= 0:
        raise ValueError("speed must be positive")
    by_index = {int(event["_index"]): event for event in events}
    localized = []
    for event_index in spec.event_indices:
        source = by_index[event_index]
        event = dict(source)
        event["start_sec"] = round(
            (float(source["start_sec"]) - spec.src_start_sec) / speed,
            2,
        )
        event["end_sec"] = round(
            (float(source["end_sec"]) - spec.src_start_sec) / speed,
            2,
        )
        if event["start_sec"] < -0.01 or event["end_sec"] < event["start_sec"]:
            raise ValueError("localized event lies outside its key clip")
        event["start_sec"] = max(0.0, event["start_sec"])
        localized.append(event)
    return localized


def probe_video(path: Path) -> VideoInfo:
    """Read exact video properties with ffprobe."""
    process = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(process.stdout)
    if not payload.get("streams"):
        raise ContractError(f"key clip has no video stream: {path}")
    stream = payload["streams"][0]
    raw_frames = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_frames in (None, "N/A"):
        raise ContractError(f"cannot determine key-clip frame count: {path}")
    return VideoInfo(
        frames=int(raw_frames),
        duration_sec=float(payload["format"]["duration"]),
        codec_name=str(stream.get("codec_name") or ""),
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=str(stream.get("r_frame_rate") or ""),
    )


def render_key_clip(
    source: Path,
    target: Path,
    start_sec: float,
    end_sec: float,
    speed: float,
    fps: int,
) -> VideoInfo:
    """Cut, speed up, and encode one frame-accurate key clip."""
    if target.exists():
        raise FileExistsError(f"refusing to replace key clip: {target}")
    source_duration = end_sec - start_sec
    if source_duration <= 0 or speed <= 0 or fps <= 0:
        raise ValueError("invalid key-clip render timing")
    target.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-xerror", "-ss", f"{start_sec:.6f}",
            "-t", f"{source_duration:.6f}", "-i", str(source), "-map", "0:v:0",
            "-an", "-vf", f"setpts=(PTS-STARTPTS)/{speed},fps={fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target),
        ],
        text=True,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(
            f"ffmpeg failed for {target}: {process.stderr[-2000:]}"
        )
    info = probe_video(target)
    if info.frames <= 0 or info.duration_sec <= 0:
        raise ContractError(f"rendered key clip is empty: {target}")
    return info


def _base_episode_id(plan: dict[str, Any]) -> str:
    episode_id = _episode_id(plan)
    if not episode_id.endswith("_visit01"):
        raise AssertionError("unexpected full-episode id format")
    return episode_id.removesuffix("_visit01")


def export_keyclip_run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    sample_fps: float = 1.0,
    speed: float = 1.5,
    blank_sec: float = 0.2,
    target_clip_count: int = 6,
    context_sec: float = 0.75,
    renderer: ClipRenderer = render_key_clip,
) -> dict[str, Any]:
    """Export each Floor as one full episode containing multiple key clips."""
    run = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    if sample_fps <= 0 or speed <= 0 or blank_sec < 0:
        raise ValueError("invalid key-clip export settings")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(run / "manifest.json")
    batch_validation = _read_json(run / "batch_validation.json")
    if not manifest.get("accepted") or not batch_validation.get("accepted"):
        raise ContractError("only an accepted batch can be exported")
    manifest_episodes = manifest.get("episodes")
    if not isinstance(manifest_episodes, list) or not manifest_episodes:
        raise ContractError("accepted manifest has no episodes")

    for manifest_episode in manifest_episodes:
        episode_dir, plan, trace, profile = _load_accepted_episode(
            run, manifest_episode,
        )
        scene = _safe_component(str(plan["recipe"]["scene"]))
        group_dir = output / scene
        group_dir.mkdir(parents=True, exist_ok=True)
        base_id = _base_episode_id(plan)
        source_video = (episode_dir / "rollout.mp4").resolve()
        fps = int(plan["trajectory"]["fps"])
        frame_count = len(plan["trajectory"]["frames"])
        private_events = build_events(plan, trace, profile)
        all_labels = list(dict.fromkeys(
            event["subject"] for event in private_events
        ))
        specs = build_clip_specs(
            private_events,
            source_frame_count=frame_count,
            fps=fps,
            target_clip_count=target_clip_count,
            context_sec=context_sec,
        )
        clip_records = []
        output_cursor = 0.0
        for spec in specs:
            clip_id = f"{base_id}_visit{spec.visit_index:02d}"
            video_path = group_dir / f"{clip_id}.mp4"
            json_path = group_dir / f"{clip_id}.json"
            info = renderer(
                source_video,
                video_path,
                spec.src_start_sec,
                spec.src_end_sec,
                speed,
                fps,
            )
            duration = round(info.frames / fps, 2)
            if abs(info.duration_sec - duration) > 0.1:
                raise ContractError(
                    f"key-clip duration/frame mismatch: {video_path}"
                )
            local_events = localize_events(private_events, spec, speed)
            questions = build_clip_questions(
                clip_id,
                local_events,
                duration,
                all_labels,
                spec.visit_index,
            )
            document = {
                "episode_id": clip_id,
                "video_path": str(video_path.resolve()),
                "duration_sec": duration,
                "sample_fps": float(sample_fps),
                "events": public_events(local_events),
                "questions": questions,
            }
            _write_json(json_path, document)
            validate_episode_document(document, json_path, video_path)
            output_start = round(output_cursor, 2)
            output_end = round(output_start + duration, 2)
            clip_records.append({
                "visit_index": spec.visit_index,
                "src_start_sec": round(spec.src_start_sec, 2),
                "src_end_sec": round(spec.src_end_sec, 2),
                "out_start_sec": output_start,
                "out_end_sec": output_end,
                "label": spec.label,
                "clip_frames": info.frames,
                "clip_path": str(video_path.resolve()),
            })
            output_cursor = output_end + blank_sec
        clip_plan = {
            "video_path": str(source_video),
            "plan_path": str((episode_dir / "episode_plan.json").resolve()),
            "fps": fps,
            "speed": float(speed),
            "blank_sec": float(blank_sec),
            "region": f"{scene} 场景",
            "clips": clip_records,
        }
        _write_json(group_dir / f"{base_id}.clips.json", clip_plan)

    return validate_keyclip_dataset(output)


def validate_keyclip_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Validate reference-style full-episode plans and all clip pairs."""
    output = Path(output_dir).expanduser().resolve()
    plans = sorted(output.glob("*/*.clips.json"))
    if not plans:
        raise ContractError("key-clip dataset has no .clips.json plans")
    expected_jsons = set()
    expected_videos = set()
    all_questions = []
    total_events = 0
    clip_counts = []
    for plan_path in plans:
        plan = _read_json(plan_path)
        if not isinstance(plan, dict) or set(plan) != CLIP_PLAN_KEYS:
            raise ContractError(f"invalid key-clip plan keys: {plan_path}")
        if not Path(plan["video_path"]).is_file():
            raise ContractError(f"full source video is missing: {plan_path}")
        if not Path(plan["plan_path"]).is_file():
            raise ContractError(f"full source plan is missing: {plan_path}")
        fps = int(plan["fps"])
        speed = float(plan["speed"])
        blank_sec = float(plan["blank_sec"])
        if fps <= 0 or speed <= 0 or blank_sec < 0:
            raise ContractError(f"invalid key-clip plan timing: {plan_path}")
        clips = plan["clips"]
        if not isinstance(clips, list) or not 4 <= len(clips) <= 8:
            raise ContractError("each full episode must contain 4–8 key clips")
        clip_counts.append(len(clips))
        expected_base = plan_path.name.removesuffix(".clips.json")
        output_cursor = 0.0
        previous_source_start = -1.0
        for expected_visit, clip in enumerate(clips, start=1):
            if not isinstance(clip, dict) or set(clip) != CLIP_KEYS:
                raise ContractError(f"invalid clip keys: {plan_path}")
            if clip["visit_index"] != expected_visit:
                raise ContractError("clip visit indices are not sequential")
            expected_stem = f"{expected_base}_visit{expected_visit:02d}"
            video_path = Path(clip["clip_path"]).resolve()
            json_path = video_path.with_suffix(".json")
            if video_path.parent != plan_path.parent.resolve():
                raise ContractError("key clip is outside its Floor directory")
            if video_path.stem != expected_stem:
                raise ContractError("key-clip stem does not match visit index")
            if not json_path.is_file():
                raise ContractError(f"paired key-clip JSON is missing: {json_path}")
            if int(clip["clip_frames"]) <= 0:
                raise ContractError("key clip has a non-positive frame count")
            src_start = float(clip["src_start_sec"])
            src_end = float(clip["src_end_sec"])
            if src_start < previous_source_start or src_start >= src_end:
                raise ContractError("key-clip source windows are not ordered")
            previous_source_start = src_start
            out_start = float(clip["out_start_sec"])
            out_end = float(clip["out_end_sec"])
            if abs(out_start - output_cursor) > 0.05 or out_end <= out_start:
                raise ContractError("key-clip output timeline is inconsistent")
            expected_duration = int(clip["clip_frames"]) / fps
            if abs((out_end - out_start) - expected_duration) > 0.05:
                raise ContractError("key-clip output span does not match frames")
            output_cursor = round(out_end + blank_sec, 2)
            document = _read_json(json_path)
            if not isinstance(document, dict) or set(document) != EPISODE_KEYS:
                raise ContractError(f"invalid key-clip episode JSON: {json_path}")
            validate_episode_document(document, json_path, video_path)
            if abs(float(document["duration_sec"]) - expected_duration) > 0.05:
                raise ContractError("clip JSON duration does not match clip frames")
            expected_jsons.add(json_path)
            expected_videos.add(video_path)
            total_events += len(document["events"])
            all_questions.extend(document["questions"])
    actual_jsons = {
        path.resolve() for path in output.glob("*/*.json")
        if not path.name.endswith(".clips.json")
    }
    actual_videos = {path.resolve() for path in output.glob("*/*.mp4")}
    if actual_jsons != expected_jsons or actual_videos != expected_videos:
        raise ContractError("key-clip plan and paired artifacts differ")
    question_ids = [question["id"] for question in all_questions]
    if len(question_ids) != len(set(question_ids)):
        raise ContractError("key-clip question ids are not globally unique")
    type_counts = Counter(question["question_type"] for question in all_questions)
    return {
        "full_episode_count": len(plans),
        "key_clip_count": len(expected_videos),
        "event_count": total_events,
        "question_count": len(all_questions),
        "clips_per_episode": dict(sorted(Counter(clip_counts).items())),
        "question_type_counts": dict(sorted(type_counts.items())),
    }
