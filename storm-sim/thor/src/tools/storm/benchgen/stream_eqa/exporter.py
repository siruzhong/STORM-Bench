"""Export accepted Benchgen run artifacts as a StreamEQA dataset."""
from __future__ import annotations

import errno
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .contracts import (
    EPISODE_KEYS,
    QUESTION_TYPES,
    ContractError,
    validate_episode_document,
)
from .event_adapter import build_events, public_events
from .questions import build_questions


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        raise ValueError(f"unsafe empty path component from {value!r}")
    return cleaned


def _episode_id(plan: dict[str, Any]) -> str:
    scene = _safe_component(str(plan["recipe"]["scene"]))
    seed = int(plan["recipe"]["seed"])
    plan_id = _safe_component(str(plan["plan_id"]))[:12]
    return f"STORM_{scene}_seed{seed}_{plan_id}_visit01"


def _materialize_video(source: Path, target: Path, link_mode: str) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing source rollout: {source}")
    if target.exists():
        raise FileExistsError(f"refusing to replace exported video: {target}")
    if link_mode == "copy":
        shutil.copy2(source, target)
        return
    if link_mode != "hardlink":
        raise ValueError(f"unsupported link mode: {link_mode}")
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


def _load_accepted_episode(
    run_dir: Path, episode: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    episode_dir = run_dir / str(episode["directory"])
    validation = _read_json(episode_dir / "episode_validation.json")
    trace = _read_json(episode_dir / "execution_trace.json")
    if not validation.get("accepted") or not trace.get("accepted"):
        raise ContractError(f"episode is not accepted: {episode_dir}")
    if trace.get("failed_frames"):
        raise ContractError(f"episode has failed frames: {episode_dir}")
    plan = _read_json(episode_dir / "episode_plan.json")
    profile = _read_json(episode_dir / "scene_profile.json")
    if plan["plan_id"] != episode["plan_id"]:
        raise ContractError(f"manifest plan id mismatch: {episode_dir}")
    return episode_dir, plan, trace, profile


def export_run(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    sample_fps: float = 1.0,
    link_mode: str = "hardlink",
) -> dict[str, Any]:
    """Export one accepted run without modifying its authoritative artifacts."""
    run = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
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

    documents: list[tuple[Path, dict[str, Any]]] = []
    for manifest_episode in manifest_episodes:
        episode_dir, plan, trace, profile = _load_accepted_episode(
            run, manifest_episode,
        )
        episode_id = _episode_id(plan)
        scene = _safe_component(str(plan["recipe"]["scene"]))
        group_dir = output / scene
        group_dir.mkdir(parents=True, exist_ok=True)
        video_path = group_dir / f"{episode_id}.mp4"
        json_path = group_dir / f"{episode_id}.json"
        _materialize_video(episode_dir / "rollout.mp4", video_path, link_mode)
        private_events = build_events(plan, trace, profile)
        fps = int(plan["trajectory"]["fps"])
        frame_count = len(plan["trajectory"]["frames"])
        if fps <= 0 or frame_count <= 0:
            raise ContractError("episode trajectory has invalid timing")
        duration = round(frame_count / fps, 2)
        questions = build_questions(episode_id, private_events, duration)
        document = {
            "episode_id": episode_id,
            "video_path": str(video_path.resolve()),
            "duration_sec": duration,
            "sample_fps": float(sample_fps),
            "events": public_events(private_events),
            "questions": questions,
        }
        _write_json(json_path, document)
        validate_episode_document(document, json_path, video_path)
        documents.append((json_path, document))

    documents.sort(key=lambda pair: pair[1]["episode_id"])
    all_questions = [
        question
        for _, document in documents
        for question in document["questions"]
    ]
    question_ids = [question["id"] for question in all_questions]
    if len(question_ids) != len(set(question_ids)):
        raise ContractError("question ids are not globally unique")
    questions_path = output / "questions.jsonl"
    _write_jsonl(questions_path, all_questions)
    type_counts = Counter(
        question["question_type"] for question in all_questions
    )
    summary = {
        "video_count": len(documents),
        "question_count": len(all_questions),
        "question_type_counts": {
            question_type: type_counts.get(question_type, 0)
            for question_type in QUESTION_TYPES
        },
        "questions_jsonl": str(questions_path.resolve()),
    }
    _write_json(output / "summary.json", summary)
    validate_dataset(output)
    return summary


def validate_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Validate pairing, per-video schema, global JSONL, and summary counts."""
    output = Path(output_dir).expanduser().resolve()
    summary_path = output / "summary.json"
    questions_path = output / "questions.jsonl"
    if not summary_path.is_file() or not questions_path.is_file():
        raise ContractError("dataset is missing summary.json or questions.jsonl")
    episode_jsons = []
    for path in sorted(output.rglob("*.json")):
        if path == summary_path or path.name.endswith(".clips.json"):
            continue
        candidate = _read_json(path)
        if isinstance(candidate, dict) and set(candidate) == EPISODE_KEYS:
            episode_jsons.append((path, candidate))
        else:
            raise ContractError(f"unexpected JSON artifact in dataset: {path}")
    if not episode_jsons:
        raise ContractError("dataset has no episode JSON files")
    episode_jsons.sort(key=lambda pair: pair[1]["episode_id"])
    all_questions = []
    expected_videos = set()
    for json_path, document in episode_jsons:
        video_path = json_path.with_suffix(".mp4")
        validate_episode_document(document, json_path, video_path)
        expected_videos.add(video_path.resolve())
        all_questions.extend(document["questions"])
    actual_videos = {path.resolve() for path in output.rglob("*.mp4")}
    if actual_videos != expected_videos:
        raise ContractError("MP4/JSON stems are not paired one-to-one")
    jsonl_questions = []
    with questions_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ContractError(f"blank questions.jsonl line {line_number}")
            jsonl_questions.append(json.loads(line))
    if jsonl_questions != all_questions:
        raise ContractError("questions.jsonl differs from per-episode questions")
    summary = _read_json(summary_path)
    expected_counts = Counter(
        question["question_type"] for question in all_questions
    )
    if summary.get("video_count") != len(episode_jsons):
        raise ContractError("summary video_count is incorrect")
    if summary.get("question_count") != len(all_questions):
        raise ContractError("summary question_count is incorrect")
    expected_type_counts = {
        question_type: expected_counts.get(question_type, 0)
        for question_type in QUESTION_TYPES
    }
    if summary.get("question_type_counts") != expected_type_counts:
        raise ContractError("summary question_type_counts are incorrect")
    if Path(summary.get("questions_jsonl", "")) != questions_path.resolve():
        raise ContractError("summary questions_jsonl path is incorrect")
    return summary
