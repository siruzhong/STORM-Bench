"""Strict local validation for the StreamEQA interchange contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


EVENT_TYPES = {
    "appearance",
    "disappearance",
    "movement",
    "attribute_change",
    "relation_change",
    "occlusion",
    "interaction",
    "observation",
}

QUESTION_TYPES = (
    "factual_retrieval",
    "current_state",
    "state_change",
    "object_tracking",
    "temporal_reasoning",
    "history_aggregation",
)

EPISTEMIC_STATUS = {"known", "uncertain", "unanswerable"}

EPISODE_KEYS = {
    "episode_id",
    "video_path",
    "duration_sec",
    "sample_fps",
    "events",
    "questions",
}

EVENT_KEYS = {
    "start_sec",
    "end_sec",
    "event_type",
    "subject",
    "before_state",
    "after_state",
    "location",
    "identity_cues",
    "certainty",
    "description",
}

QUESTION_KEYS = {
    "id",
    "episode_id",
    "query_time",
    "question_type",
    "question_subtype",
    "video_evidence",
    "question",
    "options",
    "answer_index",
    "evidence_spans",
    "diagnostics",
    "diagnostic_rationale",
}


class ContractError(ValueError):
    """Raised when an exported artifact does not satisfy the shared schema."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    return float(value)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value


def validate_event(event: Any, duration: float) -> tuple[float, float]:
    if not isinstance(event, dict) or set(event) != EVENT_KEYS:
        raise ContractError("event keys do not match the StreamEQA contract")
    start = _number(event["start_sec"], "event.start_sec")
    end = _number(event["end_sec"], "event.end_sec")
    if start < 0 or start > end or end > duration + 0.05:
        raise ContractError(f"invalid event span: {start}-{end}")
    if event["event_type"] not in EVENT_TYPES:
        raise ContractError(f"unsupported event type: {event['event_type']}")
    for key in ("subject", "before_state", "after_state", "location", "description"):
        _nonempty_string(event[key], f"event.{key}")
    cues = event["identity_cues"]
    if not isinstance(cues, list) or not cues or any(
        not isinstance(item, str) or not item.strip() for item in cues
    ):
        raise ContractError("event.identity_cues must contain strings")
    if event["certainty"] not in {"known", "uncertain"}:
        raise ContractError("event.certainty must be known or uncertain")
    return round(start, 2), round(end, 2)


def _span_matches(
    span: tuple[float, float], event_spans: Iterable[tuple[float, float]],
) -> bool:
    return any(
        abs(span[0] - start) <= 0.05 and abs(span[1] - end) <= 0.05
        for start, end in event_spans
    )


def validate_question(
    question: Any,
    episode_id: str,
    duration: float,
    event_spans: Iterable[tuple[float, float]],
) -> None:
    if not isinstance(question, dict) or set(question) != QUESTION_KEYS:
        raise ContractError("question keys do not match the StreamEQA contract")
    _nonempty_string(question["id"], "question.id")
    if question["episode_id"] != episode_id:
        raise ContractError("question.episode_id does not match its episode")
    if question["question_type"] not in QUESTION_TYPES:
        raise ContractError(f"unsupported question type: {question['question_type']}")
    for key in ("question_subtype", "video_evidence", "question"):
        _nonempty_string(question[key], f"question.{key}")
    options = question["options"]
    if not isinstance(options, list) or len(options) != 4:
        raise ContractError("question.options must contain exactly four choices")
    if any(not isinstance(item, str) or not item.strip() for item in options):
        raise ContractError("question options must be non-empty strings")
    if len({item.strip() for item in options}) != 4:
        raise ContractError("question options must be distinct")
    answer_index = question["answer_index"]
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise ContractError("question.answer_index must be an integer")
    if not 0 <= answer_index < 4:
        raise ContractError("question.answer_index must be between zero and three")
    query_time = _number(question["query_time"], "question.query_time")
    if query_time < 0 or query_time > duration + 0.05:
        raise ContractError("question.query_time lies outside the video")
    spans = question["evidence_spans"]
    if not isinstance(spans, list) or not spans:
        raise ContractError("question.evidence_spans must not be empty")
    available_spans = tuple(event_spans)
    for raw in spans:
        if not isinstance(raw, list) or len(raw) != 2:
            raise ContractError("each evidence span must be [start, end]")
        start = _number(raw[0], "evidence start")
        end = _number(raw[1], "evidence end")
        if start < 0 or start > end or end > query_time + 0.01:
            raise ContractError("evidence span is outside the query history")
        if not _span_matches((start, end), available_spans):
            raise ContractError("evidence span does not match an exported event")
    diagnostics = question["diagnostics"]
    expected_diagnostics = {
        "revision_count", "old_answer_indices", "epistemic_status",
        "uncertainty_sources",
    }
    if not isinstance(diagnostics, dict) or set(diagnostics) != expected_diagnostics:
        raise ContractError("question diagnostics have invalid keys")
    revision_count = diagnostics["revision_count"]
    if isinstance(revision_count, bool) or not isinstance(revision_count, int):
        raise ContractError("revision_count must be an integer")
    if revision_count < 0:
        raise ContractError("revision_count must be non-negative")
    old_indices = diagnostics["old_answer_indices"]
    if not isinstance(old_indices, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 4
        for item in old_indices
    ):
        raise ContractError("old_answer_indices are invalid")
    if diagnostics["epistemic_status"] not in EPISTEMIC_STATUS:
        raise ContractError("invalid epistemic_status")
    uncertainty = diagnostics["uncertainty_sources"]
    if not isinstance(uncertainty, list) or any(
        not isinstance(item, str) for item in uncertainty
    ):
        raise ContractError("uncertainty_sources must be strings")
    if diagnostics["epistemic_status"] == "known" and uncertainty:
        raise ContractError("known questions cannot have uncertainty sources")
    rationale = question["diagnostic_rationale"]
    if not isinstance(rationale, dict) or set(rationale) != {"volatility", "uncertainty"}:
        raise ContractError("diagnostic_rationale has invalid keys")
    _nonempty_string(rationale["volatility"], "diagnostic_rationale.volatility")
    _nonempty_string(rationale["uncertainty"], "diagnostic_rationale.uncertainty")


def validate_episode_document(
    document: Any, json_path: Path, video_path: Path,
) -> None:
    if not isinstance(document, dict) or set(document) != EPISODE_KEYS:
        raise ContractError("episode keys do not match the StreamEQA contract")
    episode_id = _nonempty_string(document["episode_id"], "episode_id")
    if episode_id != json_path.stem or episode_id != video_path.stem:
        raise ContractError("episode_id must equal the paired file stem")
    if Path(document["video_path"]) != video_path.resolve():
        raise ContractError("video_path must point to the paired exported MP4")
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise ContractError("paired video is missing or empty")
    duration = _number(document["duration_sec"], "duration_sec")
    sample_fps = _number(document["sample_fps"], "sample_fps")
    if duration <= 0 or sample_fps <= 0:
        raise ContractError("duration_sec and sample_fps must be positive")
    events = document["events"]
    questions = document["questions"]
    if not isinstance(events, list) or not events:
        raise ContractError("events must be a non-empty list")
    if not isinstance(questions, list) or not questions:
        raise ContractError("questions must be a non-empty list")
    event_spans = tuple(validate_event(event, duration) for event in events)
    question_ids = []
    for question in questions:
        validate_question(question, episode_id, duration, event_spans)
        question_ids.append(question["id"])
    if len(question_ids) != len(set(question_ids)):
        raise ContractError("question ids are not unique within an episode")
