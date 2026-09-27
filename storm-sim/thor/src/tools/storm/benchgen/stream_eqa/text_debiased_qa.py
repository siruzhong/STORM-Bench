"""Rebuild QA from source videos and accepted simulator traces.

Write independent QA files, link or copy the videos, and check answer-position
and text-only biases before accepting the export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import reference_qa as v1
from .event_adapter import build_events
from .exporter import _episode_id


VERSION = "text_debiased_v2"
DEFAULT_SEED = 20260824
QUESTION_TYPES = v1.QUESTION_TYPES
EPISODE_KEYS = v1.EPISODE_KEYS
QUESTION_KEYS = v1.QUESTION_KEYS
ALLOWED_UNCERTAINTY_SOURCES = v1.ALLOWED_UNCERTAINTY_SOURCES
FORBIDDEN_UNCERTAINTY_SOURCES = v1.FORBIDDEN_UNCERTAINTY_SOURCES

EVIDENCE_OPTIONS = (
    "The recorded evidence does not uniquely support any concrete alternative",
    "The available observations do not establish one concrete alternative",
    "No concrete alternative is uniquely supported by the recorded evidence",
    "The indexed observations are insufficient to select one concrete alternative",
    "The visual record does not uniquely determine any concrete alternative",
    "None of the concrete alternatives is established by the available observations",
)

SUBTYPES = {
    "factual_retrieval": "observed_fact_query_v2",
    "current_state": "current_state_query_v2",
    "history_aggregation": "event_count_query_v2",
    "state_change": "state_transition_query_v2",
    "object_tracking": "object_tracking_query_v2",
    "temporal_reasoning": "event_order_query_v2",
}

FORBIDDEN_PROMPT_CUES = (
    "provably",
    "off-screen",
    "offscreen",
    "hidden action",
    "outside the visible frames",
    "outside the recorded view",
    "cannot be determined from the visible frames",
)

MODEL_INPUT_KEYS = {"question_id", "episode_id", "video_path", "query_time", "prompt"}

ROOM_SURFACES = {
    "kitchen": ("countertop", "dining table", "stove burner"),
    "living_room": ("side table", "coffee table", "dining table", "sofa", "TV stand"),
    "bedroom": ("desk", "bed", "side table", "shelf", "dresser"),
    "bathroom": ("hand-towel holder", "towel holder", "shelf", "bathtub", "sink", "floor"),
}


class ExportError(ValueError):
    """Raised when the v2 clone or QA contract is invalid."""


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(source: Path) -> dict[str, str]:
    paths = sorted((source / "meta_data" / "qa_results").glob("*/*.json"))
    paths.extend(
        path for path in (
            source / "questions.jsonl",
            source / "summary.json",
            source / "four_scene_summary.json",
        )
        if path.is_file()
    )
    return {str(path.relative_to(source)): _sha256(path) for path in paths}


def _room_for_scene(scene: str) -> str:
    match = re.search(r"(\d+)$", scene)
    if not match:
        raise ExportError(f"cannot infer room from scene: {scene}")
    number = int(match.group(1))
    if number < 200:
        return "kitchen"
    if number < 300:
        return "living_room"
    if number < 400:
        return "bedroom"
    return "bathroom"


def _event_points(event: dict[str, Any], maximum: int) -> tuple[float, float]:
    return v1._event_points(event, maximum)


def _label(event: dict[str, Any]) -> str:
    return v1._friendly_object(event)


def _surface(event: dict[str, Any]) -> str:
    return v1._surface(event)


def _ordered(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda item: (float(item["start_sec"]), float(item["end_sec"]), int(item["_index"])))


def _event_description(event: dict[str, Any]) -> str:
    action = "disappears from" if event["event_type"] == "disappearance" else "reappears at"
    return f"The {_label(event)} {action} its original {_surface(event)} position"


def _target_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in _ordered(events):
        result[str(event["_target_id"])].append(event)
    return dict(result)


def _event_pairs(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for target in _target_events(events).values():
        pending: dict[str, Any] | None = None
        for event in target:
            if event["event_type"] == "disappearance":
                pending = event
            elif pending is not None:
                result.append((pending, event))
                pending = None
    result.sort(key=lambda pair: float(pair[0]["start_sec"]))
    if len(result) < 4:
        raise ExportError("each episode needs at least four complete event pairs")
    return result


def _visible_at(events: list[dict[str, Any]], target_id: str, query_time: float, maximum: int) -> bool:
    visible = True
    for event in _ordered(events):
        if str(event["_target_id"]) != target_id:
            continue
        _, after = _event_points(event, maximum)
        if after <= query_time + 1e-6:
            visible = event["event_type"] == "appearance"
    return visible


def _evidence_for_target(
    events: list[dict[str, Any]], target_id: str, query_time: float, maximum: int,
) -> list[list[float]]:
    latest: dict[str, Any] | None = None
    for event in _ordered(events):
        if str(event["_target_id"]) != target_id:
            continue
        _, after = _event_points(event, maximum)
        if after <= query_time + 1e-6:
            latest = event
    if latest is None:
        return [[0.0, 0.0]]
    before, after = _event_points(latest, maximum)
    return [[before, before], [after, after]]


def _dedupe_spans(spans: Iterable[list[float]]) -> list[list[float]]:
    seen: set[tuple[float, float]] = set()
    result = []
    for span in spans:
        key = (float(span[0]), float(span[1]))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
    return sorted(result)


def _evidence_option(ordinal: int, seed: int) -> str:
    return EVIDENCE_OPTIONS[(ordinal * 5 + seed) % len(EVIDENCE_OPTIONS)]


def build_model_prompt(question: dict[str, Any]) -> str:
    """Build the only text payload allowed to enter an evaluation prompt."""
    letters = "ABCD"
    options = "\n".join(
        f"{letters[index]}. {option}"
        for index, option in enumerate(question["options"])
    )
    return (
        f"Question: {question['question']}\n"
        f"Options:\n{options}\n"
        "Answer with one option letter."
    )


def _draft(
    *,
    episode_id: str,
    question_type: str,
    query_time: float,
    question: str,
    concrete_options: list[str],
    correct: str,
    evidence_option: str,
    evidence_spans: list[list[float]],
    uncertainty_sources: list[str],
    events: list[dict[str, Any]],
    semantic_class: str,
) -> dict[str, Any]:
    uncertain = bool(uncertainty_sources)
    if len(concrete_options) != 3 or len(set(concrete_options)) != 3:
        raise ExportError("every v2 question needs three distinct concrete options")
    if uncertain:
        if correct != evidence_option:
            raise ExportError("uncertain v2 questions must select the common evidence option")
        distractors = list(concrete_options)
    else:
        if correct not in concrete_options:
            raise ExportError("known v2 correct answer must be a concrete option")
        distractors = [item for item in concrete_options if item != correct]
        distractors.append(evidence_option)
    return {
        "id": "",
        "episode_id": episode_id,
        "query_time": float(query_time),
        "question_type": question_type,
        "question_subtype": SUBTYPES[question_type],
        "video_evidence": "Use only the sampled observations indexed by evidence_spans.",
        "question": question,
        "options": [],
        "answer_index": -1,
        "evidence_spans": _dedupe_spans(evidence_spans),
        "diagnostics": {
            "epistemic_status": "uncertain" if uncertain else "known",
            "uncertainty_sources": sorted(uncertainty_sources),
        },
        "diagnostic_rationale": {
            "volatility": "The answer depends on the indexed temporal observations rather than the wording of the alternatives.",
            "uncertainty": (
                "The indexed observations do not uniquely support a concrete alternative."
                if uncertain
                else "The indexed observations uniquely support one concrete alternative."
            ),
        },
        "change_intensity": max(1, sum(float(event["start_sec"]) <= query_time + 1e-6 for event in events)),
        "_correct": correct,
        "_distractors": distractors,
        "_evidence_option": evidence_option,
        "_semantic_class": semantic_class,
    }


def _two_other_labels(events: list[dict[str, Any]], correct: str, ordinal: int) -> list[str]:
    labels = list(dict.fromkeys(_label(event) for event in _ordered(events)))
    alternatives = [item for item in labels if item != correct]
    if len(alternatives) < 2:
        raise ExportError("episode does not contain enough same-episode object distractors")
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    return rotated[:2]


def _location_options(scene: str, original: str, ordinal: int) -> list[str]:
    surfaces = list(dict.fromkeys([original, *ROOM_SURFACES[_room_for_scene(scene)]]))
    alternatives = [item for item in surfaces if item != original]
    if len(alternatives) < 2:
        alternatives.extend(item for item in ("floor", "supporting surface") if item not in alternatives and item != original)
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    return [
        f"At its original {original} position",
        f"At a different position on the {rotated[0]}",
        f"At a different position on the {rotated[1]}",
    ]


def _gap_query(pair: tuple[dict[str, Any], dict[str, Any]], maximum: int) -> tuple[float, float, float, float]:
    disappearance, appearance = pair
    dis_before, dis_after = _event_points(disappearance, maximum)
    app_before, _ = _event_points(appearance, maximum)
    query = float(max(dis_after, min(app_before, math.floor((dis_after + app_before) / 2))))
    return dis_before, dis_after, app_before, query


def _build_current_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str,
) -> dict[str, Any]:
    del scene
    desired = ordinal % 4
    targets = []
    labels: dict[str, str] = {}
    for event in _ordered(events):
        target_id = str(event["_target_id"])
        if target_id not in targets:
            targets.append(target_id)
        labels[target_id] = _label(event)
    candidates: list[tuple[int, str, str]] = []
    for query in range(5, maximum + 1):
        for left_index, left in enumerate(targets):
            for right in targets[left_index + 1:]:
                if labels[left] == labels[right]:
                    continue
                state = (int(_visible_at(events, left, query, maximum)) << 1) | int(_visible_at(events, right, query, maximum))
                if state == desired:
                    candidates.append((query, left, right))
    if not candidates:
        raise ExportError(f"no current-state candidate for configuration {desired}")
    centers = (12, 27, 42, 55)
    center = centers[(ordinal // 4) % len(centers)]
    rng = random.Random(DEFAULT_SEED + ordinal * 101)
    tie = {candidate: rng.random() for candidate in candidates}
    query, left, right = min(candidates, key=lambda item: (abs(item[0] - center), tie[item]))
    left_label, right_label = labels[left], labels[right]
    configurations = {
        3: f"Both the {left_label} and the {right_label} are visible at their original positions",
        2: f"The {left_label} is visible at its original position, while the {right_label} is not visible there",
        1: f"The {left_label} is not visible at its original position, while the {right_label} is visible at its original position",
        0: f"Neither the {left_label} nor the {right_label} is visible at its original position",
    }
    omit = (desired + 1 + ((ordinal // 4) % 3)) % 4
    if omit == desired:
        omit = (omit + 1) % 4
    concrete = [configurations[index] for index in range(4) if index != omit]
    spans = _evidence_for_target(events, left, query, maximum)
    spans.extend(_evidence_for_target(events, right, query, maximum))
    return _draft(
        episode_id=episode_id,
        question_type="current_state",
        query_time=query,
        question=f"At {query:.1f}s, which joint visibility statement about the {left_label} and the {right_label} is supported?",
        concrete_options=concrete,
        correct=configurations[desired],
        evidence_option=evidence_option,
        evidence_spans=spans,
        uncertainty_sources=[],
        events=events,
        semantic_class=f"joint_visibility_{desired}",
    )


def _build_factual_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str,
) -> dict[str, Any]:
    del scene
    event = _ordered(events)[(ordinal * 3) % len(events)]
    before, after = _event_points(event, maximum)
    correct = _label(event)
    action = "disappears from" if event["event_type"] == "disappearance" else "reappears at"
    concrete = [correct, *_two_other_labels(events, correct, ordinal)]
    return _draft(
        episode_id=episode_id,
        question_type="factual_retrieval",
        query_time=after,
        question=f"Which object {action} its original {_surface(event)} position by {after:.1f}s?",
        concrete_options=concrete,
        correct=correct,
        evidence_option=evidence_option,
        evidence_spans=[[before, before], [after, after]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"object:{correct}",
    )


def _build_location_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], question_type: str,
    ordinal: int, maximum: int, evidence_option: str, sources: list[str],
) -> dict[str, Any]:
    pair = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    disappearance, _ = pair
    dis_before, dis_after, _, query = _gap_query(pair, maximum)
    label = _label(disappearance)
    concrete = _location_options(scene, _surface(disappearance), ordinal)
    return _draft(
        episode_id=episode_id,
        question_type=question_type,
        query_time=query,
        question=f"Which location best describes where the {label} is at {query:.1f}s?",
        concrete_options=concrete,
        correct=evidence_option,
        evidence_option=evidence_option,
        evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="evidence_insufficient",
    )


def _build_history_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str,
) -> dict[str, Any]:
    del scene
    disappearances = [event for event in _ordered(events) if event["event_type"] == "disappearance"]
    count = 2 + ordinal % min(4, len(disappearances) - 1)
    selected = disappearances[:count]
    query = _event_points(selected[-1], maximum)[1]
    values = [count]
    for candidate in (count - 1, count + 1, count - 2, count + 2, 1, len(disappearances)):
        if 1 <= candidate <= len(disappearances) and candidate not in values:
            values.append(candidate)
        if len(values) == 3:
            break
    concrete = [f"{value} distinct disappearance events" for value in values]
    spans = [[_event_points(event, maximum)[1]] * 2 for event in selected]
    return _draft(
        episode_id=episode_id,
        question_type="history_aggregation",
        query_time=query,
        question=f"How many distinct visible disappearance events have completed by {query:.1f}s?",
        concrete_options=concrete,
        correct=f"{count} distinct disappearance events",
        evidence_option=evidence_option,
        evidence_spans=spans,
        uncertainty_sources=[],
        events=events,
        semantic_class=f"visible_disappearance_count_{count}",
    )


def _build_history_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    pair = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    disappearance, _ = pair
    dis_before, dis_after, app_before, query = _gap_query(pair, maximum)
    label = _label(disappearance)
    concrete = ["No separate manipulation", "Exactly one separate manipulation", "Exactly two separate manipulations"]
    return _draft(
        episode_id=episode_id,
        question_type="history_aggregation",
        query_time=query,
        question=f"How many separate manipulations of the {label} occur between {dis_after:.1f}s and {app_before:.1f}s?",
        concrete_options=concrete,
        correct=evidence_option,
        evidence_option=evidence_option,
        evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="evidence_insufficient",
    )


def _build_state_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str,
) -> dict[str, Any]:
    del scene
    ordered = _ordered(events)
    event = ordered[(ordinal * 3 + 1) % len(ordered)]
    before, after = _event_points(event, maximum)
    correct = _event_description(event)
    reverse_action = "reappears at" if event["event_type"] == "disappearance" else "disappears from"
    reverse = f"The {_label(event)} {reverse_action} its original {_surface(event)} position"
    other = next(
        _event_description(candidate)
        for candidate in ordered
        if _event_description(candidate) not in {correct, reverse}
    )
    return _draft(
        episode_id=episode_id,
        question_type="state_change",
        query_time=after,
        question=f"Which visible transition occurs between {before:.1f}s and {after:.1f}s?",
        concrete_options=[correct, reverse, other],
        correct=correct,
        evidence_option=evidence_option,
        evidence_spans=[[before, before], [after, after]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"transition:{event['event_type']}",
    )


def _build_state_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    event = next(item for item in _ordered(events) if item["event_type"] == "disappearance")
    before, after = _event_points(event, maximum)
    label = _label(event)
    concrete = [
        f"The {label} is picked up by the agent",
        f"The {label} slides away from the {_surface(event)}",
        f"Another object moves the {label}",
    ]
    return _draft(
        episode_id=episode_id,
        question_type="state_change",
        query_time=after,
        question=f"Which mechanism accounts for the {label}'s change between {before:.1f}s and {after:.1f}s?",
        concrete_options=concrete,
        correct=evidence_option,
        evidence_option=evidence_option,
        evidence_spans=[[before, before], [after, after]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="evidence_insufficient",
    )


def _build_object_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str,
) -> dict[str, Any]:
    del scene
    disappearance, appearance = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    dis_before, dis_after = _event_points(disappearance, maximum)
    _, app_after = _event_points(appearance, maximum)
    correct = _label(disappearance)
    concrete = [correct, *_two_other_labels(events, correct, ordinal)]
    return _draft(
        episode_id=episode_id,
        question_type="object_tracking",
        query_time=app_after,
        question=f"Which object returns to its original {_surface(disappearance)} position at {app_after:.1f}s?",
        concrete_options=concrete,
        correct=correct,
        evidence_option=evidence_option,
        evidence_spans=[[dis_before, dis_before], [dis_after, dis_after], [app_after, app_after]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"tracked_object:{correct}",
    )


def _build_object_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    disappearance, appearance = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    dis_before, dis_after = _event_points(disappearance, maximum)
    _, app_after = _event_points(appearance, maximum)
    label = _label(disappearance)
    concrete = [
        "The two observations show the same physical instance",
        "The later observation shows a different physical instance",
        "The earlier instance is destroyed and replaced before the later observation",
    ]
    return _draft(
        episode_id=episode_id,
        question_type="object_tracking",
        query_time=app_after,
        question=f"Which identity relationship is supported between the {label} observed before {dis_after:.1f}s and the matching {label} observed at {app_after:.1f}s?",
        concrete_options=concrete,
        correct=evidence_option,
        evidence_option=evidence_option,
        evidence_spans=[[dis_before, dis_before], [dis_after, dis_after], [app_after, app_after]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="evidence_insufficient",
    )


def _build_temporal_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str,
) -> dict[str, Any]:
    del scene
    ordered = _ordered(events)
    choices: list[tuple[int, str, list[str]]] = []
    for anchor_index in range(3, len(ordered)):
        correct = _event_description(ordered[anchor_index - 1])
        alternatives = []
        for candidate in ordered[:anchor_index - 1]:
            description = _event_description(candidate)
            if description != correct and description not in alternatives:
                alternatives.append(description)
        if len(alternatives) >= 2:
            choices.append((anchor_index, correct, alternatives))
    if not choices:
        raise ExportError("episode cannot form a three-way temporal predecessor question")
    anchor_index, correct, alternatives = choices[ordinal % len(choices)]
    anchor = ordered[anchor_index]
    previous = ordered[anchor_index - 1]
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    query = _event_points(anchor, maximum)[1]
    previous_after = _event_points(previous, maximum)[1]
    return _draft(
        episode_id=episode_id,
        question_type="temporal_reasoning",
        query_time=query,
        question=f"Which event occurs immediately before the {_event_description(anchor).removeprefix('The ')} by {query:.1f}s?",
        concrete_options=[correct, rotated[0], rotated[1]],
        correct=correct,
        evidence_option=evidence_option,
        evidence_spans=[[previous_after, previous_after], [query, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"predecessor:{previous['event_type']}",
    )


def _build_temporal_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, evidence_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    pair = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    disappearance, _ = pair
    dis_before, dis_after, app_before, query = _gap_query(pair, maximum)
    label = _label(disappearance)
    concrete = [
        f"The {label} is picked up before it is placed elsewhere",
        f"The {label} is placed elsewhere before it is rotated",
        f"The {label} is rotated before it is picked up",
    ]
    return _draft(
        episode_id=episode_id,
        question_type="temporal_reasoning",
        query_time=query,
        question=f"Which action ordering involving the {label} occurs between {dis_after:.1f}s and {app_before:.1f}s?",
        concrete_options=concrete,
        correct=evidence_option,
        evidence_option=evidence_option,
        evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="evidence_insufficient",
    )


def _build_question(
    *, episode_id: str, scene: str, events: list[dict[str, Any]],
    source_question: dict[str, Any], ordinal: int, maximum: int, seed: int,
) -> dict[str, Any]:
    question_type = str(source_question["question_type"])
    sources = list(source_question["diagnostics"]["uncertainty_sources"])
    uncertain = source_question["diagnostics"]["epistemic_status"] == "uncertain"
    evidence_option = _evidence_option(ordinal, seed)
    if uncertain and not sources:
        raise ExportError("source uncertain question has no uncertainty source")
    if not uncertain and sources:
        raise ExportError("source known question has uncertainty sources")
    if question_type == "current_state":
        if uncertain:
            return _build_location_uncertain(episode_id, scene, events, question_type, ordinal, maximum, evidence_option, sources)
        return _build_current_known(episode_id, scene, events, ordinal, maximum, evidence_option)
    if question_type == "factual_retrieval":
        if uncertain:
            return _build_location_uncertain(episode_id, scene, events, question_type, ordinal, maximum, evidence_option, sources)
        return _build_factual_known(episode_id, scene, events, ordinal, maximum, evidence_option)
    if question_type == "history_aggregation":
        if uncertain:
            return _build_history_uncertain(episode_id, scene, events, ordinal, maximum, evidence_option, sources)
        return _build_history_known(episode_id, scene, events, ordinal, maximum, evidence_option)
    if question_type == "state_change":
        if uncertain:
            return _build_state_uncertain(episode_id, scene, events, ordinal, maximum, evidence_option, sources)
        return _build_state_known(episode_id, scene, events, ordinal, maximum, evidence_option)
    if question_type == "object_tracking":
        if uncertain:
            return _build_object_uncertain(episode_id, scene, events, ordinal, maximum, evidence_option, sources)
        return _build_object_known(episode_id, scene, events, ordinal, maximum, evidence_option)
    if question_type == "temporal_reasoning":
        if uncertain:
            return _build_temporal_uncertain(episode_id, scene, events, ordinal, maximum, evidence_option, sources)
        return _build_temporal_known(episode_id, scene, events, ordinal, maximum, evidence_option)
    raise ExportError(f"unsupported question type: {question_type}")


def _assign_balanced_options(questions: list[dict[str, Any]], seed: int) -> dict[str, dict[str, int]]:
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        status = str(question["diagnostics"]["epistemic_status"])
        groups[(str(question["question_type"]), status)].append(question)
    allocations: dict[tuple[str, str], list[int]] = {}
    global_counts = [0, 0, 0, 0]
    for group in sorted(groups):
        count = len(groups[group])
        base, remainder = divmod(count, 4)
        allocation = [base] * 4
        for index in range(4):
            global_counts[index] += base
        tie = random.Random(f"{seed}:{group[0]}:{group[1]}")
        for _ in range(remainder):
            candidates = [index for index in range(4) if allocation[index] == base]
            minimum = min(global_counts[index] for index in candidates)
            candidates = [index for index in candidates if global_counts[index] == minimum]
            selected = tie.choice(candidates)
            allocation[selected] += 1
            global_counts[selected] += 1
        positions = [index for index, value in enumerate(allocation) for _ in range(value)]
        random.Random(f"{seed}:positions:{group[0]}:{group[1]}").shuffle(positions)
        allocations[group] = positions
    if global_counts != [len(questions) // 4] * 4:
        raise ExportError(f"global answer positions are not balanced: {global_counts}")
    # A globally balanced shuffle can still correlate answer position with
    # coarse query-time bins in a 300-item sample.  Search only over
    # within-group permutations, preserving every type/status quota while
    # removing that accidental metadata shortcut.
    best_positions: dict[tuple[str, str], list[int]] | None = None
    best_time_hits = len(questions) + 1
    search_rng = random.Random(f"{seed}:time-decorrelation")
    for attempt in range(20000):
        trial: dict[tuple[str, str], list[int]] = {}
        time_counts: defaultdict[int, Counter[int]] = defaultdict(Counter)
        for group in sorted(groups):
            positions = list(allocations[group])
            if attempt:
                search_rng.shuffle(positions)
            trial[group] = positions
            for question, answer_index in zip(groups[group], positions):
                time_bin = int(float(question["query_time"]) // 10)
                time_counts[time_bin][answer_index] += 1
        time_hits = sum(max(counter.values()) for counter in time_counts.values())
        if time_hits < best_time_hits:
            best_time_hits = time_hits
            best_positions = trial
        if time_hits / len(questions) <= 0.28:
            break
    if best_positions is None or best_time_hits / len(questions) > 0.30:
        raise ExportError(
            f"could not decorrelate answer positions from query-time bins: {best_time_hits}/{len(questions)}"
        )

    summary: dict[str, dict[str, int]] = {}
    for group, rows in groups.items():
        positions = best_positions[group]
        for question, answer_index in zip(rows, positions):
            correct = question.pop("_correct")
            distractors = question.pop("_distractors")
            random.Random(f"{seed}:distractors:{question['episode_id']}:{len(question['question'])}:{answer_index}").shuffle(distractors)
            options = list(distractors)
            options.insert(answer_index, correct)
            question["options"] = options
            question["answer_index"] = answer_index
        summary[f"{group[0]}:{group[1]}"] = {
            str(index): positions.count(index) for index in range(4)
        }
    return dict(sorted(summary.items()))


def _majority_index_accuracy(questions: list[dict[str, Any]], key) -> float:
    groups: defaultdict[str, Counter[int]] = defaultdict(Counter)
    for question in questions:
        groups[str(key(question))][int(question["answer_index"])] += 1
    hits = sum(max(counter.values()) for counter in groups.values())
    return hits / len(questions)


def _audit_questions(questions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(questions) != 300:
        raise ExportError(f"expected 300 questions, got {len(questions)}")
    evidence_hits = []
    forbidden_hits = []
    for question in questions:
        matches = [index for index, option in enumerate(question["options"]) if option in EVIDENCE_OPTIONS]
        if len(matches) != 1:
            raise ExportError(f"question does not contain exactly one common evidence option: {question['id']}")
        evidence_hits.append(matches[0] == question["answer_index"])
        public_prompt = (question["question"] + " " + " ".join(question["options"])).lower()
        for cue in FORBIDDEN_PROMPT_CUES:
            if cue in public_prompt:
                forbidden_hits.append({"question_id": question["id"], "cue": cue})
    status_counts = Counter(question["diagnostics"]["epistemic_status"] for question in questions)
    if sum(evidence_hits) != status_counts["uncertain"]:
        raise ExportError("common evidence option is not correct exactly on the uncertainty split")
    old_current = sum(
        all(any(marker in option for option in question["options"]) for marker in (
            "Visible at a different position", "Visible in the agent's hand", "Visibly broken into pieces",
        ))
        for question in questions
    )
    history_enumeration = sum(
        bool(re.search(r"involving .+, how many have completed", question["question"], re.I))
        for question in questions
    )
    temporal_mention_order = sum(
        bool(re.search(r"orders the .+ and .+ events", question["question"], re.I))
        for question in questions
    )
    type_status_prior = _majority_index_accuracy(
        questions,
        lambda question: f"{question['question_type']}:{question['diagnostics']['epistemic_status']}",
    )
    subtype_prior = _majority_index_accuracy(questions, lambda question: question["question_subtype"])
    time_prior = _majority_index_accuracy(questions, lambda question: int(float(question["query_time"]) // 10))
    evidence_rule_accuracy = sum(evidence_hits) / len(questions)
    maximum_rule_accuracy = max(type_status_prior, subtype_prior, time_prior, evidence_rule_accuracy)
    audit = {
        "forbidden_prompt_cue_hits": forbidden_hits,
        "old_current_template_hits": old_current,
        "history_enumeration_template_hits": history_enumeration,
        "temporal_mention_order_template_hits": temporal_mention_order,
        "evidence_option_correct_count": sum(evidence_hits),
        "evidence_option_present_count": len(evidence_hits),
        "evidence_only_rule_accuracy": round(evidence_rule_accuracy, 6),
        "type_status_position_prior_accuracy": round(type_status_prior, 6),
        "subtype_position_prior_accuracy": round(subtype_prior, 6),
        "query_time_bin_position_prior_accuracy": round(time_prior, 6),
        "maximum_static_rule_accuracy": round(maximum_rule_accuracy, 6),
        "static_release_threshold": 0.30,
        "static_release_passed": (
            not forbidden_hits
            and old_current == 0
            and history_enumeration == 0
            and temporal_mention_order == 0
            and maximum_rule_accuracy <= 0.30
        ),
    }
    return audit


def _validate_document(document: dict[str, Any], json_path: Path) -> None:
    if set(document) != EPISODE_KEYS:
        raise ExportError(f"episode keys differ from reference schema: {json_path}")
    if document["episode_id"] != json_path.stem:
        raise ExportError("episode id and QA filename differ")
    video = Path(document["video_path"])
    if not video.is_file() or video.stem != document["episode_id"]:
        raise ExportError("paired output video is invalid")
    timestamps = {round(float(value), 6) for value in document["sample_timestamps"]}
    for question in document["questions"]:
        if set(question) != QUESTION_KEYS:
            raise ExportError(f"question keys differ from reference schema: {question.get('id')}")
        if question["episode_id"] != document["episode_id"]:
            raise ExportError("question episode id differs")
        if question["question_type"] not in QUESTION_TYPES:
            raise ExportError("invalid question type")
        if len(question["options"]) != 4 or len(set(question["options"])) != 4:
            raise ExportError("question options are invalid")
        if question["answer_index"] not in range(4):
            raise ExportError("answer index is invalid")
        query = round(float(question["query_time"]), 6)
        if query not in timestamps:
            raise ExportError("query time is not a sampled timestamp")
        for span in question["evidence_spans"]:
            if len(span) != 2 or span[0] > span[1] or span[1] > question["query_time"]:
                raise ExportError("evidence span ordering is invalid")
            if round(float(span[0]), 6) not in timestamps or round(float(span[1]), 6) not in timestamps:
                raise ExportError("evidence span endpoint is not sampled")
        status = question["diagnostics"]["epistemic_status"]
        sources = set(question["diagnostics"]["uncertainty_sources"])
        if sources & FORBIDDEN_UNCERTAINTY_SOURCES or not sources <= ALLOWED_UNCERTAINTY_SOURCES:
            raise ExportError("invalid uncertainty source")
        correct = question["options"][question["answer_index"]]
        if status == "known" and (sources or correct in EVIDENCE_OPTIONS):
            raise ExportError("known question has uncertain diagnostics or answer")
        if status == "uncertain" and (not sources or correct not in EVIDENCE_OPTIONS):
            raise ExportError("uncertain question does not select the common evidence option")


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    type_counts = summary["question_type_counts"]
    status_counts = summary["epistemic_status_counts"]
    audit = summary["text_leakage_audit"]
    lines = [
        "# Sim QA v2 修改与去泄漏审计报告",
        "",
        "## 1. 版本说明",
        "",
        f"- 父数据集：`{source}`",
        f"- 当前数据集：`{output}`",
        "- 视频未重新渲染，28 个 MP4 与父数据集使用 hardlink；QA 根据原始仿真事件真值重新生成。",
        "- 245 条 Known 用作主视频理解评测，55 条 Uncertain 单独作为 epistemic 子集。",
        "",
        "## 2. 场景与数据分布",
        "",
        "| 场景 | 视频 | QA | 占比 |",
        "|---|---:|---:|---:|",
        "| 厨房 | 7 | 75 | 25% |",
        "| 客厅 | 7 | 75 | 25% |",
        "| 卧室 | 7 | 75 | 25% |",
        "| 卫生间 | 7 | 75 | 25% |",
        "| 合计 | 28 | 300 | 100% |",
        "",
        "| 问题类型 | 数量 |",
        "|---|---:|",
    ]
    for question_type in QUESTION_TYPES:
        lines.append(f"| `{question_type}` | {type_counts[question_type]} |")
    lines.extend([
        "",
        f"Known：{status_counts['known']}；Uncertain：{status_counts['uncertain']}；总计：{summary['question_count']}。",
        "四类场景各 7 个视频、75 条 QA；ABCD 正确答案位置各 75 条。",
        "",
        "## 3. 主要修改",
        "",
        "- current_state 改为双物体联合可见状态，不再使用换位、手持、碎裂三个固定死干扰项。",
        "- history_aggregation 不再在题干中枚举恰好等于答案数量的物体列表。",
        "- temporal_reasoning 改为 anchor event 的直接前驱，不再按真实先后顺序书写题干对象。",
        "- 所有问题都包含一个中性证据不足选项；该选项仅在 Uncertain 子集中为正确答案。",
        "- 正确答案位置按 question_type × epistemic_status 分层均衡。",
        "- video_evidence 不再复述正确答案，评测 prompt 必须采用字段白名单。",
        "",
        "## 4. v1 → v2 直接规则对比",
        "",
        "| 规则 | v1 可直接命中 | v2 模板命中 |",
        "|---|---:|---:|",
        f"| current-state 固定死选项 | 53 | {audit['old_current_template_hits']} |",
        f"| history 题干枚举即答案 | 39 | {audit['history_enumeration_template_hits']} |",
        f"| temporal 题干提及顺序即时间顺序 | 27 | {audit['temporal_mention_order_template_hits']} |",
        "| uncertain 独占元语言选项 | 55 | 0（证据选项在全部300题出现） |",
        "| 四类规则去重覆盖 | 174/300 | 0 个旧模板命中 |",
        "",
        "## 5. 静态泄漏审计",
        "",
        f"- 旧 current-state 模板命中：{audit['old_current_template_hits']}。",
        f"- history 题干直接枚举计数命中：{audit['history_enumeration_template_hits']}。",
        f"- temporal 题干提及顺序规则命中：{audit['temporal_mention_order_template_hits']}。",
        f"- evidence-only 规则准确率：{audit['evidence_only_rule_accuracy']:.2%}。",
        f"- type/status 位置先验：{audit['type_status_position_prior_accuracy']:.2%}。",
        f"- subtype 位置先验：{audit['subtype_position_prior_accuracy']:.2%}。",
        f"- query-time 分桶位置先验：{audit['query_time_bin_position_prior_accuracy']:.2%}。",
        f"- 静态规则最大准确率：{audit['maximum_static_rule_accuracy']:.2%}；门槛 30%。",
        f"- 静态发布门禁：{'通过' if audit['static_release_passed'] else '未通过'}。",
        "",
        "## 6. 评测输入隔离",
        "",
        "- `model_inputs.jsonl` 只含 question ID、episode ID、视频路径、query time 和由 question/options 构造的 prompt。",
        "- `evaluation_labels.json` 单独保存 answer index、题型和 split；模型输入文件不含标签、diagnostics 或 video_evidence。",
        "- `evaluation_splits.json` 给出245条主 Known 与55条 Uncertain 的 question ID。",
        "",
        "## 7. 已知边界与模型发布门槛",
        "",
        "本版本复用原视频，无法为镜头外动作机制和物理实例连续性构造严格的可见 Known 最小对，因此 Uncertain 不计入主视频准确率。正式发布前仍需完成 Qwen3.5-9B 对照评测：Known text-only ≤45%，正确视频相对 text-only 提升至少 10 个百分点，且 shuffled/mismatched video 回落至 text-only 附近。未达到模型门槛时，本目录只标记为 candidate。",
        "",
    ])
    return "\n".join(lines)


def validate_export(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    qfiles = sorted((output / "meta_data" / "qa_results").glob("*/*.json"))
    videos = sorted((output / "meta_data" / "gene_videos").glob("*/*.mp4"))
    if len(qfiles) != 28 or len(videos) != 28:
        raise ExportError(f"expected 28 QA files and videos, got {len(qfiles)} and {len(videos)}")
    questions: list[dict[str, Any]] = []
    episode_ids = set()
    for path in qfiles:
        document = _read_json(path)
        _validate_document(document, path)
        episode_ids.add(document["episode_id"])
        questions.extend(document["questions"])
    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ExportError("question ids are not globally unique")
    jsonl = [json.loads(line) for line in (output / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if jsonl != questions:
        raise ExportError("questions.jsonl differs from per-episode QA")
    model_input_path = output / "model_inputs.jsonl"
    label_path = output / "evaluation_labels.json"
    if model_input_path.exists() != label_path.exists():
        raise ExportError("model inputs and evaluation labels must be present together")
    if model_input_path.exists():
        model_inputs = [
            json.loads(line) for line in model_input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        labels = _read_json(label_path)
        if len(model_inputs) != len(questions) or len(labels) != len(questions):
            raise ExportError("evaluation bundle does not cover every question")
        question_by_id = {question["id"]: question for question in questions}
        for record in model_inputs:
            if set(record) != MODEL_INPUT_KEYS:
                raise ExportError("model input record violates the prompt allowlist")
            question = question_by_id.get(record["question_id"])
            if question is None or record["prompt"] != build_model_prompt(question):
                raise ExportError("model input prompt differs from the allowlisted prompt builder")
            public_record = json.dumps(record, ensure_ascii=False)
            if any(key in public_record for key in ("answer_index", "diagnostics", "video_evidence", "diagnostic_rationale")):
                raise ExportError("private QA metadata leaked into a model input record")
    type_counts = Counter(question["question_type"] for question in questions)
    status_counts = Counter(question["diagnostics"]["epistemic_status"] for question in questions)
    answer_counts = Counter(question["answer_index"] for question in questions)
    source_counts = Counter(
        source
        for question in questions
        for source in question["diagnostics"]["uncertainty_sources"]
    )
    type_status_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for question in questions:
        type_status_counts[question["question_type"]][question["diagnostics"]["epistemic_status"]] += 1
    expected_types = {"current_state": 54, "factual_retrieval": 54, "history_aggregation": 51, "state_change": 51, "object_tracking": 50, "temporal_reasoning": 40}
    if dict(type_counts) != expected_types:
        raise ExportError(f"question type counts changed: {dict(type_counts)}")
    if dict(status_counts) != {"known": 245, "uncertain": 55}:
        raise ExportError(f"epistemic distribution changed: {dict(status_counts)}")
    if dict(answer_counts) != {0: 75, 1: 75, 2: 75, 3: 75}:
        raise ExportError(f"answer positions changed: {dict(answer_counts)}")
    audit = _audit_questions(questions)
    if not audit["static_release_passed"]:
        raise ExportError(f"static leakage release gate failed: {audit}")
    return {
        "version": VERSION,
        "candidate_status": "static_audit_passed_model_evaluation_pending",
        "episode_count": len(episode_ids),
        "video_count": len(videos),
        "question_count": len(questions),
        "question_type_counts": dict(type_counts),
        "epistemic_status_counts": dict(status_counts),
        "question_type_epistemic_counts": {
            key: dict(value) for key, value in type_status_counts.items()
        },
        "uncertainty_source_counts": dict(source_counts),
        "answer_index_counts": {str(index): answer_counts[index] for index in range(4)},
        "text_leakage_audit": audit,
    }


def export_clone(
    source_dataset: str | Path,
    output_dir: str | Path,
    *,
    link_mode: str = "hardlink",
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    source = Path(source_dataset).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source dataset does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if link_mode not in {"hardlink", "copy"}:
        raise ValueError(f"unsupported link mode: {link_mode}")
    source_fingerprint = _source_fingerprint(source)
    source_qfiles = sorted((source / "meta_data" / "qa_results").glob("*/*.json"))
    source_videos = sorted((source / "meta_data" / "gene_videos").glob("*/*.mp4"))
    if len(source_qfiles) != 28 or len(source_videos) != 28:
        raise ExportError("source dataset must contain 28 QA JSON files and 28 videos")

    # Clone first. QA is copied as an independent inode; videos may be hard-linked.
    output.mkdir(parents=True, exist_ok=True)
    for path in source_qfiles:
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if os.stat(path).st_ino == os.stat(target).st_ino:
            raise ExportError("QA clone unexpectedly shares an inode with the source")
    video_manifest = []
    for path in source_videos:
        target = output / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "hardlink":
            os.link(path, target)
        else:
            shutil.copy2(path, target)
        video_manifest.append({
            "relative_path": str(path.relative_to(source)),
            "source_inode": os.stat(path).st_ino,
            "output_inode": os.stat(target).st_ino,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    for name in ("questions.jsonl", "summary.json", "four_scene_summary.json"):
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)
    source_report = source / "QA数据分布报告_中文.md"
    if source_report.is_file():
        shutil.copy2(source_report, output / "父版本_QA数据分布报告_中文.md")

    source_summary = _read_json(source / "summary.json")
    raw_episode_dirs = [Path(path) for path in source_summary.get("raw_episode_dirs", [])]
    if not raw_episode_dirs:
        four = _read_json(source / "four_scene_summary.json")
        raw_episode_dirs = [Path(path) for path in four["reference_aligned_summary"]["raw_episode_dirs"]]
    raw_by_episode: dict[str, tuple[str, list[dict[str, Any]], int]] = {}
    for raw_dir in raw_episode_dirs:
        _, plan, trace, profile = v1._load_raw_episode(raw_dir)
        episode_id = _episode_id(plan).removesuffix("_visit01")
        duration = round(len(plan["trajectory"]["frames"]) / int(plan["trajectory"]["fps"]), 2)
        maximum = int(math.floor(duration + 1e-6))
        raw_by_episode[episode_id] = (str(plan["recipe"]["scene"]), build_events(plan, trace, profile), maximum)
    if len(raw_by_episode) != 28:
        raise ExportError(f"expected 28 raw episode sources, got {len(raw_by_episode)}")

    ordinals: defaultdict[tuple[str, str], int] = defaultdict(int)
    records: list[tuple[Path, dict[str, Any]]] = []
    all_questions: list[dict[str, Any]] = []
    semantic_counts: Counter[str] = Counter()
    global_ordinal = 0
    for source_path in source_qfiles:
        source_document = _read_json(source_path)
        episode_id = str(source_document["episode_id"])
        if episode_id not in raw_by_episode:
            raise ExportError(f"raw episode truth missing for {episode_id}")
        scene, events, maximum = raw_by_episode[episode_id]
        questions = []
        for source_question in source_document["questions"]:
            question_type = str(source_question["question_type"])
            status = str(source_question["diagnostics"]["epistemic_status"])
            key = (question_type, status)
            question = _build_question(
                episode_id=episode_id,
                scene=scene,
                events=events,
                source_question=source_question,
                ordinal=ordinals[key],
                maximum=maximum,
                seed=seed + global_ordinal,
            )
            ordinals[key] += 1
            global_ordinal += 1
            semantic_counts[question["_semantic_class"]] += 1
            questions.append(question)
            all_questions.append(question)
        target_path = output / source_path.relative_to(source)
        target_video = output / "meta_data" / "gene_videos" / scene / f"{episode_id}.mp4"
        document = dict(source_document)
        document.update({
            "video_path": str(target_video.resolve()),
            "qa_source": "simulation_ground_truth_aligned_to_rendered_video_text_debiased_v2",
            "questions": questions,
        })
        records.append((target_path, document))

    stratified_positions = _assign_balanced_options(all_questions, seed)
    for target_path, document in records:
        for index, question in enumerate(document["questions"], start=1):
            question["id"] = f"{document['episode_id']}_v2_q{index:02d}"
            question.pop("_evidence_option")
            question.pop("_semantic_class")
        _write_json(target_path, document)
        _validate_document(document, target_path)
    _write_jsonl(output / "questions.jsonl", all_questions)

    model_inputs = []
    evaluation_labels = {}
    for target_path, document in records:
        del target_path
        for question in document["questions"]:
            status = question["diagnostics"]["epistemic_status"]
            model_inputs.append({
                "question_id": question["id"],
                "episode_id": question["episode_id"],
                "video_path": document["video_path"],
                "query_time": question["query_time"],
                "prompt": build_model_prompt(question),
            })
            evaluation_labels[question["id"]] = {
                "answer_index": question["answer_index"],
                "question_type": question["question_type"],
                "epistemic_status": status,
                "split": "primary_visual_known" if status == "known" else "epistemic_uncertain",
            }
    _write_jsonl(output / "model_inputs.jsonl", model_inputs)
    _write_json(output / "evaluation_labels.json", evaluation_labels)

    preliminary = validate_export(output)
    split_manifest = {
        "primary_visual_known": [question["id"] for question in all_questions if question["diagnostics"]["epistemic_status"] == "known"],
        "epistemic_uncertain": [question["id"] for question in all_questions if question["diagnostics"]["epistemic_status"] == "uncertain"],
        "headline_metric": "answer_accuracy_on_primary_visual_known",
        "compatibility_metric": "answer_accuracy_on_all_questions_not_for_headline_use",
    }
    _write_json(output / "evaluation_splits.json", split_manifest)
    revision_manifest = {
        "version": VERSION,
        "parent_dataset": str(source),
        "output_dataset": str(output),
        "seed": seed,
        "link_mode": link_mode,
        "source_fingerprint": source_fingerprint,
        "video_manifest": video_manifest,
        "stratified_answer_positions": stratified_positions,
        "semantic_ground_truth_counts": dict(sorted(semantic_counts.items())),
        "model_release_gates": {
            "known_text_only_max": 0.45,
            "minimum_video_gain": 0.10,
            "status": "pending",
        },
    }
    _write_json(output / "qa_revision_manifest.json", revision_manifest)
    summary = dict(preliminary)
    summary.update({
        "parent_dataset": str(source),
        "dataset_dir": str(output),
        "qa_results_dir": str((output / "meta_data" / "qa_results").resolve()),
        "gene_videos_dir": str((output / "meta_data" / "gene_videos").resolve()),
        "questions_jsonl": str((output / "questions.jsonl").resolve()),
        "evaluation_splits": str((output / "evaluation_splits.json").resolve()),
        "model_inputs": str((output / "model_inputs.jsonl").resolve()),
        "evaluation_labels": str((output / "evaluation_labels.json").resolve()),
        "semantic_ground_truth_counts": dict(sorted(semantic_counts.items())),
        "stratified_answer_positions": stratified_positions,
    })
    _write_json(output / "summary.json", summary)
    four = _read_json(source / "four_scene_summary.json")
    four["parent_dataset"] = str(source)
    four["dataset_dir"] = str(output)
    four["reference_aligned_summary"] = summary
    _write_json(output / "four_scene_summary.json", four)
    report = _report_markdown(summary, source, output)
    report_path = output / "QA_v2修改与去泄漏审计报告_中文.md"
    report_path.write_text(report, encoding="utf-8")

    if _source_fingerprint(source) != source_fingerprint:
        raise ExportError("source dataset changed during v2 generation")
    for item in video_manifest:
        if link_mode == "hardlink" and item["source_inode"] != item["output_inode"]:
            raise ExportError(f"video is not hard-linked: {item['relative_path']}")
    final = validate_export(output)
    final.update({key: value for key, value in summary.items() if key not in final})
    return final


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        summary = validate_export(args.output)
    else:
        summary = export_clone(
            args.source_dataset,
            args.output,
            link_mode=args.link_mode,
            seed=args.seed,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
