"""Build v2.1 QA with separate known and uncertain question types.

Known questions use four concrete alternatives. Uncertain questions include
an insufficient-evidence answer. Evidence must end by query_time.
"""
from __future__ import annotations

import argparse
import itertools
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
from . import text_debiased_qa as v2
from .event_adapter import build_events
from .exporter import _episode_id


VERSION = "visual_grounded_v2_1"
ID_VERSION_TAG = "v21"
QA_SOURCE = "simulation_ground_truth_visual_grounded_v2_1"
REPORT_FILENAME = "QA_v2.1视觉必要性修复与分布报告_中文.md"
DEFAULT_SEED = 20260824
QUESTION_TYPES = v1.QUESTION_TYPES
EPISODE_KEYS = v1.EPISODE_KEYS
QUESTION_KEYS = v1.QUESTION_KEYS
ALLOWED_UNCERTAINTY_SOURCES = v1.ALLOWED_UNCERTAINTY_SOURCES
FORBIDDEN_UNCERTAINTY_SOURCES = v1.FORBIDDEN_UNCERTAINTY_SOURCES
REFUSAL_OPTIONS = v2.EVIDENCE_OPTIONS
MODEL_INPUT_KEYS = v2.MODEL_INPUT_KEYS

SUBTYPES = {
    "factual_retrieval": "visible_object_transition_v21",
    "current_state": "unique_object_visibility_at_query_v21",
    "history_aggregation": "bounded_visible_object_set_v21",
    "state_change": "visible_state_transition_v21",
    "object_tracking": "visible_object_return_v21",
    "temporal_reasoning": "local_visible_event_order_v21",
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

QUESTION_META_CUES = (
    "supported",
    "uniquely",
    "evidence",
    "recorded observations",
    "indexed observations",
)

ROOM_DESCRIPTIONS = {
    "kitchen": "厨房：操作台、炉灶和餐桌附近的食材、餐具取放与复位。",
    "living_room": "客厅：茶几、边桌、沙发和电视柜附近的日常物品移动。",
    "bedroom": "卧室：床、书桌、床头柜、置物架和梳妆台附近的物品变化。",
    "bathroom": "卫生间：洗手池、浴缸、毛巾架、置物架和地面附近的物品变化。",
}


class ExportError(ValueError):
    """Raised when a v2.1 clone or QA contract is invalid."""


def _read_json(path: Path) -> Any:
    return v2._read_json(path)


def _write_json(path: Path, value: Any) -> None:
    v2._write_json(path, value)


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    v2._write_jsonl(path, values)


def _sha256(path: Path) -> str:
    return v2._sha256(path)


def _source_fingerprint(source: Path) -> dict[str, str]:
    return v2._source_fingerprint(source)


def _room_for_scene(scene: str) -> str:
    return v2._room_for_scene(scene)


def _event_points(event: dict[str, Any], maximum: int) -> tuple[float, float]:
    return v2._event_points(event, maximum)


def _label(event: dict[str, Any]) -> str:
    return v2._label(event)


def _surface(event: dict[str, Any]) -> str:
    return v2._surface(event)


def _ordered(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return v2._ordered(events)


def _event_pairs(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return v2._event_pairs(events)


def _visible_at(events: list[dict[str, Any]], target_id: str, query_time: float, maximum: int) -> bool:
    return v2._visible_at(events, target_id, query_time, maximum)


def _dedupe_spans(spans: Iterable[list[float]]) -> list[list[float]]:
    return v2._dedupe_spans(spans)


def _event_description(event: dict[str, Any]) -> str:
    action = "disappears from" if event["event_type"] == "disappearance" else "reappears at"
    return f"The {_label(event)} {action} its original {_surface(event)} position"


def _refusal_option(ordinal: int, seed: int) -> str:
    return REFUSAL_OPTIONS[(ordinal * 5 + seed) % len(REFUSAL_OPTIONS)]


def build_model_prompt(question: dict[str, Any]) -> str:
    return v2.build_model_prompt(question)


def _draft(
    *,
    episode_id: str,
    question_type: str,
    query_time: float,
    question: str,
    concrete_options: list[str],
    correct: str,
    refusal_option: str,
    evidence_spans: list[list[float]],
    uncertainty_sources: list[str],
    events: list[dict[str, Any]],
    semantic_class: str,
) -> dict[str, Any]:
    uncertain = bool(uncertainty_sources)
    expected = 3 if uncertain else 4
    if len(concrete_options) != expected or len(set(concrete_options)) != expected:
        raise ExportError(f"question needs {expected} distinct concrete options")
    if uncertain:
        if correct != refusal_option:
            raise ExportError("uncertain question must select its refusal option")
        distractors = list(concrete_options)
    else:
        if correct not in concrete_options:
            raise ExportError("known correct answer must be a concrete option")
        if any(option in REFUSAL_OPTIONS for option in concrete_options):
            raise ExportError("known questions may not contain a refusal option")
        distractors = [option for option in concrete_options if option != correct]
    spans = _dedupe_spans(evidence_spans)
    if not spans or any(end > query_time + 1e-6 for _, end in spans):
        raise ExportError("evidence must be non-empty and end by query_time")
    return {
        "id": "",
        "episode_id": episode_id,
        "query_time": float(query_time),
        "question_type": question_type,
        "question_subtype": SUBTYPES[question_type],
        "video_evidence": "Use sampled video observations no later than query_time.",
        "question": question,
        "options": [],
        "answer_index": -1,
        "evidence_spans": spans,
        "diagnostics": {
            "epistemic_status": "uncertain" if uncertain else "known",
            "uncertainty_sources": sorted(uncertainty_sources),
        },
        "diagnostic_rationale": {
            "volatility": "The answer changes when the indexed visual event or state changes.",
            "uncertainty": (
                "The visible prefix does not establish one concrete alternative."
                if uncertain
                else "One concrete alternative is visible in the indexed prefix."
            ),
        },
        "change_intensity": max(1, sum(float(event["start_sec"]) <= query_time + 1e-6 for event in events)),
        "_correct": correct,
        "_distractors": distractors,
        "_refusal_option": refusal_option,
        "_semantic_class": semantic_class,
    }


def _other_labels(events: list[dict[str, Any]], correct: str, ordinal: int, count: int = 3) -> list[str]:
    labels = list(dict.fromkeys(_label(event) for event in _ordered(events)))
    alternatives = [label for label in labels if label != correct]
    if len(alternatives) < count:
        raise ExportError(f"episode needs at least {count + 1} distinct object labels")
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    return rotated[:count]


def _gap_query(pair: tuple[dict[str, Any], dict[str, Any]], maximum: int) -> tuple[float, float, float]:
    disappearance, appearance = pair
    dis_before, dis_after = _event_points(disappearance, maximum)
    app_before, _ = _event_points(appearance, maximum)
    query = float(max(dis_after, min(app_before, math.floor((dis_after + app_before) / 2))))
    return dis_before, dis_after, query


def _build_current_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str,
) -> dict[str, Any]:
    del scene
    preferred_visible = bool(ordinal % 2)
    labels: dict[str, str] = {}
    representative_by_label: dict[str, str] = {}
    for event in _ordered(events):
        target_id = str(event["_target_id"])
        label = _label(event)
        labels[target_id] = label
        representative_by_label.setdefault(label, target_id)
    candidates_by_state: dict[bool, list[tuple[float, float, str, list[str]]]] = {False: [], True: []}
    for desired_visible in (False, True):
        expected_event_type = "appearance" if desired_visible else "disappearance"
        for event in _ordered(events):
            target_id = str(event["_target_id"])
            label = labels[target_id]
            if event["event_type"] != expected_event_type or representative_by_label[label] != target_id:
                continue
            before, query = _event_points(event, maximum)
            if _visible_at(events, target_id, query, maximum) != desired_visible:
                continue
            alternatives = []
            for other_label, other_id in representative_by_label.items():
                if other_label == label:
                    continue
                if _visible_at(events, other_id, query, maximum) != desired_visible:
                    alternatives.append(other_label)
            if len(alternatives) >= 3:
                candidates_by_state[desired_visible].append((before, query, label, alternatives))
    desired_visible = preferred_visible if candidates_by_state[preferred_visible] else not preferred_visible
    candidates = candidates_by_state[desired_visible]
    if not candidates:
        state = "visible" if desired_visible else "not visible"
        raise ExportError(f"no current-state candidate with one uniquely {state} listed object")
    before, query, correct, alternatives = candidates[(ordinal // 2) % len(candidates)]
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    state = "visible" if desired_visible else "not visible"
    return _draft(
        episode_id=episode_id,
        question_type="current_state",
        query_time=query,
        question=f"At {query:.1f}s, which listed object is {state} at its original position?",
        concrete_options=[correct, rotated[0], rotated[1], rotated[2]],
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[before, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"unique_object_{'visible' if desired_visible else 'absent'}",
    )


def _build_factual_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str,
) -> dict[str, Any]:
    del scene
    event = _ordered(events)[(ordinal * 3) % len(events)]
    before, after = _event_points(event, maximum)
    correct = _label(event)
    action = "changes from visible to absent" if event["event_type"] == "disappearance" else "changes from absent to visible"
    concrete = [correct, *_other_labels(events, correct, ordinal)]
    return _draft(
        episode_id=episode_id,
        question_type="factual_retrieval",
        query_time=after,
        question=f"Which object {action} between {before:.1f}s and {after:.1f}s?",
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[before, after]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"object:{correct}",
    )


def _build_location_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], question_type: str,
    ordinal: int, maximum: int, refusal_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    disappearance, appearance = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    del appearance
    dis_before, dis_after, query = _gap_query(_event_pairs(events)[ordinal % len(_event_pairs(events))], maximum)
    label = _label(disappearance)
    surface = _surface(disappearance)
    concrete = [
        f"At its original {surface} position",
        f"At another position on the {surface}",
        "At a different visible support in the room",
    ]
    return _draft(
        episode_id=episode_id,
        question_type=question_type,
        query_time=query,
        question=f"Where is the {label} at {query:.1f}s?",
        concrete_options=concrete,
        correct=refusal_option,
        refusal_option=refusal_option,
        evidence_spans=[[dis_before, query]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="location_not_visible_in_prefix",
    )


def _build_history_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str,
) -> dict[str, Any]:
    del scene
    first_disappearance_by_label: dict[str, dict[str, Any]] = {}
    for event in _ordered(events):
        if event["event_type"] == "disappearance":
            first_disappearance_by_label.setdefault(_label(event), event)
    disappearances = _ordered(list(first_disappearance_by_label.values()))
    count = 2 + ordinal % 2
    maximum_start = len(disappearances) - count
    start_index = (ordinal // 2) % (maximum_start + 1)
    selected = disappearances[start_index:start_index + count]
    start = _event_points(selected[0], maximum)[0]
    query = _event_points(selected[-1], maximum)[1]
    labels = sorted(dict.fromkeys(_label(event) for event in disappearances))
    if len(labels) < 4:
        raise ExportError("history aggregation needs at least four distinct object labels")
    correct_labels = tuple(sorted(_label(event) for event in selected))
    combinations = list(itertools.combinations(labels, count))
    if correct_labels not in combinations:
        raise ExportError("history aggregation ground-truth set is malformed")
    alternatives = [candidate for candidate in combinations if candidate != correct_labels]
    if len(alternatives) < 3:
        raise ExportError("history aggregation needs three same-cardinality set distractors")
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    selected_combinations = [correct_labels, rotated[0], rotated[1], rotated[2]]
    concrete = [" and ".join(candidate) for candidate in selected_combinations]
    correct = " and ".join(correct_labels)
    return _draft(
        episode_id=episode_id,
        question_type="history_aggregation",
        query_time=query,
        question=f"Which set of objects changes from visible to absent between {start:.1f}s and {query:.1f}s?",
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"bounded_disappearance_set_size_{count}",
    )


def _build_history_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    pair = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    disappearance, _ = pair
    dis_before, dis_after, query = _gap_query(pair, maximum)
    label = _label(disappearance)
    concrete = ["No separate manipulation", "Exactly one separate manipulation", "Two or more separate manipulations"]
    return _draft(
        episode_id=episode_id,
        question_type="history_aggregation",
        query_time=query,
        question=f"How many separate manipulations of the {label} occur after {dis_after:.1f}s and by {query:.1f}s?",
        concrete_options=concrete,
        correct=refusal_option,
        refusal_option=refusal_option,
        evidence_spans=[[dis_before, query]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="hidden_manipulation_count",
    )


def _four_transition_options(events: list[dict[str, Any]], event: dict[str, Any], ordinal: int) -> list[str]:
    correct = _event_description(event)
    reverse_action = "reappears at" if event["event_type"] == "disappearance" else "disappears from"
    reverse = f"The {_label(event)} {reverse_action} its original {_surface(event)} position"
    alternatives = list(dict.fromkeys(_event_description(candidate) for candidate in _ordered(events)))
    alternatives = [option for option in alternatives if option not in {correct, reverse}]
    if len(alternatives) < 2:
        raise ExportError("episode needs two additional transition distractors")
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    return [correct, reverse, rotated[0], rotated[1]]


def _build_state_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str,
) -> dict[str, Any]:
    del scene
    ordered = _ordered(events)
    event = ordered[(ordinal * 3 + 1) % len(ordered)]
    before, after = _event_points(event, maximum)
    correct = _event_description(event)
    return _draft(
        episode_id=episode_id,
        question_type="state_change",
        query_time=after,
        question=f"Which visible transition occurs between {before:.1f}s and {after:.1f}s?",
        concrete_options=_four_transition_options(events, event, ordinal),
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[before, after]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"transition:{event['event_type']}",
    )


def _build_state_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene, ordinal
    event = next(item for item in _ordered(events) if item["event_type"] == "disappearance")
    before, after = _event_points(event, maximum)
    label = _label(event)
    concrete = [
        f"The agent carries the {label} away",
        f"The {label} slides away on its support",
        f"Another object moves the {label}",
    ]
    return _draft(
        episode_id=episode_id,
        question_type="state_change",
        query_time=after,
        question=f"What causes the {label} to leave its original position between {before:.1f}s and {after:.1f}s?",
        concrete_options=concrete,
        correct=refusal_option,
        refusal_option=refusal_option,
        evidence_spans=[[before, after]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="transition_mechanism_not_visible",
    )


def _build_object_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str,
) -> dict[str, Any]:
    del scene
    disappearance, appearance = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    dis_before, _ = _event_points(disappearance, maximum)
    app_before, app_after = _event_points(appearance, maximum)
    correct = _label(disappearance)
    concrete = [correct, *_other_labels(events, correct, ordinal)]
    return _draft(
        episode_id=episode_id,
        question_type="object_tracking",
        query_time=app_after,
        question=(
            f"Which object changes from absent to visible between {app_before:.1f}s and {app_after:.1f}s, "
            "after disappearing earlier?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[dis_before, app_after]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"tracked_object:{correct}",
    )


def _build_object_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    disappearance, appearance = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    dis_before, dis_after = _event_points(disappearance, maximum)
    _, app_after = _event_points(appearance, maximum)
    label = _label(disappearance)
    concrete = [
        "It is the same physical instance",
        "It is a different physical instance",
        "The earlier instance was destroyed and replaced",
    ]
    return _draft(
        episode_id=episode_id,
        question_type="object_tracking",
        query_time=app_after,
        question=f"Is the {label} visible at {app_after:.1f}s the same physical instance seen before {dis_after:.1f}s?",
        concrete_options=concrete,
        correct=refusal_option,
        refusal_option=refusal_option,
        evidence_spans=[[dis_before, app_after]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="physical_identity_not_visible",
    )


def _build_temporal_known(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str,
) -> dict[str, Any]:
    del scene
    ordered = _ordered(events)
    candidates: list[tuple[int, str, list[str]]] = []
    for anchor_index in range(4, len(ordered)):
        previous = _event_description(ordered[anchor_index - 1])
        alternatives = []
        for candidate in reversed(ordered[:anchor_index - 1]):
            description = _event_description(candidate)
            if description != previous and description not in alternatives:
                alternatives.append(description)
        if len(alternatives) >= 3:
            candidates.append((anchor_index, previous, alternatives))
    if not candidates:
        raise ExportError("episode cannot form a four-way temporal predecessor question")
    anchor_index, correct, alternatives = candidates[ordinal % len(candidates)]
    anchor = ordered[anchor_index]
    previous = ordered[anchor_index - 1]
    offset = ordinal % len(alternatives)
    rotated = alternatives[offset:] + alternatives[:offset]
    query = _event_points(anchor, maximum)[1]
    start = _event_points(previous, maximum)[0]
    return _draft(
        episode_id=episode_id,
        question_type="temporal_reasoning",
        query_time=query,
        question=f"Which event is shown immediately before the {_event_description(anchor).removeprefix('The ')} by {query:.1f}s?",
        concrete_options=[correct, rotated[0], rotated[1], rotated[2]],
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"local_predecessor:{previous['event_type']}",
    )


def _build_temporal_uncertain(
    episode_id: str, scene: str, events: list[dict[str, Any]], ordinal: int,
    maximum: int, refusal_option: str, sources: list[str],
) -> dict[str, Any]:
    del scene
    pair = _event_pairs(events)[ordinal % len(_event_pairs(events))]
    disappearance, _ = pair
    dis_before, dis_after, query = _gap_query(pair, maximum)
    label = _label(disappearance)
    concrete = [
        f"The {label} is picked up before it is rotated",
        f"The {label} is rotated before it is moved",
        f"The {label} is moved before it is picked up",
    ]
    return _draft(
        episode_id=episode_id,
        question_type="temporal_reasoning",
        query_time=query,
        question=f"Which action order involving the {label} occurs after {dis_after:.1f}s and by {query:.1f}s?",
        concrete_options=concrete,
        correct=refusal_option,
        refusal_option=refusal_option,
        evidence_spans=[[dis_before, query]],
        uncertainty_sources=sources,
        events=events,
        semantic_class="hidden_action_order",
    )


def _build_question(
    *, episode_id: str, scene: str, events: list[dict[str, Any]],
    source_question: dict[str, Any], ordinal: int, maximum: int, seed: int,
) -> dict[str, Any]:
    question_type = str(source_question["question_type"])
    sources = list(source_question["diagnostics"]["uncertainty_sources"])
    uncertain = source_question["diagnostics"]["epistemic_status"] == "uncertain"
    refusal = _refusal_option(ordinal, seed)
    if uncertain != bool(sources):
        raise ExportError("source epistemic status and uncertainty sources disagree")
    if question_type == "current_state":
        return (
            _build_location_uncertain(episode_id, scene, events, question_type, ordinal, maximum, refusal, sources)
            if uncertain else _build_current_known(episode_id, scene, events, ordinal, maximum, refusal)
        )
    if question_type == "factual_retrieval":
        return (
            _build_location_uncertain(episode_id, scene, events, question_type, ordinal, maximum, refusal, sources)
            if uncertain else _build_factual_known(episode_id, scene, events, ordinal, maximum, refusal)
        )
    if question_type == "history_aggregation":
        return (
            _build_history_uncertain(episode_id, scene, events, ordinal, maximum, refusal, sources)
            if uncertain else _build_history_known(episode_id, scene, events, ordinal, maximum, refusal)
        )
    if question_type == "state_change":
        return (
            _build_state_uncertain(episode_id, scene, events, ordinal, maximum, refusal, sources)
            if uncertain else _build_state_known(episode_id, scene, events, ordinal, maximum, refusal)
        )
    if question_type == "object_tracking":
        return (
            _build_object_uncertain(episode_id, scene, events, ordinal, maximum, refusal, sources)
            if uncertain else _build_object_known(episode_id, scene, events, ordinal, maximum, refusal)
        )
    if question_type == "temporal_reasoning":
        return (
            _build_temporal_uncertain(episode_id, scene, events, ordinal, maximum, refusal, sources)
            if uncertain else _build_temporal_known(episode_id, scene, events, ordinal, maximum, refusal)
        )
    raise ExportError(f"unsupported question type: {question_type}")


def _assign_balanced_options(questions: list[dict[str, Any]], seed: int) -> dict[str, dict[str, int]]:
    return v2._assign_balanced_options(questions, seed)


def _majority_index_accuracy(questions: list[dict[str, Any]], key) -> float:
    return v2._majority_index_accuracy(questions, key)


def _mentioned_times(question: dict[str, Any]) -> list[float]:
    return [
        float(value)
        for value in re.findall(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)s", question["question"])
    ]


def _answer_text_prior_accuracy(questions: list[dict[str, Any]], key) -> float:
    groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for question in questions:
        correct = str(question["options"][question["answer_index"]])
        groups[str(key(question))][correct] += 1
    return sum(max(counter.values()) for counter in groups.values()) / len(questions)


def _audit_questions(questions: list[dict[str, Any]]) -> dict[str, Any]:
    if len(questions) != 300:
        raise ExportError(f"expected 300 questions, got {len(questions)}")
    forbidden_hits = []
    question_meta_hits = []
    future_reference_hits = []
    zero_length_spans = []
    known_refusal_hits = []
    uncertain_refusal_correct = 0
    for question in questions:
        status = question["diagnostics"]["epistemic_status"]
        refusal_indices = [index for index, option in enumerate(question["options"]) if option in REFUSAL_OPTIONS]
        if status == "known" and refusal_indices:
            known_refusal_hits.append(question["id"])
        if status == "uncertain" and refusal_indices == [question["answer_index"]]:
            uncertain_refusal_correct += 1
        public_prompt = (question["question"] + " " + " ".join(question["options"])).lower()
        for cue in FORBIDDEN_PROMPT_CUES:
            if cue in public_prompt:
                forbidden_hits.append({"question_id": question["id"], "cue": cue})
        lowered_question = question["question"].lower()
        for cue in QUESTION_META_CUES:
            if cue in lowered_question:
                question_meta_hits.append({"question_id": question["id"], "cue": cue})
        if any(value > float(question["query_time"]) + 1e-6 for value in _mentioned_times(question)):
            future_reference_hits.append(question["id"])
        for span in question["evidence_spans"]:
            if float(span[1]) <= float(span[0]):
                zero_length_spans.append({"question_id": question["id"], "span": span})
    status_counts = Counter(question["diagnostics"]["epistemic_status"] for question in questions)
    type_status_prior = _majority_index_accuracy(
        questions,
        lambda question: f"{question['question_type']}:{question['diagnostics']['epistemic_status']}",
    )
    subtype_prior = _majority_index_accuracy(questions, lambda question: question["question_subtype"])
    time_prior = _majority_index_accuracy(questions, lambda question: int(float(question["query_time"]) // 10))
    answer_text_priors = {}
    for question_type in QUESTION_TYPES:
        rows = [
            question for question in questions
            if question["question_type"] == question_type
            and question["diagnostics"]["epistemic_status"] == "known"
        ]
        query_prior = _answer_text_prior_accuracy(
            rows, lambda question: int(float(question["query_time"]) // 10),
        )
        interval_prior = _answer_text_prior_accuracy(
            rows,
            lambda question: (
                int((max(_mentioned_times(question)) - min(_mentioned_times(question))) // 5)
                if len(_mentioned_times(question)) >= 2 else -1
            ),
        )
        answer_text_priors[question_type] = {
            "query_time_bin": round(query_prior, 6),
            "public_interval_bin": round(interval_prior, 6),
        }
    maximum_answer_text_prior = max(
        value for priors in answer_text_priors.values() for value in priors.values()
    )
    maximum_rule_accuracy = max(type_status_prior, subtype_prior, time_prior, maximum_answer_text_prior)
    audit = {
        "forbidden_prompt_cue_hits": forbidden_hits,
        "question_meta_cue_hits": question_meta_hits,
        "future_time_reference_hits": future_reference_hits,
        "zero_length_evidence_span_hits": zero_length_spans,
        "known_refusal_option_count": len(known_refusal_hits),
        "known_refusal_option_question_ids": known_refusal_hits,
        "uncertain_refusal_correct_count": uncertain_refusal_correct,
        "uncertain_count": status_counts["uncertain"],
        "type_status_position_prior_accuracy": round(type_status_prior, 6),
        "subtype_position_prior_accuracy": round(subtype_prior, 6),
        "query_time_bin_position_prior_accuracy": round(time_prior, 6),
        "known_answer_text_priors_by_type": answer_text_priors,
        "maximum_known_answer_text_prior_accuracy": round(maximum_answer_text_prior, 6),
        "maximum_static_rule_accuracy": round(maximum_rule_accuracy, 6),
        "static_release_threshold": 0.30,
    }
    audit["static_release_passed"] = (
        not forbidden_hits
        and not question_meta_hits
        and not future_reference_hits
        and not zero_length_spans
        and not known_refusal_hits
        and uncertain_refusal_correct == status_counts["uncertain"]
        and maximum_rule_accuracy <= 0.30
    )
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
            raise ExportError("question episode id differs from document")
        if question["question_type"] not in QUESTION_TYPES:
            raise ExportError("question type is invalid")
        if len(question["options"]) != 4 or len(set(question["options"])) != 4:
            raise ExportError("question options are invalid")
        if question["answer_index"] not in range(4):
            raise ExportError("answer index is invalid")
        query = round(float(question["query_time"]), 6)
        if query not in timestamps:
            raise ExportError("query time is not a sampled timestamp")
        if any(value > query + 1e-6 for value in _mentioned_times(question)):
            raise ExportError("question text refers to a future timestamp")
        for span in question["evidence_spans"]:
            if len(span) != 2 or span[0] >= span[1] or span[1] > question["query_time"]:
                raise ExportError("evidence span must be non-zero and end by query_time")
            if round(float(span[0]), 6) not in timestamps or round(float(span[1]), 6) not in timestamps:
                raise ExportError("evidence span endpoint is not sampled")
        status = question["diagnostics"]["epistemic_status"]
        sources = set(question["diagnostics"]["uncertainty_sources"])
        if sources & FORBIDDEN_UNCERTAINTY_SOURCES or not sources <= ALLOWED_UNCERTAINTY_SOURCES:
            raise ExportError("invalid uncertainty source")
        refusal_indices = [index for index, option in enumerate(question["options"]) if option in REFUSAL_OPTIONS]
        if status == "known" and (sources or refusal_indices):
            raise ExportError("known question contains uncertainty metadata or a refusal option")
        if status == "uncertain" and (not sources or refusal_indices != [question["answer_index"]]):
            raise ExportError("uncertain question does not select exactly one refusal option")


def _resolve_raw_episode_dirs(source: Path) -> list[Path]:
    cursor = source
    seen: set[Path] = set()
    while cursor not in seen:
        seen.add(cursor)
        summary_path = cursor / "summary.json"
        four_path = cursor / "four_scene_summary.json"
        summary = _read_json(summary_path) if summary_path.is_file() else {}
        raw = summary.get("raw_episode_dirs", [])
        if raw:
            return [Path(path) for path in raw]
        four = _read_json(four_path) if four_path.is_file() else {}
        raw = four.get("reference_aligned_summary", {}).get("raw_episode_dirs", [])
        if raw:
            return [Path(path) for path in raw]
        parent = summary.get("parent_dataset") or four.get("parent_dataset")
        if not parent:
            break
        cursor = Path(parent).expanduser().resolve()
    raise ExportError("could not resolve raw episode directories through dataset ancestry")


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
    model_inputs = [
        json.loads(line)
        for line in (output / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = _read_json(output / "evaluation_labels.json")
    if len(model_inputs) != len(questions) or len(labels) != len(questions):
        raise ExportError("evaluation bundle does not cover every question")
    question_by_id = {question["id"]: question for question in questions}
    for record in model_inputs:
        if set(record) != MODEL_INPUT_KEYS:
            raise ExportError("model input record violates the prompt allowlist")
        question = question_by_id.get(record["question_id"])
        if question is None or record["prompt"] != build_model_prompt(question):
            raise ExportError("model input prompt differs from the allowlisted prompt builder")
        if float(record["query_time"]) != float(question["query_time"]):
            raise ExportError("model input query_time differs from QA")
        private_dump = json.dumps(record, ensure_ascii=False)
        if any(key in private_dump for key in ("answer_index", "diagnostics", "video_evidence", "diagnostic_rationale")):
            raise ExportError("private QA metadata leaked into a model input record")
        label = labels.get(question["id"])
        if label is None or int(label["answer_index"]) != int(question["answer_index"]):
            raise ExportError("evaluation label differs from QA ground truth")
    type_counts = Counter(question["question_type"] for question in questions)
    status_counts = Counter(question["diagnostics"]["epistemic_status"] for question in questions)
    answer_counts = Counter(question["answer_index"] for question in questions)
    expected_types = {
        "current_state": 54,
        "factual_retrieval": 54,
        "history_aggregation": 51,
        "state_change": 51,
        "object_tracking": 50,
        "temporal_reasoning": 40,
    }
    if dict(type_counts) != expected_types:
        raise ExportError(f"question type counts changed: {dict(type_counts)}")
    if dict(status_counts) != {"known": 245, "uncertain": 55}:
        raise ExportError(f"epistemic distribution changed: {dict(status_counts)}")
    if dict(answer_counts) != {0: 75, 1: 75, 2: 75, 3: 75}:
        raise ExportError(f"answer positions changed: {dict(answer_counts)}")
    audit = _audit_questions(questions)
    if not audit["static_release_passed"]:
        raise ExportError(f"static release gate failed: {audit}")
    source_counts = Counter(
        source
        for question in questions
        for source in question["diagnostics"]["uncertainty_sources"]
    )
    type_status_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for question in questions:
        type_status_counts[question["question_type"]][question["diagnostics"]["epistemic_status"]] += 1
    return {
        "version": VERSION,
        "candidate_status": "static_audit_passed_dynamic_video_gain_pending",
        "episode_count": len(episode_ids),
        "video_count": len(videos),
        "question_count": len(questions),
        "question_type_counts": dict(type_counts),
        "epistemic_status_counts": dict(status_counts),
        "question_type_epistemic_counts": {key: dict(value) for key, value in type_status_counts.items()},
        "uncertainty_source_counts": dict(source_counts),
        "answer_index_counts": {str(index): answer_counts[index] for index in range(4)},
        "visual_grounding_audit": audit,
    }


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    audit = summary["visual_grounding_audit"]
    lines = [
        "# Sim QA v2.1 视觉必要性修复与分布报告",
        "",
        "## 1. 版本结论",
        "",
        f"- 父数据集：`{source}`（v2 保持不变）。",
        f"- 修复数据集：`{output}`。",
        "- v2 动态评测失败：text-only 22.00%，video offline 23.67%，视觉仅 +1.67 pp。",
        "- v2.1 的 245 条 Known 全部移除拒答选项；55 条 Uncertain 保留拒答，但不进入主视觉准确率。",
        "- 当前仅通过静态门禁；必须重新跑 text-only / matched-video / shuffled-video 后才能发布。",
        "",
        "## 2. QA 所在场景",
        "",
    ]
    lines.extend(f"- {description}" for description in ROOM_DESCRIPTIONS.values())
    lines.extend([
        "",
        "| 场景 | 视频 | QA | 占比 |",
        "|---|---:|---:|---:|",
        "| 厨房 | 7 | 75 | 25% |",
        "| 客厅 | 7 | 75 | 25% |",
        "| 卧室 | 7 | 75 | 25% |",
        "| 卫生间 | 7 | 75 | 25% |",
        "| 合计 | 28 | 300 | 100% |",
        "",
        "## 3. 问题分布",
        "",
        "| 问题类型 | 数量 | 视觉修复 |",
        "|---|---:|---|",
        "| `current_state` | 54 | 单帧唯一目标：四个同场景真实物体中恰好一个满足原位可见/不可见，答案必须落到具体物体。 |",
        "| `factual_retrieval` | 54 | 四个同视频真实物体名，判断短窗口内哪个物体出现/消失。 |",
        "| `history_aggregation` | 51 | 聚合短窗口内发生消失变化的物体集合；四个候选集合大小相同，不能由时间窗长度推出答案。 |",
        "| `state_change` | 51 | 四个同格式的具体可见转移，不含拒答填充项。 |",
        "| `object_tracking` | 50 | Known 用四个真实物体名；身份不可证问题仅留在 Uncertain。 |",
        "| `temporal_reasoning` | 40 | 四个真实事件候选，判断局部直接前驱。 |",
        "",
        "Known 245 条，Uncertain 55 条，总计 300 条；ABCD 正确位置各 75 条。",
        "",
        "## 4. 针对本次失败的修复",
        "",
        "1. **移除 Known 的拒答黑洞。** v2 的 245 条 Known 都含证据不足项，闭卷选了 180 次、视频仍选了 142 次。v2.1 Known 的拒答项数量为 0。",
        "2. **把 current_state 改成单帧唯一目标。** 不再要求双物体联合逻辑，也不让出现/消失方向随时间泄漏；四个选项都是同场景真实物体，画面中恰好一个满足状态。",
        "3. **消除计数时间公式。** history 不再让模型用窗口长度除以固定事件间隔，改为区分同样大小的真实物体集合。所有 evidence span 都是非零区间。",
        "4. **消除未来时间冲突。** 题干中出现的每个时间都不晚于 query_time；Uncertain 不再引用 query 之后的窗口终点。",
        "5. **去掉题干元语言。** Known 题不再使用 supported、uniquely、evidence 等诱导保守回答的词。",
        "6. **分开报告不确定性。** 主指标只看 Known；Uncertain 用 epistemic accuracy 单独报告，不能用 55 条易拒答样本抬高总准确率。",
        "",
        "## 5. 静态审计结果",
        "",
        f"- Known 中拒答选项：{audit['known_refusal_option_count']}。",
        f"- Uncertain 拒答 GT：{audit['uncertain_refusal_correct_count']}/{audit['uncertain_count']}。",
        f"- 题干未来时间引用：{len(audit['future_time_reference_hits'])}。",
        f"- 零长度 evidence span：{len(audit['zero_length_evidence_span_hits'])}。",
        f"- 题干元语言提示：{len(audit['question_meta_cue_hits'])}。",
        f"- Known 公开时间字段到答案文本的最大简单规则：{audit['maximum_known_answer_text_prior_accuracy']:.2%}。",
        f"- 最大静态简单规则：{audit['maximum_static_rule_accuracy']:.2%}（门槛 30%）。",
        f"- 静态发布门禁：{'通过' if audit['static_release_passed'] else '未通过'}。",
        "",
        "## 6. 动态验收协议",
        "",
        "同一模型、同一 prompt 运行三组：text-only、matched video、shuffled/mismatched video。所有视频采样必须硬截断到 `timestamp <= query_time`，不能把完整 61 秒视频交给 offline 模式。主指标为 245 条 Known：",
        "",
        "- matched-video 相对 text-only 至少提升 10 个百分点；",
        "- shuffled-video 回落到 text-only 附近（差值不超过 3 个百分点）；",
        "- `factual_retrieval`、`current_state`、`state_change` 各自至少提升 8 个百分点；",
        "- 未通过前目录状态保持 candidate，不覆盖 v2，也不标为正式版本。",
        "",
    ])
    return "\n".join(lines)


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
            v1._materialize_video(path, target, link_mode)
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

    raw_episode_dirs = _resolve_raw_episode_dirs(source)
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
            "qa_source": QA_SOURCE,
            "questions": questions,
        })
        records.append((target_path, document))

    stratified_positions = _assign_balanced_options(all_questions, seed)
    for target_path, document in records:
        for index, question in enumerate(document["questions"], start=1):
            question["id"] = f"{document['episode_id']}_{ID_VERSION_TAG}_q{index:02d}"
            question.pop("_refusal_option")
            question.pop("_semantic_class")
        _write_json(target_path, document)
        _validate_document(document, target_path)
    _write_jsonl(output / "questions.jsonl", all_questions)

    model_inputs = []
    evaluation_labels = {}
    for _, document in records:
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
    split_manifest = {
        "primary_visual_known": [question["id"] for question in all_questions if question["diagnostics"]["epistemic_status"] == "known"],
        "epistemic_uncertain": [question["id"] for question in all_questions if question["diagnostics"]["epistemic_status"] == "uncertain"],
        "headline_metric": "answer_accuracy_on_primary_visual_known",
        "compatibility_metric": "answer_accuracy_on_all_questions_not_for_headline_use",
    }
    _write_json(output / "evaluation_splits.json", split_manifest)
    _write_json(output / "evaluation_protocol.json", {
        "video_frame_rule": "timestamp <= query_time",
        "full_video_offline_allowed": False,
        "required_conditions": ["text_only", "matched_video", "shuffled_or_mismatched_video"],
        "headline_split": "primary_visual_known",
        "minimum_matched_video_gain": 0.10,
        "maximum_shuffled_video_gain": 0.03,
    })

    preliminary = validate_export(output)
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
            "maximum_shuffled_video_gain": 0.03,
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
        "evaluation_protocol": str((output / "evaluation_protocol.json").resolve()),
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
    report_path = output / REPORT_FILENAME
    report_path.write_text(_report_markdown(summary, source, output), encoding="utf-8")

    if _source_fingerprint(source) != source_fingerprint:
        raise ExportError("source dataset changed during v2.1 generation")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = (
        validate_export(args.output)
        if args.validate_only
        else export_clone(args.source_dataset, args.output, link_mode=args.link_mode, seed=args.seed)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
