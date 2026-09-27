"""Export accepted rollouts in the reference video-QA schema.

Each episode receives 10--12 questions. Question types and answer positions
follow dataset-wide quotas.
"""
from __future__ import annotations

import argparse
import errno
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .event_adapter import build_events
from .exporter import _episode_id, _load_accepted_episode, _read_json


QUESTION_TYPES = (
    "factual_retrieval",
    "current_state",
    "history_aggregation",
    "state_change",
    "object_tracking",
    "temporal_reasoning",
)

# Counts measured from the 5,851-question reference set.
REFERENCE_TYPE_COUNTS = {
    "current_state": 1049,
    "factual_retrieval": 1045,
    "history_aggregation": 998,
    "state_change": 993,
    "object_tracking": 982,
    "temporal_reasoning": 784,
}
REFERENCE_UNCERTAIN_RATES = {
    "current_state": 17 / 1049,
    "factual_retrieval": 41 / 1045,
    "history_aggregation": 227 / 998,
    "state_change": 65 / 993,
    "object_tracking": 481 / 982,
    "temporal_reasoning": 248 / 784,
}
ALLOWED_UNCERTAINTY_SOURCES = {
    "missing_observation",
    "partial_observation",
    "ambiguous_evidence",
    "multiple_candidates",
}
FORBIDDEN_UNCERTAINTY_SOURCES = {
    "low_visual_quality",
    "ambiguous_attribute",
}
UNCERTAIN_ANSWER_VARIANTS = {
    "factual_retrieval": (
        "The object's exact off-screen location is not shown",
        "The recorded views do not reveal where the object is during this interval",
        "There is insufficient visible evidence to locate the object",
    ),
    "current_state": (
        "The object's current location is not established by the visible history",
        "The available frames do not reveal the object's current location",
        "No unique current location can be inferred from the recorded views",
    ),
    "state_change": (
        "The visible frames do not show how the object was removed",
        "The removal method cannot be established from the recorded views",
        "There is insufficient visual evidence to identify the transition mechanism",
        "How the state changed is not observable in the recorded sequence",
        "The causal action is missing from the visible history",
    ),
    "object_tracking": (
        "The visible history does not prove that it is the same physical instance",
        "Instance identity cannot be resolved from the recorded frames",
        "The observations do not rule out a visually similar replacement",
        "The cross-time identity is ambiguous in the visible record",
        "The recorded views are insufficient to verify instance continuity",
    ),
    "temporal_reasoning": (
        "The order of the off-screen actions is not observable",
        "The recorded frames do not establish which hidden action occurred first",
        "There is insufficient visible evidence to order the unobserved actions",
        "The hidden actions cannot be placed in a unique temporal order",
        "No action order can be recovered from the available visual record",
    ),
    "history_aggregation": (
        "The number of off-screen manipulations is not observable",
        "The visible record is insufficient to count the hidden manipulations",
        "No exact count can be derived from the recorded frames",
        "The unobserved actions cannot be counted from the visible history",
        "The available frames do not support a unique manipulation count",
    ),
}
ALL_UNCERTAIN_ANSWERS = {
    answer
    for variants in UNCERTAIN_ANSWER_VARIANTS.values()
    for answer in variants
}
EPISODE_KEYS = {
    "episode_id",
    "video_path",
    "duration_sec",
    "sample_fps",
    "qa_source",
    "sample_timestamps",
    "questions",
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
    "change_intensity",
}

SPECIAL_OBJECT_NAMES = {
    "ButterKnife": "butter knife",
    "PepperShaker": "pepper shaker",
    "SaltShaker": "salt shaker",
    "SoapBottle": "soap bottle",
}
SURFACE_NAMES = {
    "操作台": "countertop",
    "餐桌": "dining table",
    "灶台": "stove burner",
    "茶几": "coffee table",
    "边桌": "side table",
    "椅子": "chair",
    "沙发": "sofa",
    "电视柜": "TV stand",
    "床": "bed",
    "书桌": "desk",
    "斗柜": "dresser",
    "置物架": "shelf",
    "浴缸": "bathtub",
    "浴缸内": "bathtub basin",
    "地面": "floor",
    "擦手巾架": "hand-towel holder",
    "洗手池": "sink",
    "水槽内": "sink basin",
    "毛巾架": "towel holder",
}


class ExportError(ValueError):
    """Raised when an aligned export violates its contract."""


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _friendly_object(event: dict[str, Any]) -> str:
    object_type = str(event["_target_id"]).split("|", 1)[0]
    if object_type in SPECIAL_OBJECT_NAMES:
        return SPECIAL_OBJECT_NAMES[object_type]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", object_type).lower()


def _surface(event: dict[str, Any]) -> str:
    return SURFACE_NAMES.get(str(event.get("_surface", "")), "supporting surface")


def _ceil_sample(value: float, maximum: int) -> float:
    return float(min(maximum, max(0, math.ceil(float(value) - 1e-9))))


def _floor_sample(value: float, maximum: int) -> float:
    return float(min(maximum, max(0, math.floor(float(value) + 1e-9))))


def _event_points(event: dict[str, Any], maximum: int) -> tuple[float, float]:
    before = _floor_sample(float(event["start_sec"]), maximum)
    after = _ceil_sample(float(event["end_sec"]), maximum)
    if after < before:
        after = before
    return before, after


def _largest_remainder(
    total: int,
    weights: dict[str, float],
    *,
    capacities: dict[str, int] | None = None,
) -> dict[str, int]:
    if total < 0 or not weights or sum(weights.values()) <= 0:
        raise ValueError("invalid largest-remainder allocation")
    weight_sum = float(sum(weights.values()))
    raw = {key: total * float(weight) / weight_sum for key, weight in weights.items()}
    result = {key: int(math.floor(value)) for key, value in raw.items()}
    if capacities is not None:
        result = {key: min(result[key], capacities[key]) for key in result}
    remaining = total - sum(result.values())
    order = sorted(
        weights,
        key=lambda key: (raw[key] - math.floor(raw[key]), weights[key], key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if capacities is not None and result[key] >= capacities[key]:
                continue
            result[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise ValueError("allocation exceeds capacities")
    return result


def _type_quotas(question_count: int) -> dict[str, int]:
    return _largest_remainder(question_count, REFERENCE_TYPE_COUNTS)


def _uncertain_quotas(type_quotas: dict[str, int]) -> dict[str, int]:
    target_total = round(sum(type_quotas.values()) * (1079 / 5851))
    expected = {
        key: type_quotas[key] * REFERENCE_UNCERTAIN_RATES[key]
        for key in QUESTION_TYPES
    }
    floors = {key: min(type_quotas[key], math.floor(value)) for key, value in expected.items()}
    remaining = target_total - sum(floors.values())
    for key in sorted(
        QUESTION_TYPES,
        key=lambda item: (expected[item] - math.floor(expected[item]), item),
        reverse=True,
    ):
        if not remaining:
            break
        if floors[key] < type_quotas[key]:
            floors[key] += 1
            remaining -= 1
    if remaining:
        raise ValueError("could not allocate uncertain questions")
    return floors


def _episode_type_schedule(
    episode_count: int,
    type_quotas: dict[str, int],
) -> list[list[str]]:
    """Spread exact global quotas over episodes containing 10--12 questions."""
    question_count = sum(type_quotas.values())
    if not episode_count * 10 <= question_count <= episode_count * 12:
        raise ValueError("reference-aligned episodes must contain 10 to 12 questions")
    if any(value < episode_count for value in type_quotas.values()):
        raise ValueError("every episode must contain all six question types")
    schedules = [list(QUESTION_TYPES) for _ in range(episode_count)]
    rng = random.Random(20260819)
    tie_breakers = {
        (question_type, episode_index): rng.random()
        for question_type in QUESTION_TYPES
        for episode_index in range(episode_count)
    }
    for question_type in QUESTION_TYPES:
        remaining = type_quotas[question_type] - episode_count
        while remaining:
            candidates = [
                episode_index
                for episode_index, schedule in enumerate(schedules)
                if schedule.count(question_type) < 3 and len(schedule) < 12
            ]
            if not candidates:
                raise ValueError("could not distribute question-type quota")
            episode_index = min(
                candidates,
                key=lambda index: (
                    len(schedules[index]),
                    schedules[index].count(question_type),
                    tie_breakers[(question_type, index)],
                ),
            )
            schedules[episode_index].append(question_type)
            remaining -= 1
    if any(not 10 <= len(schedule) <= 12 for schedule in schedules):
        raise ValueError("per-episode question counts are outside 10--12")
    for episode_index, schedule in enumerate(schedules):
        random.Random(20260819 + episode_index).shuffle(schedule)
    return schedules


def _spread_indices(total: int, selected: int) -> set[int]:
    if not 0 <= selected <= total:
        raise ValueError("invalid spread selection")
    if not selected:
        return set()
    result = {
        min(total - 1, math.floor((index + 0.5) * total / selected))
        for index in range(selected)
    }
    cursor = 0
    while len(result) < selected:
        if cursor not in result:
            result.add(cursor)
        cursor += 1
    return result


def _pairs(events: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    disappearances = {
        event["_target_id"]: event
        for event in events
        if event["event_type"] == "disappearance"
    }
    appearances = {
        event["_target_id"]: event
        for event in events
        if event["event_type"] == "appearance"
    }
    shared = [key for key in disappearances if key in appearances]
    result = [(disappearances[key], appearances[key]) for key in shared]
    result.sort(key=lambda pair: float(pair[0]["start_sec"]))
    if len(result) < 4:
        raise ExportError("an episode needs at least four disappearance/appearance pairs")
    return result


def _object_distractors(
    events: list[dict[str, Any]], correct: str,
) -> list[str]:
    labels = list(dict.fromkeys(_friendly_object(event) for event in events))
    choices = [label for label in labels if label != correct]
    if len(choices) < 3:
        choices.extend(label for label in ("mug", "plate", "spoon", "kettle") if label not in choices and label != correct)
    return choices[:3]


def _uncertain_answer(question_type: str, ordinal: int) -> str:
    variants = UNCERTAIN_ANSWER_VARIANTS[question_type]
    return variants[ordinal % len(variants)]


def _draft(
    *,
    episode_id: str,
    question_type: str,
    subtype: str,
    query_time: float,
    video_evidence: str,
    question: str,
    correct: str,
    distractors: list[str],
    evidence_spans: list[list[float]],
    uncertain: bool,
    volatility: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(set([correct, *distractors])) != 4:
        raise ExportError("question choices are not distinct")
    return {
        "id": "",
        "episode_id": episode_id,
        "query_time": float(query_time),
        "question_type": question_type,
        "question_subtype": subtype,
        "video_evidence": video_evidence,
        "question": question,
        "options": [],
        "answer_index": -1,
        "evidence_spans": evidence_spans,
        "diagnostics": {
            "epistemic_status": "uncertain" if uncertain else "known",
            "uncertainty_sources": [],
        },
        "diagnostic_rationale": {
            "volatility": volatility,
            "uncertainty": (
                "The rendered observations uniquely support the selected answer."
                if not uncertain
                else "The visible history does not uniquely determine the answer."
            ),
        },
        "change_intensity": max(
            1,
            sum(float(event["start_sec"]) <= query_time + 1e-6 for event in events),
        ),
        "_correct": correct,
        "_distractors": distractors,
    }


def _build_question(
    episode_id: str,
    events: list[dict[str, Any]],
    question_type: str,
    ordinal: int,
    uncertain: bool,
    maximum_time: int,
    uncertain_ordinal: int | None = None,
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda item: (item["start_sec"], item["end_sec"]))
    event_pairs = _pairs(ordered)
    disappearance, appearance = event_pairs[ordinal % len(event_pairs)]
    event = ordered[(ordinal * 3 + (1 if uncertain else 0)) % len(ordered)]
    label = _friendly_object(event)
    place = _surface(event)
    before, after = _event_points(event, maximum_time)
    pair_label = _friendly_object(disappearance)
    pair_place = _surface(disappearance)
    dis_before, dis_after = _event_points(disappearance, maximum_time)
    app_before, app_after = _event_points(appearance, maximum_time)
    uncertain_answer = _uncertain_answer(
        question_type,
        ordinal if uncertain_ordinal is None else uncertain_ordinal,
    )
    if question_type == "factual_retrieval":
        if uncertain:
            query = app_before
            return _draft(
                episode_id=episode_id,
                question_type=question_type,
                subtype="offscreen_location_retrieval",
                query_time=query,
                video_evidence=(
                    f"The {pair_label} is no longer at its original {pair_place} position by {dis_after:.1f}s, "
                    f"and the next direct observation of it is not until {app_after:.1f}s."
                ),
                question=f"Where exactly is the {pair_label} at {query:.1f}s while it is outside the recorded view?",
                correct=uncertain_answer,
                distractors=[f"Inside a cabinet", f"On the floor", f"On a different countertop"],
                evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
                uncertain=True,
                volatility="The requested fact falls inside an interval with no direct observation of the target.",
                events=ordered,
            )
        direction = "disappears from" if event["event_type"] == "disappearance" else "reappears at"
        return _draft(
            episode_id=episode_id,
            question_type=question_type,
            subtype="changed_object_retrieval",
            query_time=after,
            video_evidence=f"Between {before:.1f}s and {after:.1f}s, the {label} {direction} its original position on the {place}.",
            question=f"Which object {direction} its original {place} position by {after:.1f}s?",
            correct=label,
            distractors=_object_distractors(ordered, label),
            evidence_spans=[[before, before], [after, after]],
            uncertain=False,
            volatility="The question retrieves one completed visible transition from earlier in the episode.",
            events=ordered,
        )

    if question_type == "current_state":
        if uncertain:
            query = max(dis_after, _floor_sample((float(disappearance["end_sec"]) + float(appearance["start_sec"])) / 2, maximum_time))
            return _draft(
                episode_id=episode_id,
                question_type=question_type,
                subtype="exact_location_during_missing_interval",
                query_time=query,
                video_evidence=f"The {pair_label} leaves its original {pair_place} position by {dis_after:.1f}s and remains outside the visible frames at {query:.1f}s.",
                question=f"What is the exact current location of the {pair_label} at {query:.1f}s?",
                correct=uncertain_answer,
                distractors=[f"At its original {pair_place} position", "Inside the refrigerator", "On the floor beside the camera"],
                evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
                uncertain=True,
                volatility="The last visible state invalidates the original location but supplies no replacement location.",
                events=ordered,
            )
        restored = ordinal % 2 == 0
        if restored:
            query = app_after
            correct = f"Visible again at its original position on the {pair_place}"
            evidence = [[dis_after, dis_after], [app_after, app_after]]
            summary = f"The {pair_label} is absent by {dis_after:.1f}s and is visibly restored to its original {pair_place} position by {app_after:.1f}s."
        else:
            query = max(dis_after, _floor_sample((float(disappearance["end_sec"]) + float(appearance["start_sec"])) / 2, maximum_time))
            correct = f"Not visible at its original position on the {pair_place}"
            evidence = [[dis_before, dis_before], [dis_after, dis_after]]
            summary = f"The {pair_label} is visible at {dis_before:.1f}s and absent from its original {pair_place} position by {dis_after:.1f}s."
        return _draft(
            episode_id=episode_id,
            question_type=question_type,
            subtype="current_visible_object_state",
            query_time=query,
            video_evidence=summary,
            question=f"What is the visible state of the {pair_label} at {query:.1f}s?",
            correct=correct,
            distractors=[
                f"Visible at a different position on the {pair_place}",
                "Visible in the agent's hand",
                "Visibly broken into pieces",
            ],
            evidence_spans=evidence,
            uncertain=False,
            volatility="The answer depends on the latest visible revision before the query time.",
            events=ordered,
        )

    if question_type == "state_change":
        if uncertain:
            return _draft(
                episode_id=episode_id,
                question_type=question_type,
                subtype="unobserved_transition_mechanism",
                query_time=dis_after,
                video_evidence=f"The {pair_label} is present at {dis_before:.1f}s and absent from the same {pair_place} position at {dis_after:.1f}s, but the manipulation itself is not recorded.",
                question=f"How was the {pair_label} removed from the {pair_place} between {dis_before:.1f}s and {dis_after:.1f}s?",
                correct=uncertain_answer,
                distractors=["It was picked up by the agent", "It slid off the surface", "It was moved by another object"],
                evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
                uncertain=True,
                volatility="Only the before and after states are visible; the causal process is hidden.",
                events=ordered,
            )
        if event["event_type"] == "disappearance":
            correct = f"The {label} changes from visible to absent at its original {place} position"
            reverse = f"The {label} changes from absent to visible at its original {place} position"
        else:
            correct = f"The {label} changes from absent to visible at its original {place} position"
            reverse = f"The {label} changes from visible to absent at its original {place} position"
        return _draft(
            episode_id=episode_id,
            question_type=question_type,
            subtype="object_visibility_state_transition",
            query_time=after,
            video_evidence=f"The rendered before frame at {before:.1f}s and after frame at {after:.1f}s show a visibility transition for the {label} at the {place}.",
            question=f"What visible state change occurs to the {label} between {before:.1f}s and {after:.1f}s?",
            correct=correct,
            distractors=[reverse, f"The {label} remains continuously visible", f"The {label} visibly changes color"],
            evidence_spans=[[before, before], [after, after]],
            uncertain=False,
            volatility="The answer requires comparing distinct visible states before and after the event.",
            events=ordered,
        )

    if question_type == "object_tracking":
        if uncertain:
            return _draft(
                episode_id=episode_id,
                question_type=question_type,
                subtype="cross_gap_instance_identity",
                query_time=app_after,
                video_evidence=f"A {pair_label} disappears from the {pair_place} by {dis_after:.1f}s. After an unobserved interval, a category-matching {pair_label} is visible there at {app_after:.1f}s.",
                question=f"Is the {pair_label} seen at {app_after:.1f}s provably the exact same physical instance seen before {dis_after:.1f}s?",
                correct=uncertain_answer,
                distractors=["Yes, because the category and position match", "No, it is visibly a different instance", "No, the earlier object was visibly destroyed"],
                evidence_spans=[[dis_before, dis_before], [dis_after, dis_after], [app_after, app_after]],
                uncertain=True,
                volatility="Identity must be maintained across a period in which replacement is not visually ruled out.",
                events=ordered,
            )
        return _draft(
            episode_id=episode_id,
            question_type=question_type,
            subtype="simulator_instance_reappearance",
            query_time=app_after,
            video_evidence=f"The tracked simulator instance of the {pair_label} disappears from the {pair_place} by {dis_after:.1f}s and the same instance is rendered again at its original position by {app_after:.1f}s.",
            question=f"Which tracked object instance returns to its original {pair_place} position at {app_after:.1f}s?",
            correct=pair_label,
            distractors=_object_distractors(ordered, pair_label),
            evidence_spans=[[dis_before, dis_before], [dis_after, dis_after], [app_after, app_after]],
            uncertain=False,
            volatility="The simulator identity trace links two observations separated by a disappearance interval.",
            events=ordered,
        )

    if question_type == "temporal_reasoning":
        first = ordered[ordinal % (len(ordered) - 1)]
        second = ordered[(ordinal % (len(ordered) - 1)) + 1]
        first_label = _friendly_object(first)
        second_label = _friendly_object(second)
        first_action = "disappears" if first["event_type"] == "disappearance" else "reappears"
        second_action = "disappears" if second["event_type"] == "disappearance" else "reappears"
        _, first_after = _event_points(first, maximum_time)
        _, second_after = _event_points(second, maximum_time)
        if uncertain:
            query = app_before
            return _draft(
                episode_id=episode_id,
                question_type=question_type,
                subtype="order_of_hidden_actions",
                query_time=query,
                video_evidence=f"The {pair_label} is absent after {dis_after:.1f}s and is not observed again before {query:.1f}s; no off-screen actions are recorded in between.",
                question=(
                    f"Which off-screen action involving the {pair_label} happened first "
                    f"between {dis_after:.1f}s and {app_before:.1f}s?"
                ),
                correct=uncertain_answer,
                distractors=["The object was picked up first", "The object was placed elsewhere first", "The object was rotated first"],
                evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
                uncertain=True,
                volatility="The requested ordering concerns actions that fall entirely between sampled observations.",
                events=ordered,
            )
        correct = f"The {first_label} {first_action} before the {second_label} {second_action}"
        return _draft(
            episode_id=episode_id,
            question_type=question_type,
            subtype="visible_event_order",
            query_time=second_after,
            video_evidence=f"The {first_label} {first_action} by {first_after:.1f}s, while the {second_label} {second_action} later by {second_after:.1f}s.",
            question=(
                f"Which statement correctly orders the {first_label} and {second_label} "
                f"events by {second_after:.1f}s?"
            ),
            correct=correct,
            distractors=[
                f"The {second_label} {second_action} before the {first_label} {first_action}",
                "Both events occur at the same sampled time",
                "Neither event occurs before the query time",
            ],
            evidence_spans=[[first_after, first_after], [second_after, second_after]],
            uncertain=False,
            volatility="The answer requires ordering two episode-specific events rather than applying a semantic prior.",
            events=ordered,
        )

    if question_type == "history_aggregation":
        if uncertain:
            return _draft(
                episode_id=episode_id,
                question_type=question_type,
                subtype="unobserved_action_count",
                query_time=app_before,
                video_evidence=f"The visible record only establishes that the {pair_label} leaves its original position by {dis_after:.1f}s and is not observed again before {app_before:.1f}s.",
                question=f"How many separate off-screen manipulations of the {pair_label} occur between {dis_after:.1f}s and {app_before:.1f}s?",
                correct=uncertain_answer,
                distractors=["Exactly one manipulation", "Exactly two manipulations", "No manipulations"],
                evidence_spans=[[dis_before, dis_before], [dis_after, dis_after]],
                uncertain=True,
                volatility="The requested aggregate includes an interval with no visible observation units.",
                events=ordered,
            )
        disappearance_events = [item for item in ordered if item["event_type"] == "disappearance"]
        count = 2 + ordinal % max(1, len(disappearance_events) - 1)
        selected = disappearance_events[:count]
        query = _event_points(selected[-1], maximum_time)[1]
        spans = [[_event_points(item, maximum_time)[1]] * 2 for item in selected]
        visible_labels = ", ".join(_friendly_object(item) for item in selected)
        correct = f"{count} distinct visible disappearance events"
        numeric = [value for value in (count - 1, count + 1, max(0, count - 2), count + 2) if value != count]
        return _draft(
            episode_id=episode_id,
            question_type=question_type,
            subtype="visible_disappearance_event_count",
            query_time=query,
            video_evidence=f"By {query:.1f}s, the visible disappearance events involve {visible_labels}; consecutive sampled frames of one transition count as one event.",
            question=(
                f"Counting distinct visible disappearance events involving {visible_labels}, "
                f"how many have completed by {query:.1f}s?"
            ),
            correct=correct,
            distractors=[f"{value} distinct visible disappearance events" for value in numeric[:3]],
            evidence_spans=spans,
            uncertain=False,
            volatility="The answer aggregates multiple completed visual events up to an explicit query boundary.",
            events=ordered,
        )

    raise ValueError(f"unsupported question type: {question_type}")


def _assign_uncertainty_sources(questions: list[dict[str, Any]]) -> None:
    uncertain = [
        question for question in questions
        if question["diagnostics"]["epistemic_status"] == "uncertain"
    ]
    count = len(uncertain)
    if not count:
        return
    missing_count = round(count * (920 / 1079))
    ambiguous_count = round(count * (305 / 1079))
    partial_count = round(count * (189 / 1079))
    multiple_count = round(count * (78 / 1079))
    missing = _spread_indices(count, missing_count)
    nonmissing = set(range(count)) - missing
    if len(nonmissing) > ambiguous_count:
        raise AssertionError("reference source quotas leave unsupported uncertain items")

    def choose_from(indices: set[int], selected_count: int) -> set[int]:
        ordered = sorted(indices)
        positions = _spread_indices(len(ordered), selected_count)
        return {ordered[position] for position in positions}

    ambiguous = nonmissing | choose_from(
        missing,
        ambiguous_count - len(nonmissing),
    )
    selected = {
        "missing_observation": missing,
        "ambiguous_evidence": ambiguous,
        "partial_observation": choose_from(missing, partial_count),
        "multiple_candidates": choose_from(missing, multiple_count),
    }
    for index, question in enumerate(uncertain):
        sources = [source for source in ALLOWED_UNCERTAINTY_SOURCES if index in selected[source]]
        if not sources:
            raise AssertionError("uncertain question received no source")
        sources.sort()
        question["diagnostics"]["uncertainty_sources"] = sources
        readable = ", ".join(source.replace("_", " ") for source in sources)
        question["diagnostic_rationale"]["uncertainty"] = (
            f"The answer is not uniquely recoverable because of {readable}; the benchmark therefore marks the evidence-insufficiency option as correct."
        )


def _assign_balanced_options(questions: list[dict[str, Any]]) -> None:
    count = len(questions)
    positions = [index % 4 for index in range(count)]
    random.Random(20260820).shuffle(positions)
    for question, answer_index in zip(questions, positions):
        correct = question.pop("_correct")
        distractors = question.pop("_distractors")
        options = list(distractors)
        options.insert(answer_index, correct)
        question["options"] = options
        question["answer_index"] = answer_index


def _materialize_video(source: Path, target: Path, link_mode: str) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing source rollout: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to replace video: {target}")
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


def _load_raw_episode(
    episode_dir: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load one individually accepted episode from an incomplete batch."""
    directory = Path(episode_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"episode directory does not exist: {directory}")
    validation = _read_json(directory / "episode_validation.json")
    trace = _read_json(directory / "execution_trace.json")
    if not validation.get("accepted") or not trace.get("accepted"):
        raise ExportError(f"episode is not individually accepted: {directory}")
    if trace.get("failed_frames"):
        raise ExportError(f"episode contains failed frames: {directory}")
    plan = _read_json(directory / "episode_plan.json")
    profile = _read_json(directory / "scene_profile.json")
    video = directory / "rollout.mp4"
    if not video.is_file() or video.stat().st_size <= 0:
        raise FileNotFoundError(f"episode rollout is missing: {video}")
    return directory, plan, trace, profile


def _validate_document(document: dict[str, Any], json_path: Path) -> None:
    if set(document) != EPISODE_KEYS:
        raise ExportError(f"episode keys differ from the reference schema: {json_path}")
    episode_id = document["episode_id"]
    if json_path.stem != episode_id:
        raise ExportError("episode id and QA JSON stem differ")
    video_path = Path(document["video_path"])
    if not video_path.is_file() or video_path.stem != episode_id:
        raise ExportError("paired video is missing or has a different stem")
    timestamps = {round(float(value), 6) for value in document["sample_timestamps"]}
    if not timestamps:
        raise ExportError("sample timestamps are empty")
    for question in document["questions"]:
        if set(question) != QUESTION_KEYS:
            raise ExportError(f"question keys differ from the reference schema: {question.get('id')}")
        if question["episode_id"] != episode_id or question["question_type"] not in QUESTION_TYPES:
            raise ExportError("question episode or type is invalid")
        if len(question["options"]) != 4 or len(set(question["options"])) != 4:
            raise ExportError("question must have four distinct choices")
        if question["answer_index"] not in range(4):
            raise ExportError("answer index is invalid")
        query_time = round(float(question["query_time"]), 6)
        if query_time not in timestamps:
            raise ExportError("query time is not a supplied sample timestamp")
        for span in question["evidence_spans"]:
            if len(span) != 2 or span[0] > span[1] or span[1] > query_time:
                raise ExportError("evidence span ordering is invalid")
            if round(float(span[0]), 6) not in timestamps or round(float(span[1]), 6) not in timestamps:
                raise ExportError("evidence endpoint is not a supplied sample timestamp")
        diagnostics = question["diagnostics"]
        if set(diagnostics) != {"epistemic_status", "uncertainty_sources"}:
            raise ExportError("diagnostics do not match the reference schema")
        status = diagnostics["epistemic_status"]
        sources = set(diagnostics["uncertainty_sources"])
        if sources & FORBIDDEN_UNCERTAINTY_SOURCES or not sources <= ALLOWED_UNCERTAINTY_SOURCES:
            raise ExportError("unsupported uncertainty source")
        if status == "known" and sources:
            raise ExportError("known question has uncertainty sources")
        if status == "uncertain":
            if not sources:
                raise ExportError("uncertain question has no uncertainty source")
            correct = question["options"][question["answer_index"]]
            if correct not in ALL_UNCERTAIN_ANSWERS:
                raise ExportError("uncertain answer is not the evidence-insufficiency option")


def validate_export(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    qa_root = output / "meta_data" / "qa_results"
    documents = []
    for path in sorted(qa_root.rglob("*.json")):
        document = _read_json(path)
        _validate_document(document, path)
        documents.append(document)
    if not documents:
        raise ExportError("no QA documents found")
    questions = [question for document in documents for question in document["questions"]]
    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ExportError("question ids are not globally unique")
    uncertain_answers = Counter(
        question["options"][question["answer_index"]]
        for question in questions
        if question["diagnostics"]["epistemic_status"] == "uncertain"
    )
    summary = {
        "episode_count": len(documents),
        "video_count": len(list((output / "meta_data" / "gene_videos").rglob("*.mp4"))),
        "question_count": len(questions),
        "question_type_counts": dict(sorted(Counter(question["question_type"] for question in questions).items())),
        "question_type_epistemic_counts": {
            question_type: dict(sorted(Counter(
                question["diagnostics"]["epistemic_status"]
                for question in questions
                if question["question_type"] == question_type
            ).items()))
            for question_type in QUESTION_TYPES
        },
        "epistemic_status_counts": dict(sorted(Counter(question["diagnostics"]["epistemic_status"] for question in questions).items())),
        "answer_index_counts": {str(key): value for key, value in sorted(Counter(question["answer_index"] for question in questions).items())},
        "uncertainty_source_counts": dict(sorted(Counter(source for question in questions for source in question["diagnostics"]["uncertainty_sources"]).items())),
        "uncertain_answer_variant_count": len(uncertain_answers),
        "uncertain_answer_variant_counts": dict(sorted(uncertain_answers.items())),
        "forbidden_uncertainty_source_count": sum(
            source in FORBIDDEN_UNCERTAINTY_SOURCES
            for question in questions
            for source in question["diagnostics"]["uncertainty_sources"]
        ),
    }
    if summary["episode_count"] != summary["video_count"]:
        raise ExportError("QA/video pairing is not one-to-one")
    if sum(uncertain_answers.values()) > 1 and len(uncertain_answers) <= 1:
        raise ExportError("uncertain correct-answer wording is not diverse")
    if summary["question_count"] % 4 == 0:
        expected = summary["question_count"] // 4
        if summary["answer_index_counts"] != {str(index): expected for index in range(4)}:
            raise ExportError("answer positions are not exactly balanced")
    return summary


def export_run(
    run_dir: str | Path | list[str | Path] | None,
    output_dir: str | Path,
    *,
    episode_dirs: list[str | Path] | None = None,
    question_count: int = 300,
    sample_fps: float = 1.0,
    link_mode: str = "hardlink",
) -> dict[str, Any]:
    if run_dir is None:
        raw_runs: list[str | Path] = []
    else:
        raw_runs = run_dir if isinstance(run_dir, list) else [run_dir]
    runs = [Path(item).expanduser().resolve() for item in raw_runs]
    raw_episode_dirs = [
        Path(item).expanduser().resolve() for item in (episode_dirs or [])
    ]
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if sample_fps != 1.0:
        raise ValueError("the reference-aligned first version requires sample_fps=1.0")
    episode_sources: list[tuple[str, Any, Any]] = []
    for run in runs:
        if not run.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run}")
        manifest = _read_json(run / "manifest.json")
        validation = _read_json(run / "batch_validation.json")
        if not manifest.get("accepted") or not validation.get("accepted"):
            raise ExportError(f"only accepted batches can be exported: {run}")
        manifest_episodes = manifest.get("episodes")
        if not isinstance(manifest_episodes, list) or not manifest_episodes:
            raise ExportError(f"accepted batch has no episodes: {run}")
        episode_sources.extend(("manifest", run, episode) for episode in manifest_episodes)
    episode_sources.extend(
        ("raw", directory, None) for directory in raw_episode_dirs
    )
    if not episode_sources:
        raise ExportError("at least one accepted run or episode directory is required")
    if not len(episode_sources) * 10 <= question_count <= len(episode_sources) * 12:
        raise ExportError("question_count must yield 10 to 12 questions per episode")

    type_quotas = _type_quotas(question_count)
    uncertain_quotas = _uncertain_quotas(type_quotas)
    uncertain_indices = {
        key: _spread_indices(type_quotas[key], uncertain_quotas[key])
        for key in QUESTION_TYPES
    }
    episode_schedules = _episode_type_schedule(len(episode_sources), type_quotas)
    ordinals: defaultdict[str, int] = defaultdict(int)
    records: list[tuple[Path, Path, dict[str, Any]]] = []
    all_questions: list[dict[str, Any]] = []

    output.mkdir(parents=True, exist_ok=True)
    for episode_index, (source_kind, source, manifest_episode) in enumerate(episode_sources):
        if source_kind == "manifest":
            episode_dir, plan, trace, profile = _load_accepted_episode(source, manifest_episode)
        else:
            episode_dir, plan, trace, profile = _load_raw_episode(source)
        episode_id = _episode_id(plan).removesuffix("_visit01")
        scene = str(plan["recipe"]["scene"])
        fps = int(plan["trajectory"]["fps"])
        frame_count = len(plan["trajectory"]["frames"])
        duration = round(frame_count / fps, 2)
        maximum_time = int(math.floor(duration + 1e-6))
        timestamps = [float(value) for value in range(maximum_time + 1)]
        events = build_events(plan, trace, profile)
        episode_types = episode_schedules[episode_index]
        questions = []
        for question_type in episode_types:
            ordinal = ordinals[question_type]
            ordinals[question_type] += 1
            type_uncertain_indices = uncertain_indices[question_type]
            is_uncertain = ordinal in type_uncertain_indices
            uncertain_ordinal = sum(
                index < ordinal for index in type_uncertain_indices
            ) if is_uncertain else None
            question = _build_question(
                episode_id,
                events,
                question_type,
                ordinal,
                is_uncertain,
                maximum_time,
                uncertain_ordinal,
            )
            questions.append(question)
            all_questions.append(question)
        video_path = output / "meta_data" / "gene_videos" / scene / f"{episode_id}.mp4"
        json_path = output / "meta_data" / "qa_results" / scene / f"{episode_id}.json"
        document = {
            "episode_id": episode_id,
            "video_path": str(video_path.resolve()),
            "duration_sec": duration,
            "sample_fps": sample_fps,
            "qa_source": "simulation_ground_truth_aligned_to_rendered_video",
            "sample_timestamps": timestamps,
            "questions": questions,
        }
        records.append((episode_dir / "rollout.mp4", json_path, document))

    if dict(ordinals) != type_quotas:
        raise AssertionError(f"type schedule mismatch: {dict(ordinals)} != {type_quotas}")
    _assign_uncertainty_sources(all_questions)
    _assign_balanced_options(all_questions)
    for _, _, document in records:
        for index, question in enumerate(document["questions"], start=1):
            question["id"] = f"{document['episode_id']}_q{index:02d}"

    for source_video, json_path, document in records:
        video_path = Path(document["video_path"])
        _materialize_video(source_video, video_path, link_mode)
        _write_json(json_path, document)
        _validate_document(document, json_path)

    _write_jsonl(output / "questions.jsonl", all_questions)
    summary = validate_export(output)
    summary.update({
        "run_dirs": [str(run) for run in runs],
        "raw_episode_dirs": [str(directory) for directory in raw_episode_dirs],
        "qa_results_dir": str((output / "meta_data" / "qa_results").resolve()),
        "gene_videos_dir": str((output / "meta_data" / "gene_videos").resolve()),
        "questions_jsonl": str((output / "questions.jsonl").resolve()),
        "target_question_type_counts": type_quotas,
        "target_uncertain_type_counts": uncertain_quotas,
    })
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        help="accepted Benchgen run directory; repeat to combine batches",
    )
    parser.add_argument(
        "--episode-dir",
        action="append",
        help="individually accepted episode from an incomplete batch; repeatable",
    )
    parser.add_argument("--output", required=True, help="new aligned dataset root")
    parser.add_argument("--question-count", type=int, default=300)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.validate_only and not (args.run or args.episode_dir):
        parser.error("--run or --episode-dir is required unless --validate-only is set")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        summary = validate_export(args.output)
    else:
        summary = export_run(
            args.run,
            args.output,
            episode_dirs=args.episode_dir,
            question_count=args.question_count,
            sample_fps=args.sample_fps,
            link_mode=args.link_mode,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
