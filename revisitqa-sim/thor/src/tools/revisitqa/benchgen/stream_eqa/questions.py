"""Deterministic, simulator-grounded StreamEQA question generation."""
from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Sequence

from .contracts import QUESTION_TYPES


def _stable_index(episode_id: str, key: str, size: int) -> int:
    if size <= 0:
        raise ValueError("cannot choose from an empty collection")
    digest = hashlib.sha256(f"{episode_id}:{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % size


def _answer_slot(episode_id: str, question_type: str) -> int:
    return _stable_index(episode_id, f"answer:{question_type}", 4)


def _binary_variant(episode_id: str, key: str) -> int:
    """Balance binary styles across Benchgen seeds while staying deterministic."""
    seed_match = re.search(r"_seed(\d+)(?:_|$)", episode_id)
    if seed_match:
        salt = hashlib.sha256(key.encode()).digest()[0] % 2
        return (int(seed_match.group(1)) + salt) % 2
    return _stable_index(episode_id, key, 2)


def _pick_text(episode_id: str, key: str, choices: Sequence[str]) -> str:
    return choices[_stable_index(episode_id, key, len(choices))]


def _choices(
    correct: str, distractors: Iterable[str], answer_slot: int,
) -> tuple[list[str], int]:
    unique = []
    for choice in (correct, *distractors):
        if choice not in unique:
            unique.append(choice)
    if len(unique) < 4:
        raise ValueError("not enough distinct answer choices")
    wrong = [choice for choice in unique if choice != correct][:3]
    result = list(wrong)
    result.insert(answer_slot, correct)
    return result, answer_slot


def _span(event: dict[str, Any]) -> list[float]:
    return [float(event["start_sec"]), float(event["end_sec"])]


def _time_point(seconds: float) -> str:
    """Return a viewer-friendly time while exact timing stays in metadata."""
    return f"about {int(round(float(seconds)))} seconds"


def _time_range(event: dict[str, Any]) -> str:
    return f"{float(event['start_sec']):.1f}-{float(event['end_sec']):.1f} seconds"


def _surface(event: dict[str, Any]) -> str:
    surface = event.get("_surface")
    if isinstance(surface, str) and surface:
        return surface
    location = str(event.get("location", "original position"))
    if location.startswith("on the "):
        location = location.removeprefix("on the ").split(" in ", 1)[0]
    return location or "original position"


def _diagnostics(
    *, revision_count: int = 0, old_answer_indices: list[int] | None = None,
    volatility: str, uncertainty: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    return (
        {
            "revision_count": revision_count,
            "old_answer_indices": list(old_answer_indices or ()),
            "epistemic_status": "known",
            "uncertainty_sources": [],
        },
        {"volatility": volatility, "uncertainty": uncertainty},
    )


def _question(
    episode_id: str,
    question_type: str,
    subtype: str,
    query_time: float,
    video_evidence: str,
    prompt: str,
    correct: str,
    distractors: Iterable[str],
    evidence: list[list[float]],
    diagnostics: tuple[dict[str, Any], dict[str, str]],
    *,
    answer_key: str | None = None,
) -> dict[str, Any]:
    options, answer_index = _choices(
        correct,
        distractors,
        _answer_slot(episode_id, answer_key or question_type),
    )
    diag, rationale = diagnostics
    return {
        "id": "",
        "episode_id": episode_id,
        "query_time": round(float(query_time), 2),
        "question_type": question_type,
        "question_subtype": subtype,
        "video_evidence": video_evidence,
        "question": prompt,
        "options": options,
        "answer_index": answer_index,
        "evidence_spans": evidence,
        "diagnostics": diag,
        "diagnostic_rationale": rationale,
    }


def _event_evidence(event: dict[str, Any]) -> str:
    surface = _surface(event)
    subject = event["subject"]
    if event["event_type"] == "disappearance":
        return (
            f"At {_time_range(event)}, the view returns to the {surface}, and the "
            f"{subject} is no longer at its original position."
        )
    return (
        f"At {_time_range(event)}, the view returns to the {surface}, and the "
        f"{subject} is visible again at its original position."
    )


def _numeric_options(correct_count: int) -> list[str]:
    distractors = []
    for value in (
        max(0, correct_count - 1), correct_count + 1,
        max(0, correct_count - 2), correct_count + 2, 0,
    ):
        option = f"{value} distinct objects"
        if value != correct_count and option not in distractors:
            distractors.append(option)
    return distractors


def build_questions(
    episode_id: str, events: list[dict[str, Any]], duration: float,
) -> list[dict[str, Any]]:
    """Generate one natural, grounded question for each shared EQA type."""
    ordered = sorted(events, key=lambda item: (item["start_sec"], item["end_sec"]))
    disappearances = [item for item in ordered if item["event_type"] == "disappearance"]
    appearances = [item for item in ordered if item["event_type"] == "appearance"]
    if len(disappearances) < 4:
        raise ValueError("at least four disappearance events are required")
    appearance_by_target = {item["_target_id"]: item for item in appearances}
    pairs = []
    for disappearance in disappearances:
        reappearance = appearance_by_target.get(disappearance["_target_id"])
        if reappearance is None:
            raise ValueError(
                f"disappearing target never reappears: {disappearance['subject']}"
            )
        pairs.append((disappearance, reappearance))
    labels = list(dict.fromkeys(item["subject"] for item in disappearances))
    if len(labels) < 4:
        raise ValueError("at least four distinct target labels are required")

    base = _stable_index(episode_id, "target-rotation", len(pairs))
    factual_pair = pairs[base]
    current_pair = pairs[(base + 1) % len(pairs)]
    change_pair = pairs[(base + 2) % len(pairs)]
    tracking_pair = pairs[(base + 3) % len(pairs)]
    result = []

    # Factual retrieval: vary between a disappearance and a reappearance.
    factual_event = factual_pair[
        _binary_variant(episode_id, "factual-direction")
    ]
    factual_surface = _surface(factual_event)
    factual_subject = factual_event["subject"]
    factual_others = [label for label in labels if label != factual_subject]
    if factual_event["event_type"] == "disappearance":
        factual_prompts = (
            f"When the view returns to the {factual_surface} at {_time_range(factual_event)}, "
            "which object is missing from its original position?",
            f"By {_time_point(factual_event['end_sec'])}, which object is no longer "
            f"at its original {factual_surface} position?",
            f"During {_time_range(factual_event)}, which object disappears from "
            f"its original {factual_surface} position?",
        )
    else:
        factual_prompts = (
            f"At {_time_range(factual_event)}, which object reappears at its original "
            f"{factual_surface} position?",
            f"By {_time_point(factual_event['end_sec'])}, which object has returned "
            f"to the {factual_surface}?",
            f"When the view returns to the {factual_surface} at {_time_range(factual_event)}, "
            "which object is visible again?",
        )
    result.append(_question(
        episode_id,
        "factual_retrieval",
        "changed_object_retrieval",
        factual_event["end_sec"],
        _event_evidence(factual_event),
        _pick_text(episode_id, "factual-prompt", factual_prompts),
        factual_subject,
        factual_others,
        [_span(factual_event)],
        _diagnostics(
            volatility="The question asks about a completed visible change that later events do not alter.",
            uncertainty="The views before and after the change establish the object category and location.",
        ),
    ))

    # Current state: ask both hidden and restored states across a batch.
    current_disappearance, current_reappearance = current_pair
    current_surface = _surface(current_disappearance)
    current_subject = current_disappearance["subject"]
    initial_option = f"It remains visible at its original {current_surface} position"
    hidden_option = "It is absent from its original position and has not yet reappeared"
    returned_option = f"It is visible again at its original {current_surface} position"
    moved_option = "It is visible at a different position in the room"
    ask_restored = bool(_binary_variant(episode_id, "current-state"))
    if ask_restored:
        current_query = float(current_reappearance["end_sec"])
        current_correct = returned_option
        current_distractors = [hidden_option, initial_option, moved_option]
        current_evidence = [_span(current_disappearance), _span(current_reappearance)]
        current_video_evidence = (
            f"The {current_subject} disappears from the {current_surface} at "
            f"{_time_range(current_disappearance)} and returns to its original "
            f"position at {_time_range(current_reappearance)}."
        )
        revision_count = 2
        old_option = hidden_option
    else:
        current_query = (
            float(current_disappearance["end_sec"])
            + float(current_reappearance["start_sec"])
        ) / 2.0
        current_correct = hidden_option
        current_distractors = [initial_option, returned_option, moved_option]
        current_evidence = [_span(current_disappearance)]
        current_video_evidence = (
            f"The {current_subject} disappears from the {current_surface} at "
            f"{_time_range(current_disappearance)} and does not reappear before "
            f"{_time_point(current_reappearance['start_sec'])}."
        )
        revision_count = 1
        old_option = initial_option
    current_prompts = (
        f"What is the visible state of the {current_subject} that was originally "
        f"on the {current_surface} at {_time_point(current_query)}?",
        f"At {_time_point(current_query)}, where is the {current_subject} that was "
        f"originally on the {current_surface}?",
        f"Which option describes the visible state of the {current_subject} at "
        f"{_time_point(current_query)}?",
    )
    current = _question(
        episode_id,
        "current_state",
        "object_visibility_current",
        current_query,
        current_video_evidence,
        _pick_text(episode_id, "current-prompt", current_prompts),
        current_correct,
        current_distractors,
        current_evidence,
        _diagnostics(
            revision_count=revision_count,
            old_answer_indices=[],
            volatility="The answer updates when the object disappears or reappears and is fixed at the query time.",
            uncertainty="The relevant changes before the query have clear before-and-after views.",
        ),
    )
    current["diagnostics"]["old_answer_indices"] = [
        current["options"].index(old_option)
    ]
    result.append(current)

    # State change: make the referenced event explicit and vary its direction.
    state_event = change_pair[
        _binary_variant(episode_id, "state-change-direction")
    ]
    state_surface = _surface(state_event)
    state_subject = state_event["subject"]
    disappeared = "It changes from visible to absent at its original position"
    reappeared = "It changes from absent to visible at its original position"
    unchanged = f"It remains visible on the {state_surface}"
    moved = "It becomes visible at a different position in the room"
    if state_event["event_type"] == "disappearance":
        state_correct = disappeared
        state_distractors = [reappeared, unchanged, moved]
    else:
        state_correct = reappeared
        state_distractors = [disappeared, unchanged, moved]
    state_prompts = (
        f"What visible change occurs to the {state_subject} on the {state_surface} "
        f"during {_time_range(state_event)}?",
        f"Comparing the views before and after {_time_range(state_event)}, how does "
        f"the visible state of the {state_subject} change?",
        f"When the view returns to the {state_surface} at {_time_range(state_event)}, "
        f"which change has occurred to the {state_subject}?",
    )
    result.append(_question(
        episode_id,
        "state_change",
        "visibility_state_transition",
        state_event["end_sec"],
        f"Before the change, {state_event['before_state']}; after the change, "
        f"{state_event['after_state']}.",
        _pick_text(episode_id, "state-change-prompt", state_prompts),
        state_correct,
        state_distractors,
        [_span(state_event)],
        _diagnostics(
            revision_count=1,
            old_answer_indices=[],
            volatility="The question identifies a specific interval with different before and after states.",
            uncertainty="The views around the specified interval directly show whether the object is at its original position.",
        ),
    ))

    # Object tracking: retain stable ground truth but describe only interpretable cues.
    tracking_disappearance, tracking_reappearance = tracking_pair
    tracking_surface = _surface(tracking_disappearance)
    tracking_subject = tracking_disappearance["subject"]
    tracking_correct = f"The same {tracking_subject} that disappeared earlier is visible again"
    tracking_prompts = (
        f"The {tracking_subject} later visible again on the {tracking_surface} "
        "corresponds to which option?",
        f"The {tracking_subject} disappears from the {tracking_surface} and later "
        "returns. How are the two observations related?",
        f"Comparing {_time_range(tracking_disappearance)} with "
        f"{_time_range(tracking_reappearance)}, how is the returning "
        f"{tracking_subject} related to the one that disappeared?",
    )
    result.append(_question(
        episode_id,
        "object_tracking",
        "reappearing_instance_identity",
        tracking_reappearance["end_sec"],
        f"At {_time_range(tracking_disappearance)}, the {tracking_subject} disappears "
        f"from its original {tracking_surface} position; at "
        f"{_time_range(tracking_reappearance)}, an object of the same category "
        "returns to that position, with no other matching object entering it in between.",
        _pick_text(episode_id, "tracking-prompt", tracking_prompts),
        tracking_correct,
        [
            "A different object of the same category appears at the original position",
            f"The {tracking_subject} is moved to another position in the room",
            "Cannot be determined from the video up to this point.",
        ],
        [_span(tracking_disappearance), _span(tracking_reappearance)],
        _diagnostics(
            volatility="The object disappears and reappears while both events continue to reference the same target.",
            uncertainty="The object category, support, and position cues match across the two events.",
        ),
    ))

    # Temporal reasoning: alternate between earliest disappearance/reappearance.
    temporal_appearances = bool(_binary_variant(episode_id, "temporal-direction"))
    temporal_events = (appearances if temporal_appearances else disappearances)[:4]
    if len(temporal_events) < 4:
        raise ValueError("at least four events are required for temporal reasoning")
    if temporal_appearances:
        temporal_prompts = (
            "Several objects return to view. Which listed object reappears first?",
            "Comparing when the four listed objects reappear, which returns to its original position first?",
            f"By {_time_point(temporal_events[-1]['end_sec'])}, which listed object "
            "has reappeared first?",
        )
        temporal_evidence = "; ".join(
            f"the {item['subject']} reappears at {_time_range(item)}"
            for item in temporal_events
        )
    else:
        temporal_prompts = (
            "Comparing when the listed objects leave their original positions, which disappears first?",
            "Which of the four listed objects is the first to disappear from view?",
            f"By {_time_point(temporal_events[-1]['end_sec'])}, which listed object "
            "has disappeared from its original position first?",
        )
        temporal_evidence = "; ".join(
            f"the {item['subject']} disappears from its original position at {_time_range(item)}"
            for item in temporal_events
        )
    result.append(_question(
        episode_id,
        "temporal_reasoning",
        "earliest_visibility_change_identification",
        temporal_events[-1]["end_sec"],
        temporal_evidence + ".",
        _pick_text(episode_id, "temporal-prompt", temporal_prompts),
        temporal_events[0]["subject"],
        [item["subject"] for item in temporal_events[1:]],
        [_span(item) for item in temporal_events],
        _diagnostics(
            volatility="The four change intervals do not overlap, so later views do not alter which event occurred first.",
            uncertainty="Each object's change has a corresponding interval that directly establishes the order.",
        ),
    ))

    # History aggregation: alternate between disappeared and restored object counts.
    aggregate_appearances = bool(_binary_variant(episode_id, "aggregate-direction"))
    aggregate_source = appearances if aggregate_appearances else disappearances
    aggregate_cutoff = min(
        len(aggregate_source),
        3 + _stable_index(episode_id, "aggregate-cutoff", 3),
    )
    aggregate_events = aggregate_source[:aggregate_cutoff]
    distinct_count = len({item["_target_id"] for item in aggregate_events})
    aggregate_query = aggregate_events[-1]["end_sec"]
    if aggregate_appearances:
        aggregate_prompts = (
            f"From the first reappearance through {_time_point(aggregate_query)}, "
            "how many distinct objects return to their original positions?",
            f"By {_time_point(aggregate_query)}, how many distinct objects have "
            "reappeared in the video?",
            "Objects return to their original positions later in the video. How many "
            "distinct objects have returned by the final reappearance?",
        )
        aggregate_evidence = "; ".join(
            f"the {item['subject']} returns to its original position at {_time_range(item)}"
            for item in aggregate_events
        )
    else:
        aggregate_prompts = (
            f"From the start of the video through {_time_point(aggregate_query)}, "
            "how many distinct objects disappear from their original positions?",
            f"By {_time_point(aggregate_query)}, how many distinct objects have "
            "disappeared from view?",
            "Objects disappear from their original positions earlier in the video. "
            "How many distinct objects have disappeared by the final event?",
        )
        aggregate_evidence = "; ".join(
            f"the {item['subject']} disappears from its original position at {_time_range(item)}"
            for item in aggregate_events
        )
    result.append(_question(
        episode_id,
        "history_aggregation",
        "distinct_visibility_change_count",
        aggregate_query,
        aggregate_evidence + ".",
        _pick_text(episode_id, "aggregate-prompt", aggregate_prompts),
        f"{distinct_count} distinct objects",
        _numeric_options(distinct_count),
        [_span(item) for item in aggregate_events],
        _diagnostics(
            volatility="The question defines the counting range, and each object is counted once.",
            uncertainty="Every included change is complete in the video and the counting boundary is explicit.",
        ),
    ))

    if {item["question_type"] for item in result} != set(QUESTION_TYPES):
        raise AssertionError("question generator did not cover all shared types")
    for index, question in enumerate(result, start=1):
        question["id"] = f"{episode_id}_q{index:02d}"
        if question["query_time"] > duration + 0.05:
            raise ValueError("generated question lies outside the video")
    return result
