"""Natural questions grounded only in one exported key clip."""
from __future__ import annotations

from typing import Any

from .questions import (
    _diagnostics,
    _event_evidence,
    _numeric_options,
    _pick_text,
    _question,
    _span,
    _surface,
    _time_point,
    _time_range,
)


def _other_labels(subject: str, labels: list[str]) -> list[str]:
    others = [label for label in labels if label != subject]
    if len(others) < 3:
        raise ValueError("key-clip questions require at least four object labels")
    return others


def _state_options(event: dict[str, Any]) -> tuple[str, list[str], str]:
    subject = event["subject"]
    surface = _surface(event)
    visible = f"The {subject} is visible at its original {surface} position"
    missing = f"The {subject} is absent from its original {surface} position"
    moved = f"The {subject} is visible at another position in the room"
    unknown = "Cannot be determined from the video up to this point."
    if event["event_type"] == "disappearance":
        return missing, [visible, moved, unknown], visible
    return visible, [missing, moved, unknown], missing


def _transition_options(event: dict[str, Any]) -> tuple[str, list[str]]:
    surface = _surface(event)
    disappeared = "It changes from visible to absent at its original position"
    appeared = "It changes from absent to visible at its original position"
    unchanged = f"It remains visible on the {surface}"
    moved = "It becomes visible at another position in the room"
    if event["event_type"] == "disappearance":
        return disappeared, [appeared, unchanged, moved]
    return appeared, [disappeared, unchanged, moved]


def _sequence_options(event: dict[str, Any]) -> tuple[str, list[str]]:
    subject = event["subject"]
    surface = _surface(event)
    if event["event_type"] == "disappearance":
        correct = f"The {subject} is initially visible at its original position and is missing when the view returns"
        wrong = [
            f"The {subject} is initially absent and then reappears at its original position",
            f"The {subject} remains visible on the {surface} throughout the clip",
            f"The {subject} first appears elsewhere and then moves to the {surface}",
        ]
    else:
        correct = f"The {subject} is initially absent and then reappears at its original position"
        wrong = [
            f"The {subject} is initially visible at its original position and is missing when the view returns",
            f"The {subject} remains visible on the {surface} throughout the clip",
            f"The {subject} first appears elsewhere and then moves to the {surface}",
        ]
    return correct, wrong


def build_clip_questions(
    episode_id: str,
    events: list[dict[str, Any]],
    duration: float,
    all_labels: list[str],
    visit_index: int,
) -> list[dict[str, Any]]:
    """Build 5–7 questions whose evidence is fully contained in one clip."""
    ordered = sorted(events, key=lambda item: (item["start_sec"], item["end_sec"]))
    if not 1 <= len(ordered) <= 2:
        raise ValueError("a key clip must contain one or two events")
    result: list[dict[str, Any]] = []

    # One factual question per event, matching the reference's retrieval density.
    for event_number, event in enumerate(ordered, start=1):
        subject = event["subject"]
        surface = _surface(event)
        if event["event_type"] == "disappearance":
            if len(ordered) == 2:
                prompts = (
                    f"During the change at {_time_range(event)}, which object is no longer "
                    f"at its original {surface} position?",
                    f"In the views at {_time_range(event)}, which object is missing from the {surface}?",
                    f"When the view returns to the {surface} at {_time_range(event)}, "
                    "which object has disappeared from its original position?",
                )
            else:
                prompts = (
                    f"In this clip, which object is missing from its original position "
                    f"when the view returns to the {surface}?",
                    f"In the views at {_time_range(event)}, which object is missing from the {surface}?",
                    f"When the view returns to the {surface}, which object has disappeared "
                    "from its original position?",
                )
        else:
            if len(ordered) == 2:
                prompts = (
                    f"During the change at {_time_range(event)}, which object reappears "
                    f"at its original {surface} position?",
                    f"In the views at {_time_range(event)}, which object returns to the {surface}?",
                    f"When the view returns to the {surface} at {_time_range(event)}, "
                    "which object is visible again at its original position?",
                )
            else:
                prompts = (
                    f"In this clip, which object reappears when the view returns to the {surface}?",
                    f"In the views at {_time_range(event)}, which object returns to the {surface}?",
                    f"When the view returns to the {surface}, which object is visible "
                    "again at its original position?",
                )
        result.append(_question(
            episode_id,
            "factual_retrieval",
            "clip_changed_object_retrieval",
            event["end_sec"],
            _event_evidence(event),
            _pick_text(
                episode_id,
                f"clip-factual-{event_number}",
                prompts,
            ),
            subject,
            _other_labels(subject, all_labels),
            [_span(event)],
            _diagnostics(
                volatility="The question asks about one completed visible change in the current clip.",
                uncertainty="The views before and after the change establish the object category and location.",
            ),
            answer_key=f"clip-factual-{event_number}",
        ))

    # Current state at the end of the clip.
    current_event = ordered[-1]
    current_correct, current_wrong, previous_state = _state_options(current_event)
    current = _question(
        episode_id,
        "current_state",
        "clip_end_object_state",
        duration,
        _event_evidence(current_event),
        f"What is the visible state of the {current_event['subject']} at the end of this clip?",
        current_correct,
        current_wrong,
        [_span(current_event)],
        _diagnostics(
            revision_count=1,
            old_answer_indices=[],
            volatility="The object changes once in the clip, fixing its state at the end.",
            uncertainty="The before-and-after views show whether the object is at its original position.",
        ),
        answer_key="clip-current-state",
    )
    current["diagnostics"]["old_answer_indices"] = [
        current["options"].index(previous_state)
    ]
    result.append(current)

    # State change for the first event, with an explicit clip-local time span.
    change_event = ordered[0]
    transition_correct, transition_wrong = _transition_options(change_event)
    result.append(_question(
        episode_id,
        "state_change",
        "clip_visibility_state_transition",
        change_event["end_sec"],
        f"Before the change, {change_event['before_state']}; after the change, "
        f"{change_event['after_state']}.",
        f"What visible change occurs to the {change_event['subject']} during "
        f"{_time_range(change_event)}?",
        transition_correct,
        transition_wrong,
        [_span(change_event)],
        _diagnostics(
            revision_count=1,
            old_answer_indices=[],
            volatility="The question identifies one state change in the current clip.",
            uncertainty="The views around the specified interval directly establish the direction of change.",
        ),
        answer_key="clip-state-change",
    ))

    # Temporal reasoning compares two events, or asks the order inside one event.
    if len(ordered) == 2:
        first, second = ordered
        temporal_prompt = _pick_text(
            episode_id,
            "clip-temporal-two-events",
            (
                "Two objects change in this clip. Which one changes first?",
                "Which listed object changes first in the clip?",
                f"By {_time_point(second['end_sec'])}, which object's change occurs first?",
            ),
        )
        temporal_correct = first["subject"]
        temporal_wrong = [second["subject"], *[
            label for label in all_labels
            if label not in {first["subject"], second["subject"]}
        ]]
        temporal_evidence = (
            f"The {first['subject']} changes at {_time_range(first)}; "
            f"the {second['subject']} changes at {_time_range(second)}."
        )
        temporal_spans = [_span(first), _span(second)]
        temporal_query = second["end_sec"]
        temporal_subtype = "clip_event_order"
    else:
        event = ordered[0]
        temporal_correct, temporal_wrong = _sequence_options(event)
        temporal_prompt = (
            f"Which option correctly describes the order of the {event['subject']}'s "
            "visible states in this clip?"
        )
        temporal_evidence = _event_evidence(event)
        temporal_spans = [_span(event)]
        temporal_query = event["end_sec"]
        temporal_subtype = "clip_before_after_order"
    result.append(_question(
        episode_id,
        "temporal_reasoning",
        temporal_subtype,
        temporal_query,
        temporal_evidence,
        temporal_prompt,
        temporal_correct,
        temporal_wrong,
        temporal_spans,
        _diagnostics(
            volatility="The complete order being asked about is shown within the current clip.",
            uncertainty="The relevant changes occupy nonoverlapping intervals within the clip.",
        ),
        answer_key="clip-temporal",
    ))

    # Count only the objects whose key events are actually in this clip.
    distinct_count = len({event["_target_id"] for event in ordered})
    result.append(_question(
        episode_id,
        "history_aggregation",
        "clip_changed_object_count",
        duration,
        "; ".join(_event_evidence(event).removesuffix(".") for event in ordered)
        + ".",
        "How many distinct objects disappear or reappear in this clip?",
        f"{distinct_count} distinct objects",
        _numeric_options(distinct_count),
        [_span(event) for event in ordered],
        _diagnostics(
            volatility="The count is limited to the current clip, and each object is counted once.",
            uncertainty="Every counted change has clear before-and-after views in the clip.",
        ),
        answer_key="clip-history-count",
    ))

    # Reference data uses object tracking more selectively, so emit it on odd visits.
    if visit_index % 2 == 1:
        tracking_event = ordered[-1]
        tracking_subject = tracking_event["subject"]
        tracking_surface = _surface(tracking_event)
        if tracking_event["event_type"] == "disappearance":
            tracking_prompt = (
                f"After the view moves away and returns, which option describes the "
                f"{tracking_subject} that was on the {tracking_surface}?"
            )
            tracking_correct = "It is no longer at its original position"
            tracking_wrong = [
                "It remains at its original position",
                "It is visible elsewhere in the frame",
                "A different object of the same category appears at the original position",
            ]
        else:
            tracking_prompt = (
                f"When the view returns to the {tracking_surface}, which option describes "
                f"the previously missing {tracking_subject}?"
            )
            tracking_correct = "It is visible again at its original position"
            tracking_wrong = [
                "It is still absent from the frame",
                "It appears elsewhere in the frame",
                "A different object of the same category appears at the original position",
            ]
        result.append(_question(
            episode_id,
            "object_tracking",
            "clip_object_continuity",
            tracking_event["end_sec"],
            _event_evidence(tracking_event),
            tracking_prompt,
            tracking_correct,
            tracking_wrong,
            [_span(tracking_event)],
            _diagnostics(
                volatility="The target's visibility at its original position changes when the view returns.",
                uncertainty="The object category, support, and original position are consistent across the clip.",
            ),
            answer_key="clip-object-tracking",
        ))

    for index, question in enumerate(result, start=1):
        question["id"] = f"{episode_id}_q{index:02d}"
        if question["query_time"] > duration + 0.05:
            raise ValueError("generated clip question lies outside the video")
    prompts = [question["question"] for question in result]
    if len(prompts) != len(set(prompts)):
        raise ValueError("generated key-clip questions are not textually unique")
    return result
