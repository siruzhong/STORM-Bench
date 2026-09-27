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
    visible = f"{subject}位于{surface}的原位置，画面中可以看到"
    missing = f"{subject}已经不在{surface}的原位置"
    moved = f"{subject}被移到了场景中的另一个位置"
    unknown = f"片段结束时无法判断{subject}的位置"
    if event["event_type"] == "disappearance":
        return missing, [visible, moved, unknown], visible
    return visible, [missing, moved, unknown], missing


def _transition_options(event: dict[str, Any]) -> tuple[str, list[str]]:
    surface = _surface(event)
    disappeared = "它从原来的位置消失了"
    appeared = "它重新出现在原来的位置"
    unchanged = f"它一直留在{surface}上，没有变化"
    moved = "它被移到了场景中的另一个位置"
    if event["event_type"] == "disappearance":
        return disappeared, [appeared, unchanged, moved]
    return appeared, [disappeared, unchanged, moved]


def _sequence_options(event: dict[str, Any]) -> tuple[str, list[str]]:
    subject = event["subject"]
    surface = _surface(event)
    if event["event_type"] == "disappearance":
        correct = f"先看到{subject}在原位置，镜头返回后它已经不见"
        wrong = [
            f"先看不到{subject}，随后它重新出现在原位置",
            f"{subject}始终位于{surface}上，没有发生变化",
            f"{subject}先出现在另一处，随后被移到{surface}",
        ]
    else:
        correct = f"先看不到{subject}，随后它重新出现在原位置"
        wrong = [
            f"先看到{subject}在原位置，镜头返回后它已经不见",
            f"{subject}始终位于{surface}上，没有发生变化",
            f"{subject}先出现在另一处，随后被移到{surface}",
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
                    f"在{_time_range(event)}这段变化中，哪件物体已经不在"
                    f"{surface}的原位置？",
                    f"观察{_time_range(event)}这段画面，{surface}上少了哪件物体？",
                    f"视频{_time_range(event)}再次对准{surface}后，"
                    "哪件物体从原位置消失了？",
                )
            else:
                prompts = (
                    f"这个片段中，镜头回到{surface}时，哪件物体已经不在原来的位置？",
                    f"观察{_time_range(event)}这段画面，{surface}上少了哪件物体？",
                    f"画面再次对准{surface}后，哪件物体从原位置消失了？",
                )
        else:
            if len(ordered) == 2:
                prompts = (
                    f"在{_time_range(event)}这段变化中，哪件物体重新出现在"
                    f"{surface}的原位置？",
                    f"观察{_time_range(event)}这段画面，哪件物体回到了{surface}？",
                    f"视频{_time_range(event)}回到{surface}时，"
                    "重新出现在原位置的是哪件物体？",
                )
            else:
                prompts = (
                    f"这个片段中，镜头再次对准{surface}后，哪件物体重新出现了？",
                    f"观察{_time_range(event)}这段画面，哪件物体回到了{surface}？",
                    f"画面回到{surface}时，重新出现在原位置的是哪件物体？",
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
                volatility="问题只询问当前片段中已经完成的一次画面变化。",
                uncertainty="变化前后的画面能够确认物体类别和所在位置。",
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
        f"截至这个片段结束，{current_event['subject']}处于什么状态？",
        current_correct,
        current_wrong,
        [_span(current_event)],
        _diagnostics(
            revision_count=1,
            old_answer_indices=[],
            volatility="物体在片段中发生了一次变化，片段结束时的状态已经确定。",
            uncertainty="片段中的前后画面足以判断物体是否位于原位置。",
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
        f"变化前，{change_event['before_state']}；变化后，"
        f"{change_event['after_state']}。",
        f"在{_time_range(change_event)}这段画面里，"
        f"{change_event['subject']}发生了什么变化？",
        transition_correct,
        transition_wrong,
        [_span(change_event)],
        _diagnostics(
            revision_count=1,
            old_answer_indices=[],
            volatility="问题明确指定当前片段中的一次状态变化。",
            uncertainty="指定时间段的前后画面能够直接判断变化方向。",
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
                "这个片段里有两件物体先后发生变化，哪一件更早？",
                "按照片段中的发生顺序，下面哪件物体最先发生变化？",
                f"截至{_time_point(second['end_sec'])}，哪件物体的变化最早发生？",
            ),
        )
        temporal_correct = first["subject"]
        temporal_wrong = [second["subject"], *[
            label for label in all_labels
            if label not in {first["subject"], second["subject"]}
        ]]
        temporal_evidence = (
            f"{first['subject']}的变化发生在{_time_range(first)}；"
            f"{second['subject']}的变化发生在{_time_range(second)}。"
        )
        temporal_spans = [_span(first), _span(second)]
        temporal_query = second["end_sec"]
        temporal_subtype = "clip_event_order"
    else:
        event = ordered[0]
        temporal_correct, temporal_wrong = _sequence_options(event)
        temporal_prompt = (
            f"关于{event['subject']}在这个片段中的变化，下面哪项顺序正确？"
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
            volatility="问题涉及的先后顺序在当前片段内完整呈现。",
            uncertainty="相关变化具有互不冲突的片段内时间区间。",
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
        "；".join(_event_evidence(event).removesuffix("。") for event in ordered)
        + "。",
        "这个片段里，一共有多少件不同的物体消失或重新出现？",
        f"共有{distinct_count}件不同的物体",
        _numeric_options(distinct_count),
        [_span(event) for event in ordered],
        _diagnostics(
            volatility="计数范围仅限当前片段，同一物体只计算一次。",
            uncertainty="片段内纳入计数的变化都有明确的前后画面。",
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
                f"镜头移开再回来后，先前位于{tracking_surface}的"
                f"{tracking_subject}符合下面哪种情况？"
            )
            tracking_correct = "它已经不在原来的位置"
            tracking_wrong = [
                "它仍然留在原来的位置",
                "它被移到了画面中的另一处",
                "原位置出现了另一件同类物体",
            ]
        else:
            tracking_prompt = (
                f"镜头重新对准{tracking_surface}时，先前不见的"
                f"{tracking_subject}符合下面哪种情况？"
            )
            tracking_correct = "它已经重新出现在原来的位置"
            tracking_wrong = [
                "它仍然没有出现在画面中",
                "它出现在了画面中的另一处",
                "原位置出现了另一件同类物体",
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
                volatility="镜头移动前后，目标物体在原位置的可见状态发生了变化。",
                uncertainty="物体类别、承载面和原位置在片段前后保持一致。",
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
