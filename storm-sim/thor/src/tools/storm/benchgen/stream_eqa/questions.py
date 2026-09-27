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
    return f"约{int(round(float(seconds)))}秒"


def _time_range(event: dict[str, Any]) -> str:
    return f"{float(event['start_sec']):.1f}—{float(event['end_sec']):.1f}秒"


def _surface(event: dict[str, Any]) -> str:
    surface = event.get("_surface")
    if isinstance(surface, str) and surface:
        return surface
    location = str(event.get("location", "原来的位置"))
    if "场景的" in location:
        location = location.split("场景的", 1)[1]
    return location.removesuffix("上") or "原来的位置"


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
            f"在{_time_range(event)}，镜头重新对准{surface}时，原本放在那里的"
            f"{subject}已经不见了。"
        )
    return (
        f"在{_time_range(event)}，镜头回到{surface}后，先前不见的{subject}"
        "重新出现在原来的位置。"
    )


def _numeric_options(correct_count: int) -> list[str]:
    distractors = []
    for value in (
        max(0, correct_count - 1), correct_count + 1,
        max(0, correct_count - 2), correct_count + 2, 0,
    ):
        option = f"共有{value}件不同的物体"
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
            f"镜头在{_time_range(factual_event)}重新对准{factual_surface}时，"
            "原本放在那里的哪件物体不见了？",
            f"视频播放到{_time_point(factual_event['end_sec'])}时，"
            f"{factual_surface}上少了哪件物体？",
            f"观察{_time_range(factual_event)}这段画面，"
            f"哪件物体从{factual_surface}的原位置消失了？",
        )
    else:
        factual_prompts = (
            f"在{_time_range(factual_event)}，哪件物体重新出现在"
            f"{factual_surface}的原位置？",
            f"视频播放到{_time_point(factual_event['end_sec'])}时，"
            f"哪件物体又回到了{factual_surface}上？",
            f"镜头在{_time_range(factual_event)}回到{factual_surface}后，"
            "重新看到的是哪件物体？",
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
            volatility="问题询问一段已经完成的画面变化，后续事件不会改变该段事实。",
            uncertainty="变化前后的画面能够确认物体类别及其所在位置。",
        ),
    ))

    # Current state: ask both hidden and restored states across a batch.
    current_disappearance, current_reappearance = current_pair
    current_surface = _surface(current_disappearance)
    current_subject = current_disappearance["subject"]
    initial_option = f"它一直留在{current_surface}的原位置"
    hidden_option = "它已经从原位置消失，此时还没有重新出现"
    returned_option = f"它已经重新出现在{current_surface}的原位置"
    moved_option = "它被移到了场景中的另一个位置"
    ask_restored = bool(_binary_variant(episode_id, "current-state"))
    if ask_restored:
        current_query = float(current_reappearance["end_sec"])
        current_correct = returned_option
        current_distractors = [hidden_option, initial_option, moved_option]
        current_evidence = [_span(current_disappearance), _span(current_reappearance)]
        current_video_evidence = (
            f"{current_subject}先在{_time_range(current_disappearance)}从"
            f"{current_surface}消失，随后在{_time_range(current_reappearance)}"
            "回到原来的位置。"
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
            f"{current_subject}在{_time_range(current_disappearance)}从"
            f"{current_surface}消失；直到{_time_point(current_reappearance['start_sec'])}"
            "之后，它才重新出现在画面中。"
        )
        revision_count = 1
        old_option = initial_option
    current_prompts = (
        f"视频播放到{_time_point(current_query)}时，先前位于{current_surface}上的"
        f"{current_subject}是什么状态？",
        f"看到{_time_point(current_query)}，{current_surface}上的"
        f"{current_subject}现在在哪里？",
        f"视频来到{_time_point(current_query)}时，画面中的{current_subject}"
        "处于下面哪种状态？",
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
            volatility="答案随物体消失和重新出现而更新，查询时刻对应的状态已经确定。",
            uncertainty="查询时刻之前的相关变化都有清楚的前后画面作为依据。",
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
    disappeared = "它从原来的位置消失了"
    reappeared = "它重新出现在原来的位置"
    unchanged = f"它一直留在{state_surface}上，没有变化"
    moved = "它被移到了场景中的另一个位置"
    if state_event["event_type"] == "disappearance":
        state_correct = disappeared
        state_distractors = [reappeared, unchanged, moved]
    else:
        state_correct = reappeared
        state_distractors = [disappeared, unchanged, moved]
    state_prompts = (
        f"在{_time_range(state_event)}这段画面里，{state_surface}上的"
        f"{state_subject}发生了什么变化？",
        f"比较{_time_range(state_event)}前后的画面，{state_subject}"
        "的状态发生了怎样的变化？",
        f"镜头在{_time_range(state_event)}再次对准{state_surface}时，"
        f"{state_subject}出现了哪种变化？",
    )
    result.append(_question(
        episode_id,
        "state_change",
        "visibility_state_transition",
        state_event["end_sec"],
        f"变化前，{state_event['before_state']}；变化后，{state_event['after_state']}。",
        _pick_text(episode_id, "state-change-prompt", state_prompts),
        state_correct,
        state_distractors,
        [_span(state_event)],
        _diagnostics(
            revision_count=1,
            old_answer_indices=[],
            volatility="问题明确指定了一段前后状态不同的画面，答案对应这一次变化。",
            uncertainty="指定时间段的前后画面能够直接判断物体是否位于原位置。",
        ),
    ))

    # Object tracking: retain stable ground truth but describe only interpretable cues.
    tracking_disappearance, tracking_reappearance = tracking_pair
    tracking_surface = _surface(tracking_disappearance)
    tracking_subject = tracking_disappearance["subject"]
    tracking_correct = f"是先前消失的那个{tracking_subject}重新出现了"
    tracking_prompts = (
        f"后半段重新出现在{tracking_surface}上的{tracking_subject}，"
        "是之前消失的那个吗？",
        f"先前从{tracking_surface}消失的{tracking_subject}后来又回来了。"
        "前后画面中的它是什么关系？",
        f"比较{_time_range(tracking_disappearance)}和"
        f"{_time_range(tracking_reappearance)}两段画面，重新出现的"
        f"{tracking_subject}与先前消失的{tracking_subject}是什么关系？",
    )
    result.append(_question(
        episode_id,
        "object_tracking",
        "reappearing_instance_identity",
        tracking_reappearance["end_sec"],
        f"{_time_range(tracking_disappearance)}，{tracking_surface}原位置的"
        f"{tracking_subject}消失；{_time_range(tracking_reappearance)}，"
        "相同类别的物体回到同一位置，中间没有另一件同类物体进入该位置。",
        _pick_text(episode_id, "tracking-prompt", tracking_prompts),
        tracking_correct,
        [
            "是另一件同类物体出现在了原位置",
            f"是{tracking_subject}被移到了场景中的另一处",
            "只能看出位置相同，无法判断前后是否有关联",
        ],
        [_span(tracking_disappearance), _span(tracking_reappearance)],
        _diagnostics(
            volatility="物体经历了消失和重新出现，但前后事件始终指向同一个目标。",
            uncertainty="前后事件的物体类别、承载面和位置线索一致，生成记录也确认属于同一目标。",
        ),
    ))

    # Temporal reasoning: alternate between earliest disappearance/reappearance.
    temporal_appearances = bool(_binary_variant(episode_id, "temporal-direction"))
    temporal_events = (appearances if temporal_appearances else disappearances)[:4]
    if len(temporal_events) < 4:
        raise ValueError("at least four events are required for temporal reasoning")
    if temporal_appearances:
        temporal_prompts = (
            "后半段几件物体陆续回到画面中，下面哪一件最早重新出现？",
            "比较下面四件物体重新出现的时间，哪一件最先回到原来的位置？",
            f"视频播放到{_time_point(temporal_events[-1]['end_sec'])}时，"
            "下面哪件物体最早重新出现在画面中？",
        )
        temporal_evidence = "；".join(
            f"{item['subject']}在{_time_range(item)}重新出现"
            for item in temporal_events
        )
    else:
        temporal_prompts = (
            "比较前半段几件物体从原位置消失的时间，下面哪一件最早消失？",
            "下面四件物体中，哪一件最先从画面中不见？",
            f"视频播放到{_time_point(temporal_events[-1]['end_sec'])}时，"
            "下面哪件物体最早从原来的位置消失？",
        )
        temporal_evidence = "；".join(
            f"{item['subject']}在{_time_range(item)}从原位置消失"
            for item in temporal_events
        )
    result.append(_question(
        episode_id,
        "temporal_reasoning",
        "earliest_visibility_change_identification",
        temporal_events[-1]["end_sec"],
        temporal_evidence + "。",
        _pick_text(episode_id, "temporal-prompt", temporal_prompts),
        temporal_events[0]["subject"],
        [item["subject"] for item in temporal_events[1:]],
        [_span(item) for item in temporal_events],
        _diagnostics(
            volatility="四次变化的时间段互不重叠，最早发生的事件不会因后续画面而改变。",
            uncertainty="每件物体的变化都有对应时间段，可以直接比较先后顺序。",
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
            f"从物体开始重新出现到{_time_point(aggregate_query)}，"
            "一共有多少件不同的物体回到了原来的位置？",
            f"视频播放到{_time_point(aggregate_query)}时，先后重新出现在画面中的"
            "不同物体共有多少件？",
            "后半段物体陆续回到原位。到最后一次重新出现时，"
            "一共有多少件不同的物体回来过？",
        )
        aggregate_evidence = "；".join(
            f"{item['subject']}在{_time_range(item)}回到原来的位置"
            for item in aggregate_events
        )
    else:
        aggregate_prompts = (
            f"从视频开始到{_time_point(aggregate_query)}，"
            "一共有多少件不同的物体从原位置消失？",
            f"视频播放到{_time_point(aggregate_query)}时，画面中先后不见的"
            "不同物体共有多少件？",
            "前半段物体陆续从原位置消失。到最后一次消失为止，"
            "一共涉及多少件不同的物体？",
        )
        aggregate_evidence = "；".join(
            f"{item['subject']}在{_time_range(item)}从原位置消失"
            for item in aggregate_events
        )
    result.append(_question(
        episode_id,
        "history_aggregation",
        "distinct_visibility_change_count",
        aggregate_query,
        aggregate_evidence + "。",
        _pick_text(episode_id, "aggregate-prompt", aggregate_prompts),
        f"共有{distinct_count}件不同的物体",
        _numeric_options(distinct_count),
        [_span(item) for item in aggregate_events],
        _diagnostics(
            volatility="统计范围在问题中已经明确，同一物体只计数一次。",
            uncertainty="纳入统计的变化都已在画面中完整发生，计数边界清楚。",
        ),
    ))

    if {item["question_type"] for item in result} != set(QUESTION_TYPES):
        raise AssertionError("question generator did not cover all shared types")
    for index, question in enumerate(result, start=1):
        question["id"] = f"{episode_id}_q{index:02d}"
        if question["query_time"] > duration + 0.05:
            raise ValueError("generated question lies outside the video")
    return result
