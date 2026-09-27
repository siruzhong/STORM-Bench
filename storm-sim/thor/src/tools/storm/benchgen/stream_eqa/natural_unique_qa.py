"""Build v2.4 questions with relative event references and distinct wording.

Public questions use event references instead of numeric timestamps.
query_time and evidence_spans retain their original values.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from . import scene_grounded_qa as exp2


base = exp2.base

VERSION = "natural_unique_v2_4_exp1"
ID_VERSION_TAG = "v24e1"
QA_SOURCE = "simulation_ground_truth_natural_unique_v2_4_exp1"
REPORT_FILENAME = "QA_v2.4实验1_无秒数与自然去重报告_中文.md"
DEFAULT_SEED = 20260825

SUBTYPES = dict(exp2.SUBTYPES)
SUBTYPES.update({
    key: f"{value}_natural_unique_v24e1"
    for key, value in exp2.SUBTYPES.items()
})


CURRENT_LEADS = (
    "Looking at the agent's present surroundings,",
    "From the environment visible around the agent,",
    "Based on the room shown from the agent's viewpoint,",
    "Considering the agent's current surroundings,",
    "Taking in the space visible around the agent,",
    "Judging from the room layout now in view,",
    "From the setting currently surrounding the agent,",
    "Based on the environment in the agent's present view,",
)
CURRENT_TAILS = (
    "what type of room is this?",
    "which room category best describes this space?",
    "what kind of room is currently visible?",
    "which type of indoor space is being shown?",
    "how should the visible room be classified?",
    "which room type matches the surroundings?",
    "what room category fits the current view?",
)

FACTUAL_LEADS = (
    "Considering the setting throughout the activity,",
    "From the environment shown across the video,",
    "Based on the room where the activity unfolds,",
    "Looking at the setting shared by the whole sequence,",
    "Taking the complete activity into account,",
    "Judging from the persistent surroundings in the video,",
    "From the indoor setting visible during the activity,",
    "Based on the environment that contains the full sequence,",
)
FACTUAL_TAILS = (
    "which type of room contains the scene?",
    "what room category is the activity set in?",
    "which kind of room serves as the setting?",
    "how should the room in this activity be classified?",
    "what type of indoor space is shown?",
    "which room type best matches the setting?",
    "what category describes the room in the video?",
)

HISTORY_LEADS = (
    "Considering the final pair of return events available at the query point,",
    "Within the two consecutive returns that conclude the visible prefix,",
    "Looking across the last pair of objects to come back into view,",
    "Among the two return events nearest the end of the visible sequence,",
    "Using the final two reappearances observed before the question point,",
    "Focusing on the concluding pair of returns in the visible history,",
    "Across the two latest objects to become visible again,",
)
HISTORY_TAILS = (
    "which listed object returns later?",
    "which object is the later one to reappear?",
    "which listed item comes back into view last?",
    "which object completes the pair of returns?",
    "which listed item is restored to view after the other?",
    "which object occupies the later position in that return order?",
)

STATE_LEADS = (
    "At the current point in the visible sequence,",
    "In the latest visibility change shown before the query,",
    "During the most recent observable transition,",
    "Looking at the visibility change that ends the current prefix,",
    "For the last object-state transition now available,",
    "At the transition immediately preceding the question point,",
    "In the newest change of visibility shown so far,",
    "Considering the final transition in the available visual prefix,",
)
STATE_DISAPPEAR_TAILS = (
    "which listed object changes from visible to absent?",
    "which object is the one that leaves view?",
    "which listed item is no longer visible after the change?",
    "which object completes the visible-to-absent transition?",
    "which listed item disappears in that transition?",
    "which object is removed from view by this change?",
)
STATE_APPEAR_TAILS = (
    "which listed object changes from absent to visible?",
    "which object is the one that returns to view?",
    "which listed item becomes visible after the change?",
    "which object completes the absent-to-visible transition?",
    "which listed item reappears in that transition?",
    "which object is restored to view by this change?",
)

TRACK_LEADS = (
    "At the current query point,",
    "At the end of the visible prefix,",
    "In the return event that has just occurred,",
    "Looking at the latest reappearance shown so far,",
    "For the object that has just come back into view,",
    "In the most recent completed return event,",
)
TRACK_TAILS = (
    "which listed object has returned after disappearing earlier?",
    "which object completes its disappear-and-return track?",
    "which listed item has become visible again after an earlier absence?",
    "which object is being tracked back into view?",
    "which listed item finishes the return part of its track?",
)

TEMPORAL_LEADS = (
    "Within the final pair of consecutive return events,",
    "Looking at the two returns that end the visible prefix,",
    "In the concluding two-event return sequence,",
    "Across the latest pair of objects to reappear,",
    "Considering the two most recent return events shown so far,",
    "For the pair of reappearances immediately before the query point,",
)
TEMPORAL_TAILS = (
    "which listed object follows the other into view?",
    "which object appears later in the order?",
    "which listed item completes the sequence?",
    "which object returns after the earlier event?",
    "which listed item occupies the later place in the return order?",
)


_ORIGINAL_BUILD_QUESTION = base._build_question
_ORIGINAL_AUDIT = base._audit_questions
_EXP2_REPORT = exp2._report_markdown


def _combine(leads: tuple[str, ...], tails: tuple[str, ...], ordinal: int) -> str:
    capacity = len(leads) * len(tails)
    if ordinal < 0 or ordinal >= capacity:
        raise base.ExportError(f"natural-language variant capacity exceeded: {ordinal} >= {capacity}")
    return f"{leads[ordinal % len(leads)]} {tails[ordinal // len(leads)]}"


def _extract(pattern: str, text: str) -> str:
    match = re.search(pattern, text)
    if not match:
        raise base.ExportError(f"cannot recover natural-language anchor from: {text}")
    return match.group(1)


def _surface(question: dict[str, Any]) -> str:
    for option in question["_distractors"]:
        match = re.fullmatch(r"At its original (.+) position", option)
        if match:
            return match.group(1)
    raise base.ExportError("location question lacks its original support option")


def _natural_location(question: dict[str, Any], ordinal: int) -> str:
    label = _extract(r"^Where is the (.+?) at ", question["question"])
    surface = _surface(question)
    if question["question_type"] == "current_state":
        leads = (
            f"After the {label} leaves its original {surface} position,",
            f"While the {label} remains away from its original {surface} position,",
            f"Once the {label} is no longer visible at its original {surface} position,",
            f"During the gap between the {label} leaving and returning to its original {surface} position,",
        )
        tails = (
            "where is it located?",
            "which location description applies to it?",
            "where can the object be found?",
            "which listed location matches its whereabouts?",
        )
    else:
        leads = (
            f"With the {label} absent from its original {surface} position,",
            f"After the {label} disappears from its original {surface} position,",
            f"Before the {label} returns to its original {surface} position,",
            f"During the interval when the {label} is missing from its original {surface} position,",
        )
        tails = (
            "which listed location contains it?",
            "where has the object gone?",
            "which location description is correct?",
            "where should the object be located?",
        )
    return _combine(leads, tails, ordinal)


def _natural_history_uncertain(question: dict[str, Any], ordinal: int) -> str:
    label = _extract(r"manipulations of the (.+?) occur ", question["question"])
    leads = (
        f"While the {label} is out of view before returning,",
        f"During the hidden interval between the {label}'s disappearance and return,",
        f"After the {label} leaves view and before it comes back,",
        f"Across the gap in which the {label} is not visible,",
    )
    tails = (
        "how many separate manipulations occur?",
        "what is the number of distinct manipulations?",
        "how many individual manipulations take place?",
    )
    return _combine(leads, tails, ordinal)


def _natural_tracking_uncertain(question: dict[str, Any], ordinal: int) -> str:
    label = _extract(r"^Is the (.+?) visible at ", question["question"])
    leads = (
        f"When the {label} reappears after leaving view,",
        f"Once the {label} becomes visible again following its absence,",
        f"After the {label} returns to view,",
        f"On seeing the {label} again after it disappeared,",
        f"As the {label} comes back into view,",
        f"Following the {label}'s disappearance and later return,",
    )
    tails = (
        "is it the same physical instance seen earlier?",
        "does it match the physical instance visible before?",
        "is the returning object physically identical to the earlier one?",
        "should it be identified as the very same instance?",
    )
    return _combine(leads, tails, ordinal)


def _natural_state_uncertain(question: dict[str, Any], ordinal: int) -> str:
    label = _extract(r"^What causes the (.+?) to leave ", question["question"])
    leads = (
        f"As the {label} leaves its original position,",
        f"When the {label} moves away from where it began,",
        f"During the {label}'s departure from its starting place,",
    )
    tails = (
        "what causes that movement?",
        "which listed cause accounts for the change?",
    )
    return _combine(leads, tails, ordinal)


def _natural_temporal_uncertain(question: dict[str, Any], ordinal: int) -> str:
    label = _extract(r"involving the (.+?) occurs ", question["question"])
    leads = (
        f"During the interval when the {label} is out of view,",
        f"Between the {label}'s disappearance and return,",
        f"While the {label} remains absent from view,",
        f"Across the hidden part of the {label}'s trajectory,",
        f"After the {label} leaves view and before it reappears,",
    )
    tails = (
        "which ordering of actions occurs?",
        "what is the order of the listed actions?",
        "which action sequence describes what happens?",
    )
    return _combine(leads, tails, ordinal)


def _naturalize(question: dict[str, Any], ordinal: int) -> str:
    question_type = question["question_type"]
    status = question["diagnostics"]["epistemic_status"]
    if status == "uncertain":
        if question_type in {"current_state", "factual_retrieval"}:
            return _natural_location(question, ordinal)
        if question_type == "history_aggregation":
            return _natural_history_uncertain(question, ordinal)
        if question_type == "object_tracking":
            return _natural_tracking_uncertain(question, ordinal)
        if question_type == "state_change":
            return _natural_state_uncertain(question, ordinal)
        if question_type == "temporal_reasoning":
            return _natural_temporal_uncertain(question, ordinal)
        raise base.ExportError(f"unsupported uncertain question type: {question_type}")

    if question_type == "current_state":
        return _combine(CURRENT_LEADS, CURRENT_TAILS, ordinal)
    if question_type == "factual_retrieval":
        return _combine(FACTUAL_LEADS, FACTUAL_TAILS, ordinal)
    if question_type == "history_aggregation":
        return _combine(HISTORY_LEADS, HISTORY_TAILS, ordinal)
    if question_type == "object_tracking":
        return _combine(TRACK_LEADS, TRACK_TAILS, ordinal)
    if question_type == "state_change":
        tails = (
            STATE_DISAPPEAR_TAILS
            if "visible to absent" in question["question"]
            else STATE_APPEAR_TAILS
        )
        return _combine(STATE_LEADS, tails, ordinal)
    if question_type == "temporal_reasoning":
        return _combine(TEMPORAL_LEADS, TEMPORAL_TAILS, ordinal)
    raise base.ExportError(f"unsupported known question type: {question_type}")


def _build_question(
    *,
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    source_question: dict[str, Any],
    ordinal: int,
    maximum: int,
    seed: int,
) -> dict[str, Any]:
    question = _ORIGINAL_BUILD_QUESTION(
        episode_id=episode_id,
        scene=scene,
        events=events,
        source_question=source_question,
        ordinal=ordinal,
        maximum=maximum,
        seed=seed,
    )
    question["question"] = _naturalize(question, ordinal)
    return question


def _audit_questions(questions: list[dict[str, Any]]) -> dict[str, Any]:
    audit = _ORIGINAL_AUDIT(questions)
    counts = Counter(question["question"] for question in questions)
    duplicates = [
        {"question": text, "count": count}
        for text, count in sorted(counts.items())
        if count > 1
    ]
    timestamp_hits = [
        question["id"]
        for question in questions
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:s|seconds?)\b", question["question"], re.IGNORECASE)
    ]
    numeric_hits = [
        question["id"]
        for question in questions
        if re.search(r"\d", question["question"])
    ]
    audit.update({
        "unique_question_text_count": len(counts),
        "duplicate_question_texts": duplicates,
        "timestamp_expression_question_ids": timestamp_hits,
        "numeric_character_question_ids": numeric_hits,
        "natural_language_contract_passed": (
            len(counts) == len(questions)
            and not timestamp_hits
            and not numeric_hits
        ),
    })
    audit["static_release_passed"] = (
        audit["static_release_passed"]
        and audit["natural_language_contract_passed"]
    )
    return audit


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _EXP2_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.3 实验2", "Sim QA v2.4 实验1")
    audit = summary["visual_grounding_audit"]
    return text + "\n".join([
        "",
        "## 10. v2.4 实验1：无秒数与自然去重",
        "",
        "- 公开问题正文不再出现秒数、时间戳或任何阿拉伯数字。",
        "- `query_time` 与 `evidence_spans` 仅保留在隐藏评测字段中，用于 Online 截断和证据审计。",
        "- 时间锚点改写为当前查询点、可见前缀、离开后到返回前、最后一对返回事件等自然相对表达。",
        "- 六类 Known/Uncertain 均使用语义一致的自然变体，不通过 episode ID、场景名或正确物体名制造表面唯一性。",
        f"- 问题正文唯一数：{audit['unique_question_text_count']}/300。",
        f"- 重复正文：{len(audit['duplicate_question_texts'])} 组。",
        f"- 秒数表达命中：{len(audit['timestamp_expression_question_ids'])} 条。",
        f"- 数字字符命中：{len(audit['numeric_character_question_ids'])} 条。",
        f"- 自然语言发布门禁：{'通过' if audit['natural_language_contract_passed'] else '未通过'}。",
        "",
    ])


def _configure_base() -> None:
    exp2._configure_base()
    base.VERSION = VERSION
    base.ID_VERSION_TAG = ID_VERSION_TAG
    base.QA_SOURCE = QA_SOURCE
    base.REPORT_FILENAME = REPORT_FILENAME
    base.SUBTYPES = dict(SUBTYPES)
    base._build_question = _build_question
    base._audit_questions = _audit_questions
    base._report_markdown = _report_markdown


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
    _configure_base()
    summary = (
        base.validate_export(args.output)
        if args.validate_only
        else base.export_clone(
            args.source_dataset,
            args.output,
            link_mode=args.link_mode,
            seed=args.seed,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
