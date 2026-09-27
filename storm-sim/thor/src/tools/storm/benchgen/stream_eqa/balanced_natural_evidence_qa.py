"""Build v2.5 QA with a mix of room and object-event questions.

Current-state and factual-retrieval questions use fixed room/event quotas.
Each question includes a private video_evidence description.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import natural_unique_qa as v24
from . import visual_contrast_qa_exp2 as v22
from . import visual_grounded_qa as grounded


base = v24.base

# Save the original builders before _configure_base replaces them to avoid recursion.
_EVENT_CURRENT_BUILDER = v22._build_current_known
_EVENT_FACTUAL_BUILDER = grounded._build_factual_known

VERSION = "balanced_natural_evidence_v2_5_exp1"
ID_VERSION_TAG = "v25e1"
QA_SOURCE = "simulation_ground_truth_balanced_natural_evidence_v2_5_exp1"
REPORT_FILENAME = "QA_v2.5实验1_难度均衡与自然证据报告_中文.md"
DEFAULT_SEED = 20260825

# Experiment 1 uses 25 current-state and 24 factual-retrieval room questions.
# The remaining known questions in these categories use object events.
CURRENT_KNOWN_TOTAL = 53
FACTUAL_KNOWN_TOTAL = 52
CURRENT_SCENE_COUNT = 25
FACTUAL_SCENE_COUNT = 24

SUBTYPES = dict(v24.SUBTYPES)
SUBTYPES.update({
    "current_state": "mixed_room_and_local_return_v25e1",
    "factual_retrieval": "mixed_room_and_local_transition_v25e1",
})

CURRENT_EVENT_LEADS = (
    "In the latest return event available at the question point,",
    "Comparing the views immediately around the indexed return,",
    "At the visibility change that ends the current prefix,",
    "Looking at the object that has just come back into view,",
    "Across the newest absent-to-visible transition,",
    "In the most recent completed reappearance,",
    "At the return transition currently visible to the agent,",
    "Using the final visibility update before the question point,",
)
CURRENT_EVENT_TAILS = (
    "which listed object is newly visible at its original position?",
    "which object has just returned to where it began?",
    "which listed item completes the return into view?",
    "which object changes from absent to visible?",
    "which listed item is restored to its original place?",
    "which object is the new arrival in the visible scene?",
    "which listed item reappears in that change?",
)

FACTUAL_DISAPPEAR_LEADS = (
    "Comparing the sampled views on either side of the indexed disappearance,",
    "Across the visible-to-absent transition selected from the activity,",
    "In the sampled change where one original position becomes empty,",
    "Looking at the object-removal event indexed in the sequence,",
    "During the selected transition from presence to absence,",
    "From the before-and-after views of the indexed departure,",
    "At the sampled visibility change in which an item leaves view,",
    "Using the local disappearance event from the activity,",
)
FACTUAL_APPEAR_LEADS = (
    "Comparing the sampled views on either side of the indexed reappearance,",
    "Across the absent-to-visible transition selected from the activity,",
    "In the sampled change where one original position is filled again,",
    "Looking at the object-return event indexed in the sequence,",
    "During the selected transition from absence to presence,",
    "From the before-and-after views of the indexed return,",
    "At the sampled visibility change in which an item comes back,",
    "Using the local reappearance event from the activity,",
)
FACTUAL_DISAPPEAR_TAILS = (
    "which listed object disappears from its original position?",
    "which object changes from visible to absent?",
    "which listed item is no longer present after the change?",
    "which object leaves its original place in this event?",
    "which listed item accounts for the newly empty position?",
    "which object is removed from view?",
    "which listed item undergoes the departure?",
)
FACTUAL_APPEAR_TAILS = (
    "which listed object reappears at its original position?",
    "which object changes from absent to visible?",
    "which listed item is present again after the change?",
    "which object returns to its original place in this event?",
    "which listed item accounts for the newly occupied position?",
    "which object comes back into view?",
    "which listed item undergoes the return?",
)

ROOM_DETAILS = {
    "kitchen": "countertops, cabinets, and food-preparation surfaces remain visible",
    "living room": "sofas, side tables, and an open sitting area remain visible",
    "bedroom": "a bed and other sleeping-area furnishings remain visible",
    "bathroom": "a sink, bathing fixtures, and other washroom surfaces remain visible",
}

_GENERIC_EVIDENCE = {
    "Use sampled video observations no later than query_time.",
    "Use only the sampled observations indexed by evidence_spans.",
}


def _is_scene_anchor(ordinal: int, total: int, count: int) -> bool:
    """Select exactly ``count`` well-spread ordinals without using QA content."""
    return ((ordinal * 17 + 3) % total) < count


def _scene_question(
    *,
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
    question_type: str,
    question: str,
    semantic_prefix: str,
) -> dict[str, Any]:
    # Spread room queries across later events to reduce query-time answer bias.
    ordered = base._ordered(events)
    candidates = ordered[max(1, len(ordered) // 2):]
    event = candidates[(ordinal * 7 + 1) % len(candidates)]
    _, query = base._event_points(event, maximum)
    correct = v24.exp2._room_answer(scene)
    start = float(max(0, int(query) - 5))
    if start >= query:
        start = 0.0
    return base._draft(
        episode_id=episode_id,
        question_type=question_type,
        query_time=query,
        question=question,
        concrete_options=list(v24.exp2.ROOM_OPTIONS),
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"{semantic_prefix}:{correct}",
    )


def _build_current_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    if _is_scene_anchor(ordinal, CURRENT_KNOWN_TOTAL, CURRENT_SCENE_COUNT):
        return _scene_question(
            episode_id=episode_id,
            scene=scene,
            events=events,
            ordinal=ordinal,
            maximum=maximum,
            refusal_option=refusal_option,
            question_type="current_state",
            question="What type of room is the agent currently observing?",
            semantic_prefix="visible_room_at_query",
        )
    return _EVENT_CURRENT_BUILDER(
        episode_id, scene, events, ordinal, maximum, refusal_option
    )


def _build_factual_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    if _is_scene_anchor(ordinal, FACTUAL_KNOWN_TOTAL, FACTUAL_SCENE_COUNT):
        return _scene_question(
            episode_id=episode_id,
            scene=scene,
            events=events,
            ordinal=ordinal,
            maximum=maximum,
            refusal_option=refusal_option,
            question_type="factual_retrieval",
            question="Which type of room is the setting of the activity?",
            semantic_prefix="activity_room",
        )
    return _EVENT_FACTUAL_BUILDER(
        episode_id, scene, events, ordinal, maximum, refusal_option
    )


def _naturalize(question: dict[str, Any], ordinal: int, raw_question: str) -> str:
    semantic = str(question["_semantic_class"])
    if semantic.startswith("visible_room_at_query:"):
        return v24._combine(v24.CURRENT_LEADS, v24.CURRENT_TAILS, ordinal)
    if semantic.startswith("activity_room:"):
        return v24._combine(v24.FACTUAL_LEADS, v24.FACTUAL_TAILS, ordinal)
    if semantic.startswith("newly_visible_object:"):
        return v24._combine(CURRENT_EVENT_LEADS, CURRENT_EVENT_TAILS, ordinal)
    if question["question_type"] == "factual_retrieval" and semantic.startswith("object:"):
        disappearing = "visible to absent" in raw_question
        return v24._combine(
            FACTUAL_DISAPPEAR_LEADS if disappearing else FACTUAL_APPEAR_LEADS,
            FACTUAL_DISAPPEAR_TAILS if disappearing else FACTUAL_APPEAR_TAILS,
            ordinal,
        )
    return v24._naturalize(question, ordinal)


def _label_events(events: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    return [event for event in base._ordered(events) if base._label(event) == label]


def _surface_for(events: list[dict[str, Any]], label: str) -> str:
    matches = _label_events(events, label)
    return base._surface(matches[0]) if matches else "support surface"


def _room_evidence(
    scene: str, events: list[dict[str, Any]], correct: str, ordinal: int, current: bool
) -> str:
    labels = list(dict.fromkeys(base._label(event) for event in base._ordered(events)))
    first = labels[ordinal % len(labels)]
    second = labels[(ordinal + 2) % len(labels)]
    scope = (
        "In the available view around the question point"
        if current
        else "Across the sampled activity"
    )
    detail = ROOM_DETAILS[correct]
    links = (
        "while the {first} and {second} appear within the same room layout",
        "and the {first} and {second} are seen against that consistent layout",
        "with the {first} and {second} remaining part of the same indoor setting",
        "as the agent observes the {first} and {second} within that space",
    )
    return f"{scope}, {detail}, {links[ordinal % len(links)].format(first=first, second=second)}."


def _known_evidence(
    question: dict[str, Any], scene: str, events: list[dict[str, Any]], ordinal: int, raw_question: str
) -> str:
    semantic = str(question["_semantic_class"])
    correct = str(question["_correct"])
    if semantic.startswith("visible_room_at_query:"):
        return _room_evidence(scene, events, correct, ordinal, True)
    if semantic.startswith("activity_room:"):
        return _room_evidence(scene, events, correct, ordinal, False)

    surface = _surface_for(events, correct.split(" and ")[0])
    if semantic.startswith("newly_visible_object:"):
        return (
            f"The sampled view before the indexed change shows no {correct} at its original "
            f"{surface} position; the following view shows the {correct} there again."
        )
    if question["question_type"] == "factual_retrieval" and semantic.startswith("object:"):
        if "visible to absent" in raw_question:
            return (
                f"The {correct} is visible at its original {surface} position before the selected "
                "change, and that position is empty in the next sampled view."
            )
        return (
            f"The original {surface} position is empty before the selected change, and the "
            f"{correct} is visibly back there in the following sampled view."
        )
    if semantic.startswith("last_local_return:"):
        return (
            f"Two consecutive returns are visible in the indexed prefix; the {correct} becomes "
            "visible in the second transition, after the other returned object."
        )
    if semantic.startswith("second_local_return:"):
        return (
            f"The indexed pair shows one object return first and the {correct} reappear in the "
            "following transition, placing it second in the visible order."
        )
    if semantic.startswith("tracked_object:"):
        return (
            f"The {correct} leaves its original {surface} position earlier in the visible history "
            "and is shown returning to that position in the indexed reappearance."
        )
    if semantic.startswith("same_action_transition:"):
        disappearing = ":disappearance:" in semantic
        if disappearing:
            return (
                f"The view before the indexed transition contains the {correct} at its original "
                f"{surface} position, while the next view shows that position empty."
            )
        return (
            f"The view before the indexed transition shows an empty original {surface} position, "
            f"while the next view contains the {correct} there again."
        )
    return (
        f"The indexed observations show the answer event involving {correct} after the other "
        "listed alternatives, with both sides of the relevant visibility change in view."
    )


def _uncertain_label(question: dict[str, Any], raw_question: str) -> str:
    raw_patterns = (
        r"^Where is the (.+?) at ",
        r"manipulations of the (.+?) occur ",
        r"^Is the (.+?) visible at ",
        r"^What causes the (.+?) to leave ",
        r"involving the (.+?) occurs ",
    )
    for pattern in raw_patterns:
        match = re.search(pattern, raw_question, re.IGNORECASE)
        if match:
            return match.group(1)
    patterns = (
        r"the (.+?) leaves its original",
        r"the (.+?) is out of view",
        r"the (.+?)'s disappearance",
        r"the (.+?) reappears",
        r"the (.+?) returns",
        r"the (.+?) leaves view",
        r"the (.+?) moves away",
    )
    lowered = question["question"]
    for pattern in patterns:
        match = re.search(pattern, lowered, re.IGNORECASE)
        if match:
            return match.group(1)
    for option in question["_distractors"]:
        match = re.search(r"the (.+?)(?: is| slides| moves|,|$)", option, re.IGNORECASE)
        if match:
            return match.group(1)
    return "object"


def _uncertain_evidence(
    question: dict[str, Any], events: list[dict[str, Any]], ordinal: int, raw_question: str
) -> str:
    semantic = str(question["_semantic_class"])
    label = _uncertain_label(question, raw_question)
    surface = _surface_for(events, label)
    sources = set(question["diagnostics"]["uncertainty_sources"])
    if semantic == "location_not_visible_in_prefix":
        return (
            f"The {label} is shown leaving its original {surface} position and remains outside the "
            "visible frames at the question point; no sampled view reveals its new location."
        )
    if semantic == "hidden_manipulation_count":
        return (
            f"The visible record shows the {label} before it leaves the {surface} and later around "
            "its return, but the intervening off-screen interval does not expose separate manipulations."
        )
    if semantic == "physical_identity_not_visible":
        return (
            f"A category-matching {label} reappears at the {surface} after a visibility gap, but no "
            "persistent instance-specific mark is visible across the gap to establish physical identity."
        )
    if semantic == "transition_mechanism_not_visible":
        return (
            f"The {label} is present at its original {surface} position in one sampled view and absent "
            "in the next; the action that causes the change occurs between the recorded observations."
        )
    if semantic == "hidden_action_order":
        return (
            f"The {label} leaves the visible {surface} area and is not continuously observed during "
            "the indexed interval, so the order of any off-screen actions is not shown."
        )
    source_phrase = " and ".join(sorted(source.replace("_", " ") for source in sources))
    return (
        f"The indexed frames establish only the visible boundary states for the {label}; the "
        f"missing detail is attributable to {source_phrase}, so no single concrete option is shown."
    )


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
    question = v24._ORIGINAL_BUILD_QUESTION(
        episode_id=episode_id,
        scene=scene,
        events=events,
        source_question=source_question,
        ordinal=ordinal,
        maximum=maximum,
        seed=seed,
    )
    raw_question = question["question"]
    question["question"] = _naturalize(question, ordinal, raw_question)
    if question["diagnostics"]["epistemic_status"] == "known":
        evidence = _known_evidence(question, scene, events, ordinal, raw_question)
    else:
        evidence = _uncertain_evidence(question, events, ordinal, raw_question)
    # Use the ordinal to select wording without adding an ID to the question.
    question["video_evidence"] = evidence
    return question


def _audit_questions(questions: list[dict[str, Any]]) -> dict[str, Any]:
    audit = v24._audit_questions(questions)
    evidence_counts = Counter(question["video_evidence"] for question in questions)
    generic_ids = [
        question["id"] for question in questions
        if question["video_evidence"] in _GENERIC_EVIDENCE
    ]
    short_ids = [
        question["id"] for question in questions
        if len(question["video_evidence"].split()) < 12
    ]
    duplicates = [
        {"video_evidence": evidence, "count": count}
        for evidence, count in sorted(evidence_counts.items()) if count > 1
    ]
    scene_current = sum(
        question["diagnostics"]["epistemic_status"] == "known"
        and question["question_type"] == "current_state"
        and question["question_subtype"] == SUBTYPES["current_state"]
        and set(question["options"]) == set(v24.exp2.ROOM_OPTIONS)
        for question in questions
    )
    scene_factual = sum(
        question["diagnostics"]["epistemic_status"] == "known"
        and question["question_type"] == "factual_retrieval"
        and question["question_subtype"] == SUBTYPES["factual_retrieval"]
        and set(question["options"]) == set(v24.exp2.ROOM_OPTIONS)
        for question in questions
    )
    evidence_contract = (
        len(evidence_counts) >= 220
        and not generic_ids
        and not short_ids
        and scene_current == CURRENT_SCENE_COUNT
        and scene_factual == FACTUAL_SCENE_COUNT
    )
    audit.update({
        "unique_video_evidence_count": len(evidence_counts),
        "duplicate_video_evidence": duplicates,
        "generic_video_evidence_question_ids": generic_ids,
        "short_video_evidence_question_ids": short_ids,
        "current_state_scene_anchor_count": scene_current,
        "factual_retrieval_scene_anchor_count": scene_factual,
        "natural_video_evidence_contract_passed": evidence_contract,
    })
    audit["static_release_passed"] = audit["static_release_passed"] and evidence_contract
    return audit


def _room_from_episode(episode_id: str) -> str:
    match = re.search(r"FloorPlan\d+", episode_id)
    if not match:
        return "unknown"
    return base._room_for_scene(match.group(0))


def _distribution_tables(output: Path) -> list[str]:
    rows = [
        json.loads(line)
        for line in (output / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    uncertain = [row for row in rows if row["diagnostics"]["epistemic_status"] == "uncertain"]
    sources = sorted({source for row in uncertain for source in row["diagnostics"]["uncertainty_sources"]})
    type_cross: defaultdict[str, Counter[str]] = defaultdict(Counter)
    room_cross: defaultdict[str, Counter[str]] = defaultdict(Counter)
    combinations: Counter[tuple[str, ...]] = Counter()
    for row in uncertain:
        row_sources = tuple(row["diagnostics"]["uncertainty_sources"])
        combinations[row_sources] += 1
        for source in row_sources:
            type_cross[row["question_type"]][source] += 1
            room_cross[_room_from_episode(row["episode_id"])][source] += 1
    total = Counter(source for row in uncertain for source in row["diagnostics"]["uncertainty_sources"])
    labels = {
        "ambiguous_evidence": "证据存在多种解释",
        "missing_observation": "关键过程未被观察",
        "multiple_candidates": "多个候选均与画面相容",
        "partial_observation": "仅观察到过程的一部分",
    }
    lines = [
        "",
        "## 12. Uncertainty 类型分布",
        "",
        "本节仅统计 55 条 Uncertain。单题允许有多个 uncertainty source，因此来源实例数可以大于题数。",
        "",
        "| uncertainty source | 中文含义 | 涉及题数 | 占 Uncertain |",
        "|---|---|---:|---:|",
    ]
    for source in sources:
        lines.append(f"| `{source}` | {labels.get(source, source)} | {total[source]} | {total[source] / len(uncertain):.2%} |")
    lines.extend([
        "",
        "### 12.1 来源组合",
        "",
        "| 单题来源组合 | 题数 | 占 Uncertain |",
        "|---|---:|---:|",
    ])
    for combination, count in sorted(combinations.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {' + '.join(f'`{source}`' for source in combination)} | {count} | {count / len(uncertain):.2%} |")
    lines.extend([
        "",
        "### 12.2 按问题类型交叉分布",
        "",
        "| 问题类型 | " + " | ".join(f"`{source}`" for source in sources) + " |",
        "|---|" + "---:|" * len(sources),
    ])
    for question_type in sorted(type_cross):
        lines.append("| `" + question_type + "` | " + " | ".join(str(type_cross[question_type][source]) for source in sources) + " |")
    room_names = {
        "kitchen": "厨房",
        "living_room": "客厅",
        "bedroom": "卧室",
        "bathroom": "卫生间",
        "unknown": "未知",
    }
    lines.extend([
        "",
        "### 12.3 按场景交叉分布",
        "",
        "| 场景 | " + " | ".join(f"`{source}`" for source in sources) + " |",
        "|---|" + "---:|" * len(sources),
    ])
    for room in ("kitchen", "living_room", "bedroom", "bathroom", "unknown"):
        if room not in room_cross:
            continue
        lines.append("| " + room_names[room] + " | " + " | ".join(str(room_cross[room][source]) for source in sources) + " |")
    return lines + [""]


_V24_REPORT = v24._report_markdown


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _V24_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.4 实验1", "Sim QA v2.5 实验1")
    audit = summary["visual_grounding_audit"]
    lines = [
        "",
        "## 11. v2.5 实验1：难度均衡与自然证据",
        "",
        f"- Known `current_state`：{CURRENT_SCENE_COUNT} 条房间识别 + {CURRENT_KNOWN_TOTAL - CURRENT_SCENE_COUNT} 条局部物体返回。",
        f"- Known `factual_retrieval`：{FACTUAL_SCENE_COUNT} 条房间识别 + {FACTUAL_KNOWN_TOTAL - FACTUAL_SCENE_COUNT} 条局部物体出现/消失。",
        "- 混合比例依据 v2.4 与 v2.2 的逐类型实测标定；动态目标是总体 Video Offline 约 40%、Video Online 约 50%–58%，不再追求两类题接近满分。",
        "- `video_evidence` 参考 Bike 数据改为逐题自然证据描述：Known 说明可见的前后状态或场景依据，Uncertain 说明可见边界与未观察内容。",
        "- `video_evidence`、diagnostics 与答案标签继续从模型输入白名单排除，不会向被测模型泄漏答案。",
        f"- 自然证据唯一表述数：{audit['unique_video_evidence_count']}/300；通用占位句：{len(audit['generic_video_evidence_question_ids'])} 条。相同可见事实允许复用同一句证据，不使用编号强行去重。",
        f"- 自然证据静态门禁：{'通过' if audit['natural_video_evidence_contract_passed'] else '未通过'}。",
    ]
    lines.extend(_distribution_tables(output))
    return text + "\n".join(lines)


def _configure_base() -> None:
    v24._configure_base()
    base.VERSION = VERSION
    base.ID_VERSION_TAG = ID_VERSION_TAG
    base.QA_SOURCE = QA_SOURCE
    base.REPORT_FILENAME = REPORT_FILENAME
    base.SUBTYPES = dict(SUBTYPES)
    base._build_current_known = _build_current_known
    base._build_factual_known = _build_factual_known
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
