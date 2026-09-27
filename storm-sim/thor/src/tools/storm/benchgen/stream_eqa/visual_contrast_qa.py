"""Build v2.2 experiment 1 with object-name answer options.

State-change and temporal questions use distinct object names to remove
forward/reverse transition pairs and appearance-if-present answer cues.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import visual_grounded_qa as base


VERSION = "visual_contrast_v2_2_exp1"
ID_VERSION_TAG = "v22e1"
QA_SOURCE = "simulation_ground_truth_visual_contrast_v2_2_exp1"
REPORT_FILENAME = "QA_v2.2实验1_视觉对比优化报告_中文.md"
DEFAULT_SEED = 20260824

SUBTYPES = dict(base.SUBTYPES)
SUBTYPES.update({
    "state_change": "same_action_object_transition_v22e1",
    "temporal_reasoning": "local_predecessor_object_v22e1",
})

_BASE_REPORT = base._report_markdown


def _build_state_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Ask for the changed object; all four alternatives are object labels."""
    del scene
    ordered = base._ordered(events)
    event = ordered[(ordinal * 5 + 2) % len(ordered)]
    before, after = base._event_points(event, maximum)
    correct = base._label(event)
    action = (
        "changes from visible to absent"
        if event["event_type"] == "disappearance"
        else "changes from absent to visible"
    )
    concrete = [correct, *base._other_labels(events, correct, ordinal)]
    return base._draft(
        episode_id=episode_id,
        question_type="state_change",
        query_time=after,
        question=f"Which object {action} between {before:.1f}s and {after:.1f}s?",
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[before, after]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"same_action_transition:{event['event_type']}:{correct}",
    )


def _build_temporal_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Ask which object changes in the event immediately before a local anchor."""
    del scene
    ordered = base._ordered(events)
    candidates: list[tuple[int, dict[str, Any], dict[str, Any], list[str]]] = []
    for anchor_index in range(1, len(ordered)):
        previous = ordered[anchor_index - 1]
        anchor = ordered[anchor_index]
        correct = base._label(previous)
        try:
            alternatives = base._other_labels(events, correct, ordinal + anchor_index)
        except base.ExportError:
            continue
        candidates.append((anchor_index, previous, anchor, alternatives))
    if not candidates:
        raise base.ExportError("episode cannot form a four-object local predecessor question")
    _, previous, anchor, alternatives = candidates[(ordinal * 3) % len(candidates)]
    previous_before, _ = base._event_points(previous, maximum)
    anchor_before, query = base._event_points(anchor, maximum)
    correct = base._label(previous)
    anchor_label = base._label(anchor)
    anchor_action = (
        "changes from visible to absent"
        if anchor["event_type"] == "disappearance"
        else "changes from absent to visible"
    )
    return base._draft(
        episode_id=episode_id,
        question_type="temporal_reasoning",
        query_time=query,
        question=(
            f"Which object changes visibility immediately before the {anchor_label} {anchor_action} "
            f"between {anchor_before:.1f}s and {query:.1f}s?"
        ),
        concrete_options=[correct, *alternatives],
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[previous_before, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"local_predecessor_object:{correct}",
    )


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _BASE_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.1", "Sim QA v2.2 实验1")
    text = text.replace("v2.1 的 245 条 Known", "v2.2 实验1的 245 条 Known")
    text = text.replace("v2.1 Known", "v2.2 Known")
    return text + "\n".join([
        "",
        "## 7. 实验1专项修改",
        "",
        "- Baseline：Qwen3.5-9B Known text-only 29.39%，online matched-video 38.37%，视觉增量 +8.98 pp。",
        "- `state_change`：删除同一物体同时出现 disappear/reappear 的成对泄漏；四个选项全部改为同视频真实物体名。",
        "- `temporal_reasoning`：删除 appearance-if-present 规则；题目只询问局部直接前驱物体，四个选项全部为真实物体名。",
        "- 实验保留门槛：Known matched-video − text-only 必须高于 +8.98 pp；最终目标至少 +20 pp。",
        "",
    ])


def _configure_base() -> None:
    base.VERSION = VERSION
    base.ID_VERSION_TAG = ID_VERSION_TAG
    base.QA_SOURCE = QA_SOURCE
    base.REPORT_FILENAME = REPORT_FILENAME
    base.SUBTYPES = dict(SUBTYPES)
    base._build_state_known = _build_state_known
    base._build_temporal_known = _build_temporal_known
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
