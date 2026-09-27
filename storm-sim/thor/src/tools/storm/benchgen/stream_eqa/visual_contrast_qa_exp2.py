"""Build v2.2 experiment 2 around visible return events.

Current-state, history and temporal builders use local return events.
State-change questions retain the experiment 1 builder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import visual_contrast_qa as exp1


base = exp1.base

VERSION = "visual_contrast_v2_2_exp2"
ID_VERSION_TAG = "v22e2"
QA_SOURCE = "simulation_ground_truth_visual_contrast_v2_2_exp2"
REPORT_FILENAME = "QA_v2.2实验2_短窗视觉优化报告_中文.md"
DEFAULT_SEED = 20260824

SUBTYPES = dict(exp1.SUBTYPES)
SUBTYPES.update({
    "current_state": "newly_visible_object_v22e2",
    "history_aggregation": "last_local_return_object_v22e2",
    "temporal_reasoning": "second_local_return_object_v22e2",
})


def _appearances(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [event for event in base._ordered(events) if event["event_type"] == "appearance"]
    if len(values) < 2:
        raise base.ExportError("episode needs at least two visible return events")
    return values


def _object_options(events: list[dict[str, Any]], event: dict[str, Any], ordinal: int) -> tuple[str, list[str]]:
    correct = base._label(event)
    return correct, [correct, *base._other_labels(events, correct, ordinal)]


def _build_current_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Identify the object that is newly visible at the current query time."""
    del scene
    event = _appearances(events)[(ordinal * 3) % len(_appearances(events))]
    before, query = base._event_points(event, maximum)
    correct, concrete = _object_options(events, event, ordinal)
    return base._draft(
        episode_id=episode_id,
        question_type="current_state",
        query_time=query,
        question=(
            f"At {query:.1f}s, which listed object is newly visible at its original position "
            f"after being absent immediately before {before:.1f}s?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[before, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"newly_visible_object:{correct}",
    )


def _local_return_pair(events: list[dict[str, Any]], ordinal: int) -> tuple[dict[str, Any], dict[str, Any]]:
    appearances = _appearances(events)
    anchor_index = 1 + (ordinal * 3) % (len(appearances) - 1)
    return appearances[anchor_index - 1], appearances[anchor_index]


def _build_history_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Aggregate two local returns and identify the last returned object."""
    del scene
    first, second = _local_return_pair(events, ordinal)
    start = base._event_points(first, maximum)[0]
    query = base._event_points(second, maximum)[1]
    correct, concrete = _object_options(events, second, ordinal + 1)
    return base._draft(
        episode_id=episode_id,
        question_type="history_aggregation",
        query_time=query,
        question=(
            f"Considering the two objects that return to view between {start:.1f}s and {query:.1f}s, "
            "which one returns last?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"last_local_return:{correct}",
    )


def _build_temporal_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Ask for the second object in a local, two-return temporal sequence."""
    del scene
    first, second = _local_return_pair(events, ordinal + 1)
    start = base._event_points(first, maximum)[0]
    query = base._event_points(second, maximum)[1]
    correct, concrete = _object_options(events, second, ordinal + 2)
    return base._draft(
        episode_id=episode_id,
        question_type="temporal_reasoning",
        query_time=query,
        question=(
            f"Which object is second to change from absent to visible in the two-event sequence "
            f"between {start:.1f}s and {query:.1f}s?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"second_local_return:{correct}",
    )


_EXP1_REPORT = exp1._report_markdown


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _EXP1_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.2 实验1", "Sim QA v2.2 实验2")
    return text + "\n".join([
        "",
        "## 8. 实验2专项修改",
        "",
        "- 实验1动态结果：Known text-only 25.31%，matched-video 37.96%，视觉增量 +12.65 pp。",
        "- `current_state`：改为短窗内刚刚恢复可见的物体，答案仍是四个真实物体名。",
        "- `history_aggregation`：聚合相邻两次恢复可见事件，询问最后恢复的物体。",
        "- `temporal_reasoning`：询问相邻两次恢复可见事件中的第二个物体，不再暴露命名锚点。",
        "- 三类题都只使用 `query_time` 之前的证据，并保留原类型、状态、房间及 ABCD 分布。",
        "",
    ])


def _configure_base() -> None:
    exp1._configure_base()
    base.VERSION = VERSION
    base.ID_VERSION_TAG = ID_VERSION_TAG
    base.QA_SOURCE = QA_SOURCE
    base.REPORT_FILENAME = REPORT_FILENAME
    base.SUBTYPES = dict(SUBTYPES)
    base._build_current_known = _build_current_known
    base._build_history_known = _build_history_known
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
