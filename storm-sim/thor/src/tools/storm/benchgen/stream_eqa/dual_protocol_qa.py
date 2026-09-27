"""Build v2.3 experiment 1 using the final sampled query time.

Questions refer to the last visible return so offline and online evaluation
use the same observed history.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import visual_contrast_qa_exp2 as exp2


base = exp2.base

VERSION = "dual_protocol_v2_3_exp1"
ID_VERSION_TAG = "v23e1"
QA_SOURCE = "simulation_ground_truth_dual_protocol_v2_3_exp1"
REPORT_FILENAME = "QA_v2.3实验1_双协议末端证据优化报告_中文.md"
DEFAULT_SEED = 20260825

SUBTYPES = dict(exp2.SUBTYPES)
SUBTYPES.update({
    "current_state": "final_return_current_state_v23e1",
    "factual_retrieval": "final_return_factual_v23e1",
    "history_aggregation": "final_return_history_v23e1",
    "object_tracking": "final_return_track_v23e1",
})


def _last_return(events: list[dict[str, Any]]) -> dict[str, Any]:
    appearances = [
        event for event in base._ordered(events)
        if event["event_type"] == "appearance"
    ]
    if not appearances:
        raise base.ExportError("episode needs a visible return event")
    return appearances[-1]


def _final_query(maximum: int) -> float:
    if maximum <= 0:
        raise base.ExportError("video needs a positive final query time")
    return float(maximum)


def _object_options(
    events: list[dict[str, Any]], event: dict[str, Any], ordinal: int,
) -> tuple[str, list[str]]:
    correct = base._label(event)
    return correct, [correct, *base._other_labels(events, correct, ordinal)]


def _final_return_start(event: dict[str, Any], maximum: int) -> float:
    return base._event_points(event, maximum)[0]


def _history_start(events: list[dict[str, Any]], maximum: int) -> float:
    ordered = base._ordered(events)
    if not ordered:
        raise base.ExportError("episode needs visibility events")
    return base._event_points(ordered[0], maximum)[0]


def _build_current_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Ask which object has the most recent visible return at the final state."""
    del scene
    event = _last_return(events)
    query = _final_query(maximum)
    start = _final_return_start(event, maximum)
    correct, concrete = _object_options(events, event, ordinal)
    return base._draft(
        episode_id=episode_id,
        question_type="current_state",
        query_time=query,
        question=(
            f"At the final observed state at {query:.1f}s, which listed object has most "
            "recently returned to view at its original position?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"final_return_current_state:{correct}",
    )


def _build_factual_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Retrieve the object involved in the final absent-to-visible event."""
    del scene
    event = _last_return(events)
    query = _final_query(maximum)
    start = _final_return_start(event, maximum)
    correct, concrete = _object_options(events, event, ordinal + 1)
    return base._draft(
        episode_id=episode_id,
        question_type="factual_retrieval",
        query_time=query,
        question=(
            f"By {query:.1f}s, which listed object completes the final change from absent "
            "to visible in the observed video?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"final_return_factual:{correct}",
    )


def _build_history_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Aggregate the complete visible history and identify its final return."""
    del scene
    event = _last_return(events)
    query = _final_query(maximum)
    start = _history_start(events, maximum)
    correct, concrete = _object_options(events, event, ordinal + 2)
    return base._draft(
        episode_id=episode_id,
        question_type="history_aggregation",
        query_time=query,
        question=(
            f"Across the complete visible history through {query:.1f}s, which listed object "
            "is the last one to return to view?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"final_return_history:{correct}",
    )


def _build_object_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Track the object whose disappear-and-return cycle finishes last."""
    del scene
    event = _last_return(events)
    query = _final_query(maximum)
    start = _history_start(events, maximum)
    correct, concrete = _object_options(events, event, ordinal + 3)
    return base._draft(
        episode_id=episode_id,
        question_type="object_tracking",
        query_time=query,
        question=(
            f"Which listed object completes its disappear-and-return track last by {query:.1f}s?"
        ),
        concrete_options=concrete,
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[start, query]],
        uncertainty_sources=[],
        events=events,
        semantic_class=f"final_return_track:{correct}",
    )


_EXP2_REPORT = exp2._report_markdown


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _EXP2_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.2 实验2", "Sim QA v2.3 实验1")
    return text + "\n".join([
        "",
        "## 9. v2.3 实验1：双协议末端证据",
        "",
        "- v2.2 基线：Known Offline-video 26.12%、Offline-text 24.08%，增益仅 +2.04 pp。",
        "- 同一基线的 Online-video 44.49%、Online-text 24.08%，增益 +20.41 pp。",
        "- 失败诊断显示，整段 Offline 视频中的后续事件会干扰局部时刻题。",
        "- 本实验只重写 `current_state`、`factual_retrieval`、`history_aggregation`、`object_tracking` 的 Known 题。",
        "- 新题询问整段可见历史里的最后一次恢复可见事件；查询点位于视频最后采样秒，Offline/Online 获得一致的任务证据。",
        "- 四个选项仍全部是该视频中的真实物体名，不加入拒答项或题面答案提示。",
        "- `state_change` 与 `temporal_reasoning` 保持 v2.2 实验2不变，用于隔离本轮改动。",
        "",
    ])


def _configure_base() -> None:
    exp2._configure_base()
    base.VERSION = VERSION
    base.ID_VERSION_TAG = ID_VERSION_TAG
    base.QA_SOURCE = QA_SOURCE
    base.REPORT_FILENAME = REPORT_FILENAME
    base.SUBTYPES = dict(SUBTYPES)
    base._build_current_known = _build_current_known
    base._build_factual_known = _build_factual_known
    base._build_history_known = _build_history_known
    base._build_object_known = _build_object_known
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
