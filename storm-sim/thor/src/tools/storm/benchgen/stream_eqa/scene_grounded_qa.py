"""Build v2.3 experiment 2 with room-based current-state and factual questions.

The other four question types use the v2.2 event builders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import visual_contrast_qa_exp2 as exp2


base = exp2.base

VERSION = "scene_grounded_v2_3_exp2"
ID_VERSION_TAG = "v23e2"
QA_SOURCE = "simulation_ground_truth_scene_grounded_v2_3_exp2"
REPORT_FILENAME = "QA_v2.3实验2_场景视觉锚定优化报告_中文.md"
DEFAULT_SEED = 20260825

ROOM_LABELS = {
    "kitchen": "kitchen",
    "living_room": "living room",
    "bedroom": "bedroom",
    "bathroom": "bathroom",
}
ROOM_OPTIONS = list(ROOM_LABELS.values())

SUBTYPES = dict(exp2.SUBTYPES)
SUBTYPES.update({
    "current_state": "visible_room_at_query_v23e2",
    "factual_retrieval": "activity_room_retrieval_v23e2",
})


def _room_answer(scene: str) -> str:
    room = base._room_for_scene(scene)
    try:
        return ROOM_LABELS[room]
    except KeyError as error:
        raise base.ExportError(f"unsupported AI2-THOR room type: {room}") from error


def _scene_question(
    *,
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    maximum: int,
    refusal_option: str,
    question_type: str,
    question: str,
    semantic_prefix: str,
) -> dict[str, Any]:
    query = float(maximum)
    if query <= 0:
        raise base.ExportError("scene question needs a positive final query time")
    correct = _room_answer(scene)
    return base._draft(
        episode_id=episode_id,
        question_type=question_type,
        query_time=query,
        question=question.format(query=query),
        concrete_options=list(ROOM_OPTIONS),
        correct=correct,
        refusal_option=refusal_option,
        evidence_spans=[[0.0, query]],
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
    """Identify the room visible at the final current state."""
    del ordinal
    return _scene_question(
        episode_id=episode_id,
        scene=scene,
        events=events,
        maximum=maximum,
        refusal_option=refusal_option,
        question_type="current_state",
        question="At {query:.1f}s, what type of room is the agent currently observing?",
        semantic_prefix="visible_room_at_query",
    )


def _build_factual_known(
    episode_id: str,
    scene: str,
    events: list[dict[str, Any]],
    ordinal: int,
    maximum: int,
    refusal_option: str,
) -> dict[str, Any]:
    """Retrieve the persistent room setting from the video."""
    del ordinal
    return _scene_question(
        episode_id=episode_id,
        scene=scene,
        events=events,
        maximum=maximum,
        refusal_option=refusal_option,
        question_type="factual_retrieval",
        question="Which type of room is the setting of the activity shown through {query:.1f}s?",
        semantic_prefix="activity_room",
    )


_EXP2_REPORT = exp2._report_markdown


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _EXP2_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.2 实验2", "Sim QA v2.3 实验2")
    return text + "\n".join([
        "",
        "## 9. v2.3 实验2：场景视觉锚定",
        "",
        "- v2.3 实验1证明：即使 Offline/Online 证据对齐，Qwen3.5-9B 仍不能稳定追踪整段视频中的小物体返回顺序。",
        "- 本实验回到 v2.2 基线，仅重写 53 条 Known `current_state` 和 52 条 Known `factual_retrieval`。",
        "- 两类题分别询问查询时刻可见的房间类型，以及整段活动发生的房间类型。",
        "- 选项固定为 kitchen、living room、bedroom、bathroom；四类场景各 75 题，答案位置继续全局均衡。",
        "- 房间类型是贯穿视频的大尺度视觉事实，不依赖小物体时序，也无法从不含场景名的题干闭卷推出。",
        "- `history_aggregation`、`object_tracking`、`state_change`、`temporal_reasoning` 保持 v2.2 实验2不变，继续保留事件级推理难度。",
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
