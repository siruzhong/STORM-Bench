"""Build v2.5 experiment 2 with revised room/event quotas.

Room questions account for 33 of 53 known current-state questions and 32 of 52
known factual-retrieval questions. Other builders use the experiment 1 settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import balanced_natural_evidence_qa as exp1


base = exp1.base

VERSION = "balanced_natural_evidence_v2_5_exp2"
ID_VERSION_TAG = "v25e2"
QA_SOURCE = "simulation_ground_truth_balanced_natural_evidence_v2_5_exp2"
REPORT_FILENAME = "QA_v2.5实验2_视觉增益再均衡报告_中文.md"
DEFAULT_SEED = 20260825

CURRENT_SCENE_COUNT = 33
FACTUAL_SCENE_COUNT = 32

SUBTYPES = dict(exp1.SUBTYPES)
SUBTYPES.update({
    "current_state": "mixed_room_and_local_return_v25e2",
    "factual_retrieval": "mixed_room_and_local_transition_v25e2",
})

_EXP1_REPORT = exp1._report_markdown


def _report_markdown(summary: dict[str, Any], source: Path, output: Path) -> str:
    text = _EXP1_REPORT(summary, source, output)
    text = text.replace("Sim QA v2.5 实验1", "Sim QA v2.5 实验2")
    text = text.replace("v2.5 实验1：难度均衡与自然证据", "v2.5 实验2：难度均衡与自然证据")
    return text + "\n".join([
        "",
        "## 13. v2.5 实验2：Offline 视觉增益再均衡",
        "",
        "- 实验1实测：Video Offline 总体 45.00%、Known 46.53%；Video Online 总体 54.67%、Known 59.18%。",
        "- 实验1难度已命中目标，但 Text-only Known 为 30.61%，导致 Offline Known 视觉净增益只有 +15.92 pp，未达到 +20 pp 门槛。",
        "- 本实验只把额外 8 条 current-state 和 8 条 factual-retrieval 从局部事件题改为大尺度房间题。",
        "- 新混合为 current-state 33 条房间题 + 20 条事件题，factual-retrieval 32 条房间题 + 20 条事件题。",
        "- 按实验1逐题结果估算：Video Offline 总体约 48%–49%，Video Online 约 57%，仍在难度目标区间；Offline Known 增益约 +20 pp。",
        "- 其余 284 道题的任务定义、自然改写、uncertainty 标签和逐题 video_evidence 生成逻辑不变。",
        "",
    ])


def _configure_base() -> None:
    # The exp1 builders and audit resolve these module globals at runtime.
    exp1.CURRENT_SCENE_COUNT = CURRENT_SCENE_COUNT
    exp1.FACTUAL_SCENE_COUNT = FACTUAL_SCENE_COUNT
    exp1.SUBTYPES = dict(SUBTYPES)
    exp1._configure_base()
    base.VERSION = VERSION
    base.ID_VERSION_TAG = ID_VERSION_TAG
    base.QA_SOURCE = QA_SOURCE
    base.REPORT_FILENAME = REPORT_FILENAME
    base.SUBTYPES = dict(SUBTYPES)
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
