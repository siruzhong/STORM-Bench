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
REPORT_FILENAME = "QA_v2.5_experiment_2_visual_gain_rebalancing_report.md"
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
    text = text.replace("Sim QA v2.5 Experiment 1", "Sim QA v2.5 Experiment 2")
    text = text.replace(
        "v2.5 Experiment 1: Balanced Difficulty and Natural Evidence",
        "v2.5 Experiment 2: Balanced Difficulty and Natural Evidence",
    )
    return text + "\n".join([
        "",
        "## 13. v2.5 Experiment 2: Rebalancing Offline Visual Gain",
        "",
        "- Experiment 1 measured 45.00% overall and 46.53% Known for offline video, and 54.67% overall and 59.18% Known for online video.",
        "- Experiment 1 met the difficulty target, but text-only Known reached 30.61%, leaving only +15.92 pp of offline Known visual gain, below the +20 pp gate.",
        "- This experiment converts 8 additional `current_state` and 8 additional `factual_retrieval` items from local-event questions to large-scale room questions.",
        "- The new mixture contains 33 room and 20 event `current_state` items, plus 32 room and 20 event `factual_retrieval` items.",
        "- Estimates from item-level Experiment 1 results place overall offline video near 48%-49%, online video near 57%, and offline Known gain near +20 pp while retaining the target difficulty.",
        "- Task definitions, natural rewrites, uncertainty labels, and item-specific `video_evidence` generation remain unchanged for the other 284 questions.",
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
