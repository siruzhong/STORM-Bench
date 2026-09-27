#!/usr/bin/env python
"""Score complete STORM predictions or legacy answer-only STORM-Bench outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import load_records, normalize_mcq_sample
from scripts.eval.storm_diagnostics import score_answer_predictions, score_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="data/annotations/storm_cook.questions.jsonl")
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    args = parser.parse_args()

    rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    predictions = load_records(args.pred_file)
    has_status = all(
        isinstance(
            prediction.get(
                "pred_epistemic_status", prediction.get("predicted_epistemic_status")
            ),
            str,
        )
        for prediction in predictions
    )
    if has_status:
        result = score_predictions(
            rows,
            predictions,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        result["score_scope"] = "full"
    else:
        result = score_answer_predictions(rows, predictions)

    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output_file:
        Path(args.output_file).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
