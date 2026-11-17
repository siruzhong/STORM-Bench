#!/usr/bin/env python
"""Score a prediction JSONL against a RevisitQA ground-truth JSONL.

Usage:
  python scripts/eval/score_revisitqa_predictions.py \
    --gt-file data/eval_video/RevisitQA/revisitqa_bike/bike_question_answer.jsonl \
    --pred-file /path/to/pred.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import load_records, normalize_mcq_sample  # noqa: E402
from scripts.eval.revisitqa_reliability import format_headline, score_selective  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--out", default=None, help="Optional path for result.json")
    args = parser.parse_args(argv)

    gt_rows = [normalize_mcq_sample(row) for row in load_records(args.gt_file)]
    preds = [json.loads(line) for line in Path(args.pred_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    score = score_selective(gt_rows, preds, bootstrap_replicates=args.bootstrap_replicates)
    print(format_headline(score))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
