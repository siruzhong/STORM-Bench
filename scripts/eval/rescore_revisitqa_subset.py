#!/usr/bin/env python
"""Re-score existing RevisitQA predictions on a later manifest that only removed items.

Per-item inference is independent (each item sees its own causal prefix), so predictions made on a superset
manifest stay valid for every item the later manifest keeps unchanged. This script checks that each kept item's
question text and options match the source run's normalized prompt, filters predictions to the kept ids, and
scores them with the current KA / AR / H metrics. The result records the source run and both manifest hashes.

Usage:
  python scripts/eval/rescore_revisitqa_subset.py --model qwen25_vl_3b \
      --source outputs/revisitqa/previous --target outputs/revisitqa/released
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import dataset_manifest, load_records, normalize_mcq_sample  # noqa: E402
from scripts.eval.revisitqa_datasets import REVISITQA_SUBSETS, revisitqa_paths  # noqa: E402
from scripts.eval.revisitqa_reliability import format_headline, score_selective  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args(argv)
    for subset in REVISITQA_SUBSETS:
        gt_file, _ = revisitqa_paths(subset)
        manifest = dataset_manifest(gt_file)
        gt = {str(r["id"]): normalize_mcq_sample(r) for r in load_records(gt_file)}
        for run in sorted((Path(args.source) / subset / "models").glob(f"*/{args.model}/pred.jsonl")):
            protocol = run.parent.parent.name
            preds = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept = [p for p in preds if str(p["id"]) in gt]
            missing = set(gt) - {str(p["id"]) for p in kept}
            if missing:
                raise ValueError(f"{run}: {len(missing)} manifest items have no prediction")
            for p in kept:
                g = gt[str(p["id"])]
                if p["question"] != g["question"] or p["answer"] != g["answer"]:
                    raise ValueError(f"{run}: item {p['id']} changed after the source run; re-run inference")
            score = score_selective([gt[str(p["id"])] for p in kept], kept, bootstrap_replicates=args.bootstrap_replicates)
            score.update(manifest)
            score["eval_signature"] = kept[0]["eval_signature"]
            score["rescored_from"] = {"pred": str(run), "source_items": len(preds), "kept_items": len(kept)}
            out = Path(args.target) / subset / "models" / protocol / args.model
            out.mkdir(parents=True, exist_ok=True)
            (out / "pred.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept), encoding="utf-8")
            (out / "result.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
            print(f"[rescored] {subset}/{protocol}/{args.model} {len(preds)}->{len(kept)}: {format_headline(score)}")


if __name__ == "__main__":
    main()
