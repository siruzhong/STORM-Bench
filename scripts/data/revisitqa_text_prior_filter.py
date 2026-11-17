#!/usr/bin/env python
"""Remove Known items whose answer several models pick without seeing the video (text-prior filter).

Each model is run in the closed-book forced-choice control (`--forced-choice`: question stem plus the three
substantive options, no abstention option, no video). A Known item is text-solvable when at least
--min-votes of the models answer it correctly. With three options and five models, ≥4 correct by chance has
probability 4.5%. Text-solvable items are removed from the main release and kept in the full release with
their vote count; Uncertain items are not affected.

Usage:
  python scripts/data/revisitqa_text_prior_filter.py --models qwen25_vl_3b,qwen25_vl_7b,... --dry-run
  python scripts/data/revisitqa_text_prior_filter.py --models ... --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import SUBSET_FILES, check_rows, read_jsonl, write_jsonl  # noqa: E402

PRED = "outputs/revisitqa/text_forced_choice/{subset}/models/{model}/pred.jsonl"


def votes_for(subset: str, models: list[str]) -> dict[str, list[bool]]:
    votes: dict[str, list[bool]] = {}
    for model in models:
        path = Path(PRED.format(subset=subset, model=model))
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            votes.setdefault(row["id"], []).append(bool(row["correct"]))
    if any(len(v) != len(models) for v in votes.values()):
        raise ValueError(f"{subset}: incomplete predictions")
    return votes


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    parser.add_argument("--full-root", default="data/eval_video/RevisitQA_full")
    parser.add_argument("--models", required=True)
    parser.add_argument("--min-votes", type=int, default=4)
    parser.add_argument("--record", default="scripts/data/revisitqa_filter/text_prior_dropped_ids.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)
    models = args.models.split(",")
    n = len(models)
    chance = sum(comb(n, k) * (1 / 3) ** k * (2 / 3) ** (n - k) for k in range(args.min_votes, n + 1))
    record = {"models": models, "min_votes": args.min_votes, "chance_rate": round(chance, 4), "subsets": {}}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for subset, (directory, filename) in SUBSET_FILES.items():
        votes = votes_for(subset, models)
        path = Path(args.root) / directory / filename
        rows = read_jsonl(path)
        known = [r for r in rows if r["diagnostics"]["epistemic_status"] == "known"]
        drop = {r["id"] for r in known if sum(votes[r["id"]]) >= args.min_votes}
        keep = [r for r in rows if r["id"] not in drop]
        before, after = check_rows(subset, rows), check_rows(subset, keep)
        types_before = Counter(r["question_type"] for r in rows)
        types_after = Counter(r["question_type"] for r in keep)
        shift = max(abs(100 * types_after[t] / len(keep) - 100 * types_before[t] / len(rows)) for t in types_before)
        dist = Counter(sum(votes[r["id"]]) for r in known)
        record["subsets"][subset] = {
            "known": len(known), "dropped": len(drop), "drop_rate": round(len(drop) / max(len(known), 1), 4),
            "votes_histogram": {k: dist[k] for k in range(n + 1)}, "question_type_max_shift_pp": round(shift, 2),
            "dropped_ids": sorted(drop),
        }
        print(json.dumps({"subset": subset, "known": len(known), "dropped": len(drop),
                          "drop_rate": f"{100 * len(drop) / max(len(known), 1):.1f}%",
                          "votes": {k: dist[k] for k in range(n + 1)}, "n": f"{len(rows)}->{len(keep)}",
                          "uncertain_share": f"{100 * (1 - (len(known) - len(drop)) / len(keep)):.1f}%",
                          "type_shift_pp": round(shift, 2), "errors": after["error_count"],
                          "auroc_q": f"{before['text_status_auroc_question']}->{after['text_status_auroc_question']}"}))
        if args.in_place:
            if after["error_count"]:
                raise RuntimeError(after["errors"])
            backup = Path(args.root) / f"_backup_pre_text_prior_{stamp}" / directory
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup / filename)
            write_jsonl(path, keep)
            full_path = Path(args.full_root) / directory / filename
            full = read_jsonl(full_path)
            for r in full:
                if r["id"] in votes and r["diagnostics"]["epistemic_status"] == "known":
                    r["text_prior_votes"] = f"{sum(votes[r['id']])}/{n}"
            write_jsonl(full_path, full)
    print(f"chance of >= {args.min_votes}/{n} correct for a 3-way item: {chance:.3f}")
    if args.in_place:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"record: {args.record}")


if __name__ == "__main__":
    main()
