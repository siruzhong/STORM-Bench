#!/usr/bin/env python
"""Drop THOR questions that refer to an "indexed" return/departure/removal without saying which one.

An earlier text edit removed the index from these stems, so the question no longer names a unique event and
cannot be answered as posed (found in the model-annotation audit).

Usage: python scripts/data/revisitqa_drop_unindexed_questions.py [--in-place]
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import check_rows, read_jsonl, write_jsonl  # noqa: E402

# "the indexed return/disappearance/departure", "the object-removal event indexed in the sequence", ...
PATTERN = re.compile(r"\bindexed\b", re.I)
MANIFEST = Path("revisitqa_thor/thor_question_answer.jsonl")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {}
    for root in (Path("data/eval_video/RevisitQA"), Path("data/eval_video/RevisitQA_full")):
        rows = read_jsonl(root / MANIFEST)
        drop = [r["id"] for r in rows if PATTERN.search(r["question"].split("Question: ")[-1])]
        keep = [r for r in rows if r["id"] not in set(drop)]
        report = check_rows("thor", keep)
        print(json.dumps({"root": str(root), "n": f"{len(rows)}->{len(keep)}", "dropped": len(drop),
                          "known_dropped": sum(r["diagnostics"]["epistemic_status"] == "known" for r in rows if r["id"] in set(drop)),
                          "errors": report["error_count"]}))
        record[str(root)] = sorted(drop)
        if args.in_place:
            backup = root / f"_backup_pre_unindexed_{stamp}" / MANIFEST.parent
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / MANIFEST, backup / MANIFEST.name)
            write_jsonl(root / MANIFEST, keep)
    if args.in_place:
        Path("scripts/data/revisitqa_filter/unindexed_dropped_ids.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
