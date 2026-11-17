#!/usr/bin/env python
"""Apply the two-annotator relabel to RevisitQA.

Rule per item, over the released label and the blind labels of annotator A (all items) and B (A/release
disagreements plus an audit sample), each label = (status, choice):
  - release supported by A or B                         -> keep
  - A and B agree on known with the same choice, release uncertain -> relabel known with that choice
  - otherwise (both against release, invalid, or A, B and release all differ) -> drop
THOR query-time pair candidates (relabel_thor/candidates.jsonl) are judged the same way and the confirmed
ones replace the existing pair items; their Known items still go through the text-prior filter before entering
the main release (`--thor-text-votes`), and each release keeps equal Uncertain/Known counts per 2 s query-time bin.

Usage:
  python scripts/data/revisitqa_relabel_apply.py --dry-run
  python scripts/data/revisitqa_relabel_apply.py --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import ABSTENTION_INDEX, SUBSET_FILES, check_rows, read_jsonl, write_jsonl  # noqa: E402

RELABEL = Path("outputs/revisitqa")
PAIR_SUBTYPE = "query_time_reappearance_order"
VERSION = "two_annotator_relabel"


def _load(path: Path) -> dict:
    return {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}


def _label(row):
    status = str(row["status"]).lower()
    return status, (row.get("choice") if status == "known" else None)


def decisions(package: str) -> dict[str, tuple[str, str | None]]:
    """id -> ("keep" | "known" | "drop", choice)."""
    root = RELABEL / package
    key, a, b = _load(root / "key.jsonl"), _load(root / "annotator_a.jsonl"), _load(root / "annotator_b.jsonl")
    out = {}
    for i, k in key.items():
        rel = (k["status"], k["choice"])
        x = _label(a[i])
        y = _label(b[i]) if i in b else None
        if x == rel or y == rel:
            out[i] = ("keep", None)
        elif y is not None and x == y and rel[0] == "uncertain" and x[0] == "known":
            out[i] = ("known", x[1])
        else:
            out[i] = ("drop", None)
    return out


def relabel_known(row: dict, choice: str) -> dict:
    row = dict(row)
    row["answer_index"] = ord(choice) - ord("A")
    row["diagnostics"] = {**row["diagnostics"], "epistemic_status": "known", "uncertainty_sources": []}
    row["release_meta"] = {"version": VERSION, "change": "uncertain_to_known"}
    return row


def apply_subset(rows: list[dict], dec: dict) -> tuple[list[dict], Counter]:
    out, stats = [], Counter()
    for row in rows:
        action, choice = dec.get(row["id"], ("unchecked", None))
        stats[action] += 1
        if action == "drop":
            continue
        out.append(relabel_known(row, choice) if action == "known" else row)
    return out, stats


def thor_pairs(main_ids: set[str] | None = None) -> list[dict]:
    dec = decisions("relabel_thor")
    rows = []
    for row in read_jsonl(RELABEL / "relabel_thor" / "candidates.jsonl"):
        action, choice = dec[row["id"]]
        row.pop("needs_frame_check", None)
        if action == "drop":
            continue
        row = relabel_known(row, choice) if action == "known" else row
        row["release_meta"] = {**row.get("release_meta", {}), "version": VERSION, "frame_verified": True}
        rows.append(row)
    return rows


def balance_bins(rows: list[dict]) -> list[dict]:
    """Equal Uncertain and Known counts in every 2 s query-time bin, so the query time carries no label information.

    Verified Known pairs are rare early in an episode (the other object's empty spot is seldom filmed again before
    ~38 s), so without this the query time separates the labels."""
    import hashlib

    rank = lambda r: hashlib.sha256(r["id"].encode()).hexdigest()
    groups: dict = {}
    for r in rows:
        groups.setdefault((int(r["query_time"]) // 2, r["diagnostics"]["epistemic_status"]), []).append(r)
    out = []
    for b in sorted({k[0] for k in groups}):
        u, k = sorted(groups.get((b, "uncertain"), []), key=rank), sorted(groups.get((b, "known"), []), key=rank)
        n = min(len(u), len(k))
        out += u[:n] + k[:n]
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    parser.add_argument("--full-root", default="data/eval_video/RevisitQA_full")
    parser.add_argument("--thor-text-votes", help="JSON {id: correct_votes} for THOR pair Known items (5 models).")
    parser.add_argument("--min-votes", type=int, default=4)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    args = parser.parse_args(argv)
    dec = {**decisions("relabel_real"), **decisions("relabel_vhome")}
    pairs = thor_pairs()
    votes = json.loads(Path(args.thor_text_votes).read_text()) if args.thor_text_votes else None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {"version": VERSION, "subsets": {}}
    for subset, (directory, filename) in SUBSET_FILES.items():
        for root, main in ((Path(args.root), True), (Path(args.full_root), False)):
            path = root / directory / filename
            rows = read_jsonl(path)
            if subset == "thor":
                kept = [r for r in rows if r.get("question_subtype") != PAIR_SUBTYPE]
                new = [dict(r) for r in pairs]
                if votes is not None:
                    for r in new:
                        if r["diagnostics"]["epistemic_status"] == "known":
                            r["text_prior_votes"] = f"{votes[r['id']]}/5"
                    if main:
                        new = [r for r in new if r["diagnostics"]["epistemic_status"] != "known"
                               or votes[r["id"]] < args.min_votes]
                elif main and args.in_place:
                    raise SystemExit("--thor-text-votes is required to write the THOR main release")
                new = balance_bins(new)
                out, stats = kept + new, Counter(replaced_pairs=sum(r.get("question_subtype") == PAIR_SUBTYPE for r in rows),
                                                 new_pairs=len(new))
            else:
                out, stats = apply_subset(rows, dec)
            report = check_rows(subset, out)
            n_u = sum(r["diagnostics"]["epistemic_status"] == "uncertain" for r in out)
            tag = "main" if main else "full"
            record["subsets"][f"{subset}/{tag}"] = {"before": len(rows), "after": len(out), **stats,
                                                     "dropped_ids": sorted({r["id"] for r in rows} - {r["id"] for r in out})}
            print(json.dumps({"subset": subset, "release": tag, "n": f"{len(rows)}->{len(out)}", **stats,
                              "uncertain_share": f"{100 * n_u / max(len(out), 1):.1f}%", "errors": report["error_count"],
                              "auroc_q": report["text_status_auroc_question"],
                              "len_cue": report.get("known_gold_strict_longest_given_unique_pct")}))
            if args.in_place:
                if report["error_count"]:
                    raise RuntimeError(report["errors"][:5])
                backup = root / f"_backup_pre_relabel_{stamp}" / directory
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup / filename)
                write_jsonl(path, out)
    if args.in_place:
        out = Path("scripts/data/revisitqa_filter/relabel_record.json")
        out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"record: {out}")


if __name__ == "__main__":
    main()
