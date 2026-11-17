#!/usr/bin/env python
"""Merge blind relabels of RevisitQA with the released labels.

Inputs (all JSONL, one line per item; annotator lines follow scripts/data/revisitqa_label_rubric.md):
  --key        key.jsonl from revisitqa_relabel_package.py (released status/choice/causes)
  --primary    annotator A, covering every item
  --second     annotator B, covering items flagged by `select` plus a random audit sample
Subcommands:
  select   write the ids annotator B must label: every A/release disagreement + an audit sample of agreements
  merge    apply the decision rules and write decisions.jsonl and a summary

Decision rules (merge):
  A agrees with release                                   -> keep (B, if present, only feeds the audit miss rate)
  A disagrees, B agrees with release                      -> keep
  A and B agree with each other against release:
      uncertain -> known with the same choice             -> relabel known
      known -> uncertain                                   -> adjudicate on original-resolution frames
      known with a different choice                        -> adjudicate
      invalid                                              -> adjudicate (drop if confirmed)
  anything else (B missing or three-way split)             -> adjudicate
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load(path):
    return {r["id"]: r for r in (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}


def label(row):
    status = str(row.get("status", "")).lower()
    return (status, row.get("choice") if status == "known" else None)


def select(args):
    key, primary = load(args.key), load(args.primary)
    missing = sorted(set(key) - set(primary))
    if missing:
        raise SystemExit(f"annotator A misses {len(missing)} items, e.g. {missing[:3]}")
    disagree = [i for i in key if label(primary[i]) != label(key[i])]
    agree = sorted(set(key) - set(disagree))
    audit = random.Random(args.seed).sample(agree, round(len(agree) * args.audit_fraction))
    rows = [{"id": i, "reason": "disagree"} for i in sorted(disagree)] + [{"id": i, "reason": "audit"} for i in sorted(audit)]
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(json.dumps({"items": len(key), "disagree": len(disagree), "audit": len(audit), "out": args.out}))


def decide(rel, a, b):
    if label(a) == label(rel):
        return "keep", None
    if b is None:
        return "adjudicate", "second_missing"
    if label(b) == label(rel):
        return "keep", "second_sides_with_release"
    if label(a) != label(b):
        return "adjudicate", "three_way_split"
    status, choice = label(a)
    if status == "known" and rel["status"] == "uncertain":
        return "relabel_known", None
    if status == "uncertain":
        return "adjudicate", "known_to_uncertain"
    if status == "invalid":
        return "adjudicate", "invalid"
    return "adjudicate", "different_choice"


def merge(args):
    key, primary = load(args.key), load(args.primary)
    second = load(args.second) if args.second else {}
    out, counts, audit = [], Counter(), Counter()
    for i, rel in key.items():
        a, b = primary.get(i), second.get(i)
        if a is None:
            raise SystemExit(f"annotator A misses {i}")
        if b is not None and label(a) == label(rel):
            audit["checked"] += 1
            audit["second_disagrees"] += label(b) != label(rel)
        decision, reason = decide(rel, a, b)
        counts[(rel["subset"], decision)] += 1
        out.append({"id": i, "subset": rel["subset"], "decision": decision, "reason": reason,
                    "release": {"status": rel["status"], "choice": rel["choice"], "causes": rel["causes"]},
                    "a": {k: a.get(k) for k in ("status", "choice", "causes", "rule", "evidence")},
                    "b": None if b is None else {k: b.get(k) for k in ("status", "choice", "causes", "rule", "evidence")}})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    summary = {"by_subset_decision": {f"{s}/{d}": n for (s, d), n in sorted(counts.items())}, "audit": dict(audit)}
    print(json.dumps(summary, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--key", required=True); s.add_argument("--primary", required=True)
    s.add_argument("--audit-fraction", type=float, default=0.15); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--key", required=True); m.add_argument("--primary", required=True)
    m.add_argument("--second"); m.add_argument("--out", required=True)
    args = p.parse_args(argv)
    (select if args.cmd == "select" else merge)(args)


if __name__ == "__main__":
    main()
