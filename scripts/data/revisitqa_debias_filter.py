#!/usr/bin/env python
"""Adversarial filtering of RevisitQA Real subsets against text-only status prediction, with balance constraints.

Repeatedly scores every item with an out-of-fold text classifier (question + substantive options -> Uncertain?)
and removes the most text-identifiable Uncertain items together with the least Uncertain-looking Known items,
keeping these distributions within tolerance of the unfiltered subset:
  epistemic status ratio, question type, change-intensity bin, evidence age bin (<10 s / >=10 s),
  and the six uncertainty causes among Uncertain items.

Usage:
  python scripts/data/revisitqa_debias_filter.py --dry-run
  python scripts/data/revisitqa_debias_filter.py --in-place --full-root data/eval_video/RevisitQA_full
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
from scripts.data.revisitqa_unify_abstention import (  # noqa: E402
    SUBSET_FILES,
    _auroc,
    _nb_status_scores,
    check_rows,
    read_jsonl,
    write_jsonl,
)

REAL = ("cook", "bike", "healthy", "music", "sports")
TOLERANCE_PP = 2.0
STATUS_TOLERANCE_PP = 1.5
TARGET_AUROC = 0.65
MIN_KEEP_FRACTION = 0.65
STEP_FRACTION = 0.03


def text(row: dict) -> str:
    return row["question"].split("Question: ")[-1] + " " + " ".join(row["options"][:3])


def status(row: dict) -> str:
    return row["diagnostics"]["epistemic_status"]


def intensity_bin(row: dict) -> str:
    value = int(row["change_intensity"])
    return "low" if value <= 3 else ("medium" if value <= 6 else "high")


def age_bin(row: dict) -> str:
    age = float(row["query_time"]) - max(float(span[1]) for span in row["evidence_spans"])
    return "ge10s" if age >= 10 else "lt10s"


def shares(rows: list[dict]) -> dict[str, float]:
    """Percent shares of every constrained category."""
    n = len(rows)
    out: dict[str, float] = {}
    counters = {
        "status": Counter(status(r) for r in rows),
        "type": Counter(r["question_type"] for r in rows),
        "intensity": Counter(intensity_bin(r) for r in rows),
        "age": Counter(age_bin(r) for r in rows),
    }
    for name, counter in counters.items():
        for key, value in counter.items():
            out[f"{name}:{key}"] = 100 * value / n
    uncertain = [r for r in rows if status(r) == "uncertain"]
    causes = Counter(c for r in uncertain for c in set(r["diagnostics"]["uncertainty_sources"]))
    for key, value in causes.items():
        out[f"cause:{key}"] = 100 * value / max(len(uncertain), 1)
    return out


def within(reference: dict[str, float], current: dict[str, float], cause_tolerance: float = TOLERANCE_PP) -> bool:
    for key, ref in reference.items():
        if key.startswith("status:"):
            tol = STATUS_TOLERANCE_PP
        elif key.startswith("cause:"):
            tol = cause_tolerance
        else:
            tol = TOLERANCE_PP
        if abs(current.get(key, 0.0) - ref) > tol:
            return False
    return True


def filter_subset(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    reference = shares(rows)
    # One Uncertain item moves a cause share by 100/n_U points; allow at least 1.5 such steps.
    n_uncertain = sum(status(r) == "uncertain" for r in rows)
    cause_tolerance = max(TOLERANCE_PP, 150 / max(n_uncertain, 1))
    keep = list(rows)
    history = []
    while True:
        scores, labels = _nb_status_scores(keep, text)
        auroc = _auroc([scores[r["id"]] for r in keep], labels)
        history.append({"n": len(keep), "auroc": round(auroc, 4)})
        if auroc <= TARGET_AUROC or len(keep) <= MIN_KEEP_FRACTION * len(rows):
            break
        uncertain = sorted((r for r in keep if status(r) == "uncertain"), key=lambda r: -scores[r["id"]])
        known = sorted((r for r in keep if status(r) == "known"), key=lambda r: scores[r["id"]])
        ratio = len(known) / max(len(uncertain), 1)
        dropped: set[str] = set()
        accepted_u = 0
        # Per-type quotas: every question type gives up its most text-identifiable Uncertain items in proportion.
        by_type: dict[str, list[dict]] = {}
        for row in uncertain:
            by_type.setdefault(row["question_type"], []).append(row)
        type_quota = {t: max(1, round(STEP_FRACTION * len(rs))) for t, rs in by_type.items()}
        taken = Counter()
        ordered = [r for rs in by_type.values() for r in rs]
        ordered.sort(key=lambda r: -scores[r["id"]])
        for candidate in ordered:
            if taken[candidate["question_type"]] >= type_quota[candidate["question_type"]]:
                continue
            # Pair each Uncertain removal with ~ratio Known removals from the same question type,
            # preferring the same intensity and evidence-age bins, so every distribution holds.
            need_k = round((accepted_u + 1) * ratio) - round(accepted_u * ratio)
            pool = sorted(
                (r for r in known if r["id"] not in dropped and r["question_type"] == candidate["question_type"]),
                key=lambda r: (intensity_bin(r) != intensity_bin(candidate), age_bin(r) != age_bin(candidate), scores[r["id"]]),
            )
            partners = pool[:need_k]
            if len(partners) < need_k:
                continue
            removed = dropped | {candidate["id"]} | {r["id"] for r in partners}
            if within(reference, shares([r for r in keep if r["id"] not in removed]), cause_tolerance):
                dropped = removed
                accepted_u += 1
                taken[candidate["question_type"]] += 1
        if not dropped:
            break
        keep = [r for r in keep if r["id"] not in dropped]
    return keep, history


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    parser.add_argument("--subsets", default=",".join(REAL))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    parser.add_argument("--full-root", default="data/eval_video/RevisitQA_full")
    parser.add_argument("--record", default="scripts/data/revisitqa_filter/text_classifier_dropped_ids.json")
    args = parser.parse_args(argv)
    root = Path(args.root)
    record = {"version": "text_classifier_filter", "target_auroc": TARGET_AUROC, "tolerance_pp": TOLERANCE_PP,
              "status_tolerance_pp": STATUS_TOLERANCE_PP, "subsets": {}}
    if args.in_place:
        full = Path(args.full_root)
        if full.exists():
            raise RuntimeError(f"{full} exists; refusing to overwrite the full release")
        for subset, (directory, filename) in SUBSET_FILES.items():
            (full / directory).mkdir(parents=True)
            shutil.copy2(root / directory / filename, full / directory / filename)
            (full / directory / "video").symlink_to((root / directory / "video").resolve())
    backup = root / f"_backup_pre_text_filter_{datetime.now():%Y%m%d_%H%M%S}"
    for subset in args.subsets.split(","):
        directory, filename = SUBSET_FILES[subset]
        path = root / directory / filename
        rows = read_jsonl(path)
        keep, history = filter_subset(rows)
        before, after = shares(rows), shares(keep)
        report = check_rows(subset, keep)
        dropped = sorted({r["id"] for r in rows} - {r["id"] for r in keep})
        record["subsets"][subset] = {"before": len(rows), "after": len(keep), "auroc_path": history,
                                     "dropped_ids": dropped,
                                     "max_share_shift_pp": round(max(abs(after.get(k, 0) - v) for k, v in before.items()), 2),
                                     "episodes_before": len({r['episode_id'] for r in rows}),
                                     "episodes_after": len({r['episode_id'] for r in keep})}
        print(json.dumps({"subset": subset, "n": f"{len(rows)}->{len(keep)}",
                          "auroc": f"{history[0]['auroc']}->{history[-1]['auroc']}",
                          "auroc_q_after": report["text_status_auroc_question"],
                          "max_shift_pp": record['subsets'][subset]["max_share_shift_pp"],
                          "episodes": f"{record['subsets'][subset]['episodes_before']}->{record['subsets'][subset]['episodes_after']}",
                          "shares": {k: [round(v, 1), round(after.get(k, 0), 1)] for k, v in sorted(before.items())}}))
        if args.in_place:
            if report["error_count"]:
                raise RuntimeError(report["errors"])
            (backup / directory).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup / directory / filename)
            write_jsonl(path, keep)
    if args.in_place:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        Path(args.record).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        print(f"backup: {backup}\nfull release: {args.full_root}\nrecord: {args.record}")


if __name__ == "__main__":
    main()
