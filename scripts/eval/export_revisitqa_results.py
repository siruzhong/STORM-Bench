#!/usr/bin/env python
"""Collect every RevisitQA result.json into one compact JSON for the paper tables and figures.

Usage (on B or C, where outputs/ lives):
  python scripts/eval/export_revisitqa_results.py
Writes outputs/revisitqa/results.json:
  {"runs": {protocol: {model: {subset: summary}}}, "removed": {model: {subset: summary}}, "pooled": {...}}
Pooled Real / Sim / All scores weight each subset by its Known and Uncertain counts (KA and AR are pooled
separately, then H is their harmonic mean), matching the release-level definition.
Each summary also carries by_window: counts, KA, and AR for items whose evidence lies within the latest 16 frames
before t_q ("last16") or starts earlier ("earlier"), recomputed from pred.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REAL = ["cook", "bike", "healthy", "music", "sports"]
SIM = ["thor", "vhome"]
KEEP = ["count", "known_count", "uncertain_count", "accuracy", "known_accuracy", "abstention_recall",
        "false_abstention_rate", "abstention_precision", "coverage", "risk", "effective_reliability",
        "known_abstention_hmean", "invalid_rate", "by_intensity", "by_question_type", "by_evidence_age",
        "by_evidence_age_ge16s", "ci95", "choice_distribution"]


def compact(result: dict) -> dict:
    out = {k: result[k] for k in KEEP if k in result}
    sig = json.loads(result.get("eval_signature") or "{}")
    out["signature"] = {k: sig.get(k) for k in ("protocol_label", "model_path", "mcq_prompt_version", "buffer_frames",
                                                "max_pixels", "dataset_sha256")}
    return out


def pool(per_subset: dict, subsets: list[str]) -> dict | None:
    if not all(s in per_subset for s in subsets):
        return None
    kc = sum(per_subset[s]["known_count"] for s in subsets)
    uc = sum(per_subset[s]["uncertain_count"] for s in subsets)
    ka = sum(per_subset[s]["known_accuracy"] * per_subset[s]["known_count"] for s in subsets) / kc if kc else None
    ar = sum(per_subset[s]["abstention_recall"] * per_subset[s]["uncertain_count"] for s in subsets) / uc if uc else None
    h = None if ka is None or ar is None else (2 * ka * ar / (ka + ar) if ka + ar else 0.0)
    fa = sum(per_subset[s]["false_abstention_rate"] * per_subset[s]["known_count"] for s in subsets) / kc if kc else None
    return {"known_count": kc, "uncertain_count": uc, "known_accuracy": ka, "abstention_recall": ar,
            "known_abstention_hmean": h, "false_abstention_rate": fa}


def gold_rows(subset: str) -> dict[str, dict]:
    from scripts.eval.revisitqa_datasets import revisitqa_paths

    rows = [json.loads(line) for line in Path(revisitqa_paths(subset)[0]).read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def gold_sources(rows: dict[str, dict]) -> dict[str, list[str]]:
    return {i: r["diagnostics"].get("uncertainty_sources") or [] for i, r in rows.items()
            if r["diagnostics"]["epistemic_status"] == "uncertain"}


def by_window(pred_path: Path, rows: dict[str, dict]) -> dict:
    """KA and AR split by whether the evidence starts before the latest 16 frames (raw counts kept for pooling)."""
    from scripts.eval.revisitqa_reliability import evidence_window

    out: dict = {}
    if not pred_path.exists():
        return out
    for line in pred_path.read_text().splitlines():
        pred = json.loads(line) if line.strip() else None
        gt = rows.get(pred["id"]) if pred else None
        window = evidence_window(gt) if gt else None
        if window is None:
            continue
        cell = out.setdefault(window, {"known_count": 0, "known_correct": 0, "uncertain_count": 0, "abstained": 0})
        if gt["diagnostics"]["epistemic_status"] == "known":
            cell["known_count"] += 1
            cell["known_correct"] += bool(pred.get("correct"))
        else:
            cell["uncertain_count"] += 1
            cell["abstained"] += bool(pred.get("correct"))
    for cell in out.values():
        cell["known_accuracy"] = cell["known_correct"] / cell["known_count"] if cell["known_count"] else None
        cell["abstention_recall"] = cell["abstained"] / cell["uncertain_count"] if cell["uncertain_count"] else None
    return out


def by_source(pred_path: Path, sources: dict[str, list[str]]) -> dict:
    """Abstention recall per annotated uncertainty source: {source: [uncertain items, abstained correctly]}."""
    out: dict = {}
    if not pred_path.exists():
        return out
    for line in pred_path.read_text().splitlines():
        row = json.loads(line) if line.strip() else None
        if not row or row["id"] not in sources:
            continue
        for src in sources[row["id"]]:
            cell = out.setdefault(src, [0, 0])
            cell[0] += 1
            cell[1] += bool(row.get("correct"))
    return out


def collect(root: Path, with_sources: bool = True) -> dict:
    runs: dict = {}
    gold: dict = {}
    for path in sorted(root.glob("*/models/*/*/result.json")):
        subset, _, protocol, model = path.parts[-5:-1]
        entry = compact(json.loads(path.read_text()))
        if with_sources:
            if subset not in gold:
                gold[subset] = gold_rows(subset)
            entry["by_window"] = by_window(path.parent / "pred.jsonl", gold[subset])
            if protocol in ("online", "streaming_native"):
                entry["by_source"] = by_source(path.parent / "pred.jsonl", gold_sources(gold[subset]))
        runs.setdefault(protocol, {}).setdefault(model, {})[subset] = entry
    return runs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/revisitqa/released")
    parser.add_argument("--removed-root", default="outputs/revisitqa/textprior_removed")
    parser.add_argument("--out", default="outputs/revisitqa/results.json")
    args = parser.parse_args(argv)
    runs = collect(Path(args.root))
    removed = collect(Path(args.removed_root), with_sources=False).get("online", {})
    pooled = {p: {m: {"real": pool(d, REAL), "sim": pool(d, SIM), "all": pool(d, REAL + SIM)} for m, d in ms.items()}
              for p, ms in runs.items()}
    Path(args.out).write_text(json.dumps({"runs": runs, "removed": removed, "pooled": pooled}) + "\n", encoding="utf-8")
    for p, ms in sorted(pooled.items()):
        done = sum(len(runs[p][m]) for m in ms)
        print(f"{p:26s} models={len(ms):2d} runs={done}")


if __name__ == "__main__":
    main()
