#!/usr/bin/env python
"""Reference rows and uncertainty for the RevisitQA paper tables (run where outputs/ lives).

Writes outputs/revisitqa/reference_rows.json with
  blind_agent    per subset: text-only agent that abstains when an episode-grouped NB classifier predicts Uncertain
                 and otherwise picks the longest substantive option; threshold chosen for the best H on the test labels.
  annotators     per annotator and subset: KA and AR of the two label-audit annotators on the 141-item sample,
                 reweighted to the release composition (stratum = subset x release status x question type).
  macro_ci       per protocol and model: 95% interval of the seven-subset averages of KA, AR, and H (mean of per-subset
                 H), from episode-clustered bootstrap within each subset.
  forced_pairing per model with a forced-choice run: on Known items, abstention rate of the default run split by whether
                 the same model answers correctly when the abstention option is removed.
Usage: python scripts/eval/revisitqa_reference_rows.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import ABSTENTION_INDEX, SUBSET_FILES, _nb_status_scores, read_jsonl  # noqa: E402

SUBSETS = list(SUBSET_FILES)


def hmean(a, b):
    return None if a is None or b is None else (0.0 if a + b == 0 else 2 * a * b / (a + b))


def blind_agent(rows: list[dict]) -> dict:
    nb, _ = _nb_status_scores(rows, lambda r: r["question"] + " " + " ".join(r["options"][:3]))
    known = [r for r in rows if r["diagnostics"]["epistemic_status"] == "known"]
    unc = [r for r in rows if r["diagnostics"]["epistemic_status"] != "known"]
    longest = {r["id"]: max(range(3), key=lambda i: (len(r["options"][i]), r["options"][i])) for r in known}
    best = None
    for threshold in sorted(set(nb.values())) + [float("inf")]:
        ka = sum(nb[r["id"]] < threshold and longest[r["id"]] == r["answer_index"] for r in known) / len(known)
        ar = sum(nb[r["id"]] >= threshold for r in unc) / len(unc)
        h = hmean(ka, ar)
        if best is None or h > best["h"]:
            best = {"ka": ka, "ar": ar, "h": h, "threshold": threshold}
    return best


def _choice_index(value) -> int | None:
    if isinstance(value, str) and value.strip().upper() in ("A", "B", "C"):
        return "ABC".index(value.strip().upper())
    return value if isinstance(value, int) else None


def annotator_rows(audit: Path, release: dict[str, dict], by_type: bool = True) -> dict:
    audited = [r["id"] for r in read_jsonl(audit / "key.jsonl")]
    key = {i: True for i in audited if i in release}
    stratum = lambda r: (r["_subset"], r["diagnostics"]["epistemic_status"]) + ((r.get("question_type"),) if by_type else ())
    population = defaultdict(int)
    for r in release.values():
        population[stratum(r)] += 1
    sampled = defaultdict(int)
    for i in key:
        sampled[stratum(release[i])] += 1
    weight = {i: population[stratum(release[i])] / sampled[stratum(release[i])] for i in key}
    out = {}
    for name in ("annotator_ai_a", "annotator_ai_b"):
        labels = {r["id"]: r for r in read_jsonl(audit / f"{name}.jsonl")}
        per_subset, agree_w, total_w = {}, 0.0, 0.0
        for s in SUBSETS:
            ids = [i for i in key if release[i]["_subset"] == s]
            kn = [i for i in ids if release[i]["diagnostics"]["epistemic_status"] == "known"]
            un = [i for i in ids if i not in kn]
            ka = (sum(weight[i] * (labels[i]["status"] == "known"
                                   and _choice_index(labels[i].get("choice")) == release[i]["answer_index"])
                      for i in kn) / sum(weight[i] for i in kn)) if kn else None
            ar = (sum(weight[i] * (labels[i]["status"] == "uncertain") for i in un) / sum(weight[i] for i in un)) if un else None
            per_subset[s] = {"ka": ka, "ar": ar, "h": hmean(ka, ar), "known_n": len(kn), "uncertain_n": len(un)}
        for i in key:
            agree_w += weight[i] * (labels[i]["status"] == release[i]["diagnostics"]["epistemic_status"])
            total_w += weight[i]
        out[name] = {"per_subset": per_subset, "weighted_status_agreement": agree_w / total_w}
    out["audited"], out["in_release"] = len(audited), len(key)
    return out


def outcomes(pred_path: Path, release: dict[str, dict]) -> dict[str, list[tuple[bool, bool]]]:
    """episode -> [(is_known, hit)] where hit is a correct answer (Known) or an abstention (Uncertain)."""
    by_episode = defaultdict(list)
    for line in pred_path.read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        r = release[p["id"]]
        by_episode[r["episode_id"]].append((r["diagnostics"]["epistemic_status"] == "known", bool(p.get("correct"))))
    return by_episode


def rates(groups: list[list[tuple[bool, bool]]]) -> tuple[float | None, float | None]:
    k = [hit for g in groups for known, hit in g if known]
    u = [hit for g in groups for known, hit in g if not known]
    return (sum(k) / len(k) if k else None, sum(u) / len(u) if u else None)


def macro_ci(root: Path, protocol: str, model: str, release: dict, reps: int, seed: int) -> dict | None:
    data = {}
    for s in SUBSETS:
        path = root / s / "models" / protocol / model / "pred.jsonl"
        if not path.exists():
            return None
        data[s] = list(outcomes(path, release).values())
    rng = random.Random(seed)
    samples = {"ka": [], "ar": [], "h": []}
    for _ in range(reps):
        kas, ars, hs = [], [], []
        for s in SUBSETS:
            groups = [rng.choice(data[s]) for _ in data[s]]
            ka, ar = rates(groups)
            kas.append(ka), ars.append(ar), hs.append(hmean(ka, ar))
        if any(v is None for v in kas + ars):
            continue
        samples["ka"].append(sum(kas) / 7), samples["ar"].append(sum(ars) / 7), samples["h"].append(sum(hs) / 7)
    out = {}
    for name, values in samples.items():
        values.sort()
        out[name] = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]]
    return out


def forced_pairing(root: Path, model: str, release: dict) -> dict | None:
    cells = {"right": [0, 0], "wrong": [0, 0]}
    for s in SUBSETS:
        default = root / s / "models" / "online" / model / "pred.jsonl"
        forced = root / s / "models" / "online_all_forced_choice" / model / "pred.jsonl"
        if not (default.exists() and forced.exists()):
            return None
        fmap = {p["id"]: p for p in map(json.loads, filter(str.strip, forced.read_text().splitlines()))}
        for p in map(json.loads, filter(str.strip, default.read_text().splitlines())):
            r = release[p["id"]]
            if r["diagnostics"]["epistemic_status"] != "known" or p["id"] not in fmap:
                continue
            cell = cells["right" if fmap[p["id"]].get("correct") else "wrong"]
            cell[0] += p.get("pred_index") == r.get("abstention_index", ABSTENTION_INDEX)
            cell[1] += 1
    return {k: {"abstained": a, "n": n, "rate": a / n if n else None} for k, (a, n) in cells.items()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/eval_video/RevisitQA")
    parser.add_argument("--pred-root", default="outputs/revisitqa/released")
    parser.add_argument("--audit", default="outputs/revisitqa/label_audit")
    parser.add_argument("--out", default="outputs/revisitqa/reference_rows.json")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args(argv)
    rows_by_subset, release = {}, {}
    for s, (d, f) in SUBSET_FILES.items():
        rows_by_subset[s] = read_jsonl(Path(args.data_root) / d / f)
        for r in rows_by_subset[s]:
            r["_subset"] = s
            release[r["id"]] = r
    root = Path(args.pred_root)
    result = {
        "blind_agent": {s: blind_agent(rows) for s, rows in rows_by_subset.items()},
        "annotators": annotator_rows(Path(args.audit), release),
        "macro_ci": {},
        "forced_pairing": {},
    }
    for protocol in ("online", "streaming_native"):
        models = sorted(p.name for p in (root / "cook" / "models" / protocol).iterdir())
        result["macro_ci"][protocol] = {m: macro_ci(root, protocol, m, release, args.replicates, args.seed) for m in models}
    for model in sorted(p.name for p in (root / "cook" / "models" / "online_all_forced_choice").iterdir()):
        result["forced_pairing"][model] = forced_pairing(root, model, release)
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for s, v in result["blind_agent"].items():
        print("blind", s, {k: round(100 * x, 1) for k, x in v.items() if k != "threshold"})
    ann = result["annotators"]
    print("audited", ann["audited"], "in release", ann["in_release"])
    coarse = annotator_rows(Path(args.audit), release, by_type=False)
    print("subset x status weighting:", {n: round(100 * coarse[n]["weighted_status_agreement"], 1)
                                         for n in ("annotator_ai_a", "annotator_ai_b")})
    for name in ("annotator_ai_a", "annotator_ai_b"):
        print(name, "weighted status agreement", round(100 * ann[name]["weighted_status_agreement"], 1))
    print("forced", {m: {k: round(100 * c["rate"], 1) for k, c in v.items()} for m, v in result["forced_pairing"].items() if v})


if __name__ == "__main__":
    main()
