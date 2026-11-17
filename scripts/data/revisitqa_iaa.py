#!/usr/bin/env python
"""Inter-annotator agreement for RevisitQA answerability (Known/Uncertain) and uncertainty-cause labels.

sample: draw a stratified sample (subset x status x question type) and write
        <out>/sheet.jsonl   blind annotation sheet (no labels, no options order hints beyond the item text)
        <out>/key.jsonl     gold labels for scoring (keep away from annotators)
        <out>/GUIDE.md      annotation instructions
score:  read two or more filled sheets and report Cohen's kappa (pairwise), Fleiss' kappa, per-cause kappa,
        adjudication rate, and agreement of the adjudicated label with the released label.

Usage:
  python scripts/data/revisitqa_iaa.py sample --root data/eval_video/RevisitQA --out data/eval_video/RevisitQA/_annotation --n 300
  python scripts/data/revisitqa_iaa.py score --out data/eval_video/RevisitQA/_annotation annotator_a.jsonl annotator_b.jsonl
Each filled sheet line: {"id": ..., "status": "known"|"uncertain", "causes": [...]}  (causes only when uncertain)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import SUBSET_FILES, read_jsonl  # noqa: E402

CAUSES = ["missing_observation", "partial_observation", "ambiguous_evidence", "low_visual_quality",
          "multiple_candidates", "ambiguous_attribute"]
GUIDE = """# RevisitQA answerability annotation guide

Each item gives a video path, query time `query_time` (seconds), the question, and three candidate answers.
Watch only frames at or before `query_time` (1 FPS is enough). Do not use later frames.

1. `status`:
   - `known`: the observed prefix uniquely supports one candidate.
   - `uncertain`: the observed prefix does not uniquely decide among the candidates
     (e.g., the critical action is off-screen, occluded, illegible, or multiple candidates fit).
2. `causes` (only when `uncertain`; multi-label allowed):
   - `missing_observation`
   - `partial_observation`
   - `ambiguous_evidence`
   - `low_visual_quality`
   - `multiple_candidates`
   - `ambiguous_attribute`
3. Annotators work independently. Disagreements are adjudicated by a third person into a separate file.

Write one JSON object per line: `{"id": "...", "status": "known", "causes": []}`.
For the full rubric, see `scripts/data/revisitqa_label_rubric.md`.
"""


def _h(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def sample(args) -> None:
    root = Path(args.root)
    rows = []
    for subset, (directory, filename) in SUBSET_FILES.items():
        for r in read_jsonl(root / directory / filename):
            r["_subset"], r["_video"] = subset, str(root / directory / "video" / f"{r['episode_id']}.mp4")
            rows.append(r)
    strata = defaultdict(list)
    for r in rows:
        strata[(r["_subset"], r["diagnostics"]["epistemic_status"], r.get("question_type"))].append(r)
    # Oversample Uncertain to 40% so cause agreement has enough items; allocate within status by stratum size.
    n_unc = round(0.4 * args.n)
    picked = []
    for status, n_status in (("uncertain", n_unc), ("known", args.n - n_unc)):
        keys = [k for k in strata if k[1] == status]
        total = sum(len(strata[k]) for k in keys)
        quota = {k: max(1, round(n_status * len(strata[k]) / total)) for k in keys}
        for k in keys:
            pool = sorted(strata[k], key=lambda r: _h(f"iaa:{r['id']}"))
            picked.extend(pool[: quota[k]])
    picked.sort(key=lambda r: _h(f"order:{r['id']}"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sheet.jsonl").open("w", encoding="utf-8") as sheet, (out / "key.jsonl").open("w", encoding="utf-8") as key:
        for r in picked:
            question = r["question"].split("Question: ")[-1]
            sheet.write(json.dumps({"id": r["id"], "subset": r["_subset"], "video": r["_video"], "query_time": r["query_time"],
                                    "question": question, "candidates": r["options"][:3],
                                    "status": None, "causes": []}, ensure_ascii=False) + "\n")
            key.write(json.dumps({"id": r["id"], "subset": r["_subset"], "status": r["diagnostics"]["epistemic_status"],
                                  "causes": r["diagnostics"].get("uncertainty_sources") or []}, ensure_ascii=False) + "\n")
    (out / "GUIDE.md").write_text(GUIDE, encoding="utf-8")
    c = Counter((r["_subset"], r["diagnostics"]["epistemic_status"]) for r in picked)
    print(json.dumps({"n": len(picked), "by_subset_status": {f"{a}:{b}": v for (a, b), v in sorted(c.items())}}))


def contact(args) -> None:
    """One image per sampled item: 1 FPS frames from 0 s to query_time as a timestamped grid, plus the last frame enlarged."""
    import subprocess
    import tempfile
    from PIL import Image, ImageDraw

    out = Path(args.out)
    target = out / "contact"
    target.mkdir(parents=True, exist_ok=True)
    for item in read_jsonl(out / "sheet.jsonl"):
        dest = target / f"{item['id']}.jpg"
        if dest.exists():
            continue
        tq = float(item["query_time"])
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["ffmpeg", "-v", "error", "-i", item["video"], "-t", f"{tq + 0.01:.2f}", "-vf", "fps=1",
                            f"{tmp}/f_%03d.jpg"], check=True)
            frames = sorted(Path(tmp).glob("f_*.jpg"))
            images = [Image.open(f).convert("RGB") for f in frames]
        cols, tw = 8, 240
        th = round(tw * images[0].height / images[0].width)
        rows = (len(images) + cols - 1) // cols
        big_w = cols * tw
        big_h = round(big_w * images[-1].height / images[-1].width / 2)
        sheet = Image.new("RGB", (big_w, rows * (th + 18) + big_h + 24), "white")
        draw = ImageDraw.Draw(sheet)
        for i, im in enumerate(images):
            x, y = (i % cols) * tw, (i // cols) * (th + 18)
            sheet.paste(im.resize((tw, th)), (x, y + 18))
            draw.text((x + 4, y + 2), f"t={i}s", fill="black")
        y0 = rows * (th + 18)
        draw.text((4, y0 + 4), f"last frame t={len(images) - 1}s (query time {tq:g}s), enlarged", fill="black")
        sheet.paste(images[-1].resize((big_w // 2, big_h)), (0, y0 + 24))
        sheet.save(dest, quality=82)
    print(f"contact sheets: {len(list(target.glob('*.jpg')))} in {target}")


def cohen(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def fleiss(table: list[list]) -> float:
    cats = sorted({x for row in table for x in row})
    n_raters = len(table[0])
    p_j = {c: sum(row.count(c) for row in table) / (len(table) * n_raters) for c in cats}
    p_i = [(sum(row.count(c) ** 2 for c in cats) - n_raters) / (n_raters * (n_raters - 1)) for row in table]
    pbar, pe = sum(p_i) / len(p_i), sum(v * v for v in p_j.values())
    return 1.0 if pe == 1 else (pbar - pe) / (1 - pe)


def score(args) -> None:
    out = Path(args.out)
    key = {r["id"]: r for r in read_jsonl(out / "key.jsonl")}
    sheets = [{r["id"]: r for r in read_jsonl(Path(p))} for p in args.sheets]
    ids = sorted(set(key).intersection(*map(set, sheets)))
    status = [[s[i]["status"] for s in sheets] for i in ids]
    report = {"n": len(ids), "raters": len(sheets)}
    report["status_cohen_pairwise"] = {f"{a}-{b}": round(cohen([r[a] for r in status], [r[b] for r in status]), 3)
                                       for a, b in combinations(range(len(sheets)), 2)}
    report["status_fleiss"] = round(fleiss(status), 3)
    report["status_raw_agreement"] = round(sum(len(set(r)) == 1 for r in status) / len(ids), 3)
    report["adjudication_rate"] = round(sum(len(set(r)) > 1 for r in status) / len(ids), 3)
    majority = [Counter(r).most_common(1)[0][0] for r in status]
    report["majority_vs_release_status"] = round(sum(m == key[i]["status"] for m, i in zip(majority, ids)) / len(ids), 3)
    report["majority_vs_release_cohen"] = round(cohen(majority, [key[i]["status"] for i in ids]), 3)
    both_unc = [i for i, r in zip(ids, status) if all(x == "uncertain" for x in r)]
    per_cause = {}
    for cause in CAUSES:
        votes = [[cause in (s[i].get("causes") or []) for s in sheets] for i in both_unc]
        if votes and len(sheets) >= 2:
            per_cause[cause] = round(cohen([v[0] for v in votes], [v[1] for v in votes]), 3)
    report["cause_cohen_on_items_all_raters_uncertain"] = per_cause
    report["cause_items"] = len(both_unc)
    by_subset = defaultdict(list)
    for i, r in zip(ids, status):
        by_subset[key[i]["subset"]].append(r)
    report["status_fleiss_by_subset"] = {k: round(fleiss(v), 3) if len(v) > 1 else None for k, v in sorted(by_subset.items())}
    print(json.dumps(report, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--root", default="data/eval_video/RevisitQA")
    s.add_argument("--out", default="data/eval_video/RevisitQA/_annotation")
    s.add_argument("--n", type=int, default=300)
    k = sub.add_parser("contact")
    k.add_argument("--out", default="data/eval_video/RevisitQA/_annotation")
    c = sub.add_parser("score")
    c.add_argument("--out", default="data/eval_video/RevisitQA/_annotation")
    c.add_argument("sheets", nargs="+")
    args = parser.parse_args(argv)
    {"sample": sample, "contact": contact, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
