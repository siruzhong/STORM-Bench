#!/usr/bin/env python
"""Build a blind relabeling package for RevisitQA (no released labels inside the item files).

For every item of the selected subsets the package holds
  items.jsonl                       id, subset, query_time, question, three substantive candidates (A-C)
  frames/<id>/timeline.jpg          1 FPS thumbnails from 0 s to t_q with timestamps (event order)
  frames/<id>/full_<t>.jpg          original-resolution frames at t_q-3 .. t_q
  frames/<id>/eval_tq.png           the t_q frame resized as the evaluation does (200,704-pixel budget)
  RUBRIC.md                         copy of scripts/data/revisitqa_label_rubric.md
  key.jsonl                         released status/causes/answer (keep away from annotators)

Usage:
  python scripts/data/revisitqa_relabel_package.py --subsets cook,bike,healthy,music,sports \
      --out outputs/revisitqa/relabel --workers 16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data.revisitqa_unify_abstention import ABSTENTION_INDEX, SUBSET_FILES, read_jsonl  # noqa: E402

ROOT = REPO / "data/eval_video/RevisitQA"
MAX_PIXELS = 200704


def _episode(task):
    subset, episode, items, out = task
    from PIL import Image, ImageDraw

    from qwen_vl_utils.vision_process import smart_resize
    from scripts.eval.data_io import _revisitqa_video_path
    from scripts.eval.revisitqa_streaming import IncrementalVideoSource

    directory = SUBSET_FILES[subset][0]
    video = ROOT / directory / "video" / _revisitqa_video_path(episode)[0]
    frames = IncrementalVideoSource(video, 1.0).take_all()
    done = 0
    for item in items:
        tq = float(item["query_time"])
        visible = [f for f in frames if f.timestamp <= tq + 1e-6]
        target = Path(out) / "frames" / item["id"]
        target.mkdir(parents=True, exist_ok=True)
        cols, tw = (6, 320) if subset in ("thor", "vhome") else (10, 160)  # sim objects are small
        th = round(tw * visible[0].image.height / visible[0].image.width)
        rows = (len(visible) + cols - 1) // cols
        grid = Image.new("RGB", (cols * tw, rows * (th + 14)), "white")
        draw = ImageDraw.Draw(grid)
        for i, frame in enumerate(visible):
            x, y = (i % cols) * tw, (i // cols) * (th + 14)
            grid.paste(frame.image.resize((tw, th)), (x, y + 14))
            draw.text((x + 3, y + 1), f"t={frame.timestamp:.0f}s", fill="black")
        grid.save(target / "timeline.jpg", quality=85)
        for frame in visible[-4:]:
            frame.image.save(target / f"full_{frame.timestamp:.0f}s.jpg", quality=90)
        last = visible[-1].image
        h, w = smart_resize(last.height, last.width, factor=28, min_pixels=128 * 28 * 28, max_pixels=MAX_PIXELS)
        last.resize((w, h), Image.BICUBIC).save(target / "eval_tq.png")
        done += 1
    return done


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", default="cook,bike,healthy,music,sports")
    parser.add_argument("--out", default="outputs/revisitqa/relabel")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--manifest", help="Package rows from this JSONL instead of the release (e.g. THOR candidates).")
    parser.add_argument("--subset-label", default="thor", help="Subset name for --manifest rows.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / "scripts/data/revisitqa_label_rubric.md", out / "RUBRIC.md")
    items, keys, tasks = [], [], {}
    sources = ([(args.subset_label, Path(args.manifest))] if args.manifest else
               [(s, ROOT / SUBSET_FILES[s][0] / SUBSET_FILES[s][1]) for s in args.subsets.split(",")])
    for subset, manifest in sources:
        for row in read_jsonl(manifest):
            question = row["question"].split("Question: ")[-1]
            items.append({"id": row["id"], "subset": subset, "query_time": row["query_time"], "question": question,
                          "candidates": {chr(65 + i): row["options"][i] for i in range(ABSTENTION_INDEX)},
                          "frames": f"frames/{row['id']}/"})
            gold = row["answer_index"]
            keys.append({"id": row["id"], "subset": subset, "status": row["diagnostics"]["epistemic_status"],
                         "causes": row["diagnostics"].get("uncertainty_sources") or [],
                         "choice": None if gold == ABSTENTION_INDEX else chr(65 + gold)})
            tasks.setdefault((subset, row["episode_id"]), []).append(row)
    (out / "items.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    (out / "key.jsonl").write_text("".join(json.dumps(k, ensure_ascii=False) + "\n" for k in keys), encoding="utf-8")
    jobs = [(s, e, rows, str(out)) for (s, e), rows in tasks.items()]
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n in pool.map(_episode, jobs):
            done += n
    print(json.dumps({"items": len(items), "episodes": len(jobs), "frames_written_for": done, "out": str(out)}))


if __name__ == "__main__":
    main()
