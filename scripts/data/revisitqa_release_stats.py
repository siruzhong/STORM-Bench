#!/usr/bin/env python
"""Release statistics for the RevisitQA paper tables (dataset overview, composition, evidence spans).

Writes data/summary.json and prints the per-subset rows.
Definitions:

  intensity bins L/M/H = change_intensity 1-3 / 4-6 / 7-10;
  query position = mean of query_time / episode duration;
  evidence ends = share of items whose latest evidence endpoint equals query_time;
  evidence before last 16 frames = share of items with a span starting before the latest 16 1 FPS frames up to t_q;
  instant spans = share of spans with start == end; mean span duration over non-instant spans;
  items > 10 s = share of items with at least one span longer than 10 s.
Usage: python scripts/data/revisitqa_release_stats.py [--root data/eval_video/RevisitQA]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import SUBSET_FILES, read_jsonl  # noqa: E402
from scripts.eval.data_io import _revisitqa_video_path  # noqa: E402
from scripts.eval.revisitqa_reliability import evidence_window  # noqa: E402

TYPES = ["current_state", "factual_retrieval", "history_aggregation", "state_change", "object_tracking", "temporal_reasoning"]
CAUSES = ["missing_observation", "partial_observation", "ambiguous_evidence", "low_visual_quality",
          "multiple_candidates", "ambiguous_attribute"]


def duration(path: Path) -> float:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return len(reader) / float(reader.get_avg_fps())


def subset_stats(subset: str, root: Path) -> dict:
    directory, filename = SUBSET_FILES[subset]
    rows = read_jsonl(root / directory / filename)
    episodes = sorted({r["episode_id"] for r in rows})
    dur = {e: duration(root / directory / "video" / _revisitqa_video_path(e)[0]) for e in episodes}
    status = lambda r: r["diagnostics"]["epistemic_status"]
    known = [r for r in rows if status(r) == "known"]
    unc = [r for r in rows if status(r) == "uncertain"]
    ints = [int(r["change_intensity"]) for r in rows]
    spans = [s for r in rows for s in r.get("evidence_spans") or []]
    long_spans = [e - s for s, e in spans if e > s]
    out = {
        "qa": len(rows), "episodes": len(episodes), "hours": sum(dur.values()) / 3600,
        "mean_duration_s": mean(dur.values()), "qa_per_episode": len(rows) / len(episodes),
        "known": len(known), "uncertain": len(unc),
        "intensity_bins": [sum(1 <= i <= 3 for i in ints), sum(4 <= i <= 6 for i in ints), sum(7 <= i <= 10 for i in ints)],
        "question_types": {t: sum(r["question_type"] == t for r in rows) for t in TYPES},
        "question_types_known_uncertain": {t: [sum(r["question_type"] == t for r in known), sum(r["question_type"] == t for r in unc)] for t in TYPES},
        "causes_on_uncertain": {c: sum(c in (r["diagnostics"].get("uncertainty_sources") or []) for r in unc) for c in CAUSES},
        "intensity_mean": mean(ints), "intensity_mean_known": mean(int(r["change_intensity"]) for r in known),
        "intensity_mean_uncertain": mean(int(r["change_intensity"]) for r in unc) if unc else None,
        "query_time_mean": mean(float(r["query_time"]) for r in rows),
        "query_position_mean": mean(float(r["query_time"]) / dur[r["episode_id"]] for r in rows),
        "evidence_ends_at_query": mean(abs(max(e for _, e in r["evidence_spans"]) - float(r["query_time"])) < 1e-6 for r in rows if r.get("evidence_spans")),
        "evidence_before_last16": mean(evidence_window(r) == "earlier" for r in rows if r.get("evidence_spans")),
        "evidence_before_last16_known": mean(evidence_window(r) == "earlier" for r in known if r.get("evidence_spans")),
        "spans_per_item": len(spans) / len(rows),
        "instant_span_share": sum(e == s for s, e in spans) / len(spans),
        "mean_span_duration_s": mean(long_spans) if long_spans else 0.0,
        "items_with_span_over_10s": mean(any(e - s > 10 for s, e in r.get("evidence_spans") or []) for r in rows),
        "all_spans_before_query": all(e <= float(r["query_time"]) + 1e-6 for r in rows for _, e in r.get("evidence_spans") or []),
    }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    parser.add_argument("--out", default="data/summary.json")
    args = parser.parse_args(argv)
    stats = {s: subset_stats(s, Path(args.root)) for s in SUBSET_FILES}
    total = {k: sum(v[k] for v in stats.values()) for k in ("qa", "episodes", "hours", "known", "uncertain")}
    total["intensity_bins"] = [sum(v["intensity_bins"][i] for v in stats.values()) for i in range(3)]
    total["question_types"] = {t: sum(v["question_types"][t] for v in stats.values()) for t in TYPES}
    total["causes_on_uncertain"] = {c: sum(v["causes_on_uncertain"][c] for v in stats.values()) for c in CAUSES}
    total["mean_duration_s"] = total["hours"] * 3600 / total["episodes"]
    total["qa_per_episode"] = total["qa"] / total["episodes"]
    total["qa_per_minute"] = total["qa"] / (total["hours"] * 60)
    stats["total"] = total
    Path(args.out).write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    for s, v in stats.items():
        print(s, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items() if not isinstance(x, dict)})


if __name__ == "__main__":
    main()
