#!/usr/bin/env python
"""Rebuild RevisitQA-Sim THOR Uncertain items as query-time pairs: same question, same options, different evidence.

THOR events are off-camera disable/enable interventions (scripts/data/revisitqa_sim_traces/thor_timeline.json,
extracted from the AI2-THOR rollouts' execution_trace.json / observed_events.json); every event is observed
as a before/after visibility change. The original THOR Uncertain templates (instance identity, off-screen
manipulation counts and orders, movement cause, location while absent) can only ever be Uncertain, so their
wording alone reveals the label. This script replaces them with one template whose label depends only on
what the causal prefix shows:

  "Which happens first: the X reappears at its original position or the Y reappears at its original position?"
  options: X first / Y first / both at the same moment (+ the shared abstention option)

Both X and Y have visibly disappeared before the query. The order is Known only when the 1 FPS frames up to the
query settle it: some frame shows the first object back at its spot, and a frame at or after that one shows the
other object's spot still empty. The camera plan (revisitqa_sim_traces/thor_focus.json, from
revisitqa_thor_focus_extract.py) only approximates which spots are in view: its focus lists include camera travel at
phase boundaries. It is used to pre-select candidates; items where at least one object has reappeared are verified
on the frames in the blind relabel before release (`--write` exports oversampled candidates for that). Otherwise the item is Uncertain: neither has reappeared, or one has
reappeared but the other's spot is not seen afterwards, so it may have returned earlier while the camera was away.
Queries within CONFIRM_SECONDS of any relevant event are skipped.

Usage:
  python scripts/data/revisitqa_sim_known_from_traces.py --source <pre-trace THOR manifest> --dry-run
  python scripts/data/revisitqa_sim_known_from_traces.py --source <pre-trace THOR manifest> --in-place
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import ABSTENTION, ABSTENTION_INDEX, check_rows, read_jsonl, write_jsonl  # noqa: E402

TRACE = Path(__file__).resolve().parent / "revisitqa_sim_traces" / "thor_timeline.json"
FOCUS = Path(__file__).resolve().parent / "revisitqa_sim_traces" / "thor_focus.json"
MANIFEST = Path("revisitqa_thor") / "thor_question_answer.jsonl"
CONFIRM_SECONDS = 2
MIN_EVENT_GAP = 3
KNOWN_PER_UNCERTAIN = 2
SUBTYPE = "query_time_reappearance_order_v31"
NAMES = {
    "CellPhone": "cell phone", "RemoteControl": "remote control", "ScrubBrush": "scrub brush", "SoapBar": "soap bar",
    "SoapBottle": "soap bottle", "HandTowel": "hand towel", "TissueBox": "tissue box", "KeyChain": "key chain",
    "CreditCard": "credit card", "SprayBottle": "spray bottle", "TeddyBear": "teddy bear", "AlarmClock": "alarm clock",
    "ToiletPaper": "toilet paper", "DishSponge": "dish sponge", "PepperShaker": "pepper shaker", "SaltShaker": "salt shaker",
    "WineBottle": "wine bottle", "ButterKnife": "butter knife", "BaseballBat": "baseball bat", "BasketBall": "basketball",
    "TennisRacket": "tennis racket",
}


def name(obj: str) -> str:
    return NAMES.get(obj, re.sub(r"(?<!^)(?=[A-Z])", " ", obj).lower())


def _h(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def view_seconds(spans: list[list[float]]) -> list[int]:
    """Integer seconds (1 FPS sample times) at which an object's original spot is in camera view."""
    return sorted({s for a, b in spans for s in range(math.ceil(a - 1e-6), math.floor(b + 1e-6) + 1)})


def decided_order(views: dict, x: str, y: str, cx: dict, cy: dict, tq: float):
    """Return (first, seen_back_at, seen_empty_at) if the frames up to tq fix the order, else None."""
    for a, b, ca, cb in ((x, y, cx, cy), (y, x, cy, cx)):
        back = [s for s in views[a] if ca["back"] <= s <= tq]
        if not back:
            continue
        empty = [s for s in views[b] if back[0] <= s < cb["back"] and s <= tq]
        if empty:
            return a, back[0], empty[-1]
    return None


def candidates(timeline: dict, horizon: dict[str, float], focus: dict) -> list[dict]:
    out = []
    for episode, info in sorted(timeline.items()):
        events = info["events"]
        if not all(e["observed"] for e in events):
            continue
        # Only objects with exactly one disappearance followed by one reappearance: "reappears" then names a
        # unique event, so the question needs no time anchor (a time in the stem would itself leak the label).
        per_obj = defaultdict(list)
        for e in events:
            per_obj[e["obj"]].append(e)
        cycles = {}
        for obj, evs in per_obj.items():
            kinds = [e["type"] for e in sorted(evs, key=lambda e: e["t"])]
            if kinds == ["disappear", "appear"]:
                gone, back = sorted(e["t"] for e in evs)
                cycles[obj] = {"gone": gone, "back": back}
        if len({name(o) for o in cycles}) != len(cycles):
            continue
        for x, y in combinations(sorted(cycles), 2):
            cx, cy = cycles[x], cycles[y]
            if abs(cx["back"] - cy["back"]) < MIN_EVENT_GAP:
                continue
            relevant = [cx["gone"], cy["gone"], cx["back"], cy["back"]]
            for tq in range(math.ceil(max(cx["gone"], cy["gone"])) + CONFIRM_SECONDS, int(horizon[episode]) + 1):
                if any(t <= tq < t + CONFIRM_SECONDS for t in relevant):
                    continue
                views = {o: view_seconds(focus[episode][o]) for o in (x, y)}
                decided = decided_order(views, x, y, cx, cy, tq)
                status = "known" if decided else "uncertain"
                out.append({"episode": episode, "x": x, "y": y, "tq": float(tq), "status": status,
                            "first": decided[0] if decided else None, "seen": decided[1:] if decided else None,
                            "cx": cx, "cy": cy, "intensity": min(10, max(1, sum(e["t"] <= tq for e in events)))})
    return out


def to_row(c: dict) -> dict:
    a, b = (c["x"], c["y"]) if _h(f"{c['episode']}:{c['x']}:{c['y']}") % 2 else (c["y"], c["x"])
    na, nb = name(a), name(b)
    question = (f"Which happens first: the {na} reappears at its original position "
                f"or the {nb} reappears at its original position?")
    substantive = [f"The {na} reappears first", f"The {nb} reappears first", "Both reappear at the same moment"]
    order = sorted(range(3), key=lambda i: _h(f"{c['episode']}:{question}:{c['tq']}:{i}"))
    options = [substantive[i] for i in order]
    ca, cb = (c["cx"], c["cy"]) if a == c["x"] else (c["cy"], c["cx"])
    start = math.floor(min(ca["gone"], cb["gone"]))
    if c["status"] == "known":
        first = c["first"]
        other = c["y"] if first == c["x"] else c["x"]
        gold = options.index(f"The {name(first)} reappears first")
        evidence = (f"The {na} disappears at about {ca['gone']:.0f} s and the {nb} at about {cb['gone']:.0f} s; "
                    f"the {name(first)} is seen back at its spot at {c['seen'][0]} s, and the {name(other)}'s spot "
                    f"is still empty at {c['seen'][1]} s.")
        diagnostics = {"epistemic_status": "known", "uncertainty_sources": []}
        rationale = "One object is seen back while the other's spot is later seen still empty, which fixes the order."
    else:
        gold = ABSTENTION_INDEX
        back = [f"the {n} is back by {t['back']:.0f} s" for n, t in ((na, ca), (nb, cb)) if t["back"] < c["tq"]]
        evidence = (f"The {na} disappears at about {ca['gone']:.0f} s and the {nb} at about {cb['gone']:.0f} s; "
                    + ("neither has reappeared by the query time." if not back else
                       "; ".join(back) + ", but no later frame before the query shows the other's spot still empty, "
                       "so it may have returned earlier while the camera was away."))
        diagnostics = {"epistemic_status": "uncertain", "uncertainty_sources": ["missing_observation"]}
        rationale = "The frames up to the query do not fix which object returned first."
    return {
        "id": None, "episode_id": c["episode"], "query_time": c["tq"], "question_type": "temporal_reasoning",
        "question_subtype": SUBTYPE, "video_evidence": evidence, "question": question,
        "options": options + [ABSTENTION], "answer_index": gold, "abstention_index": ABSTENTION_INDEX,
        "evidence_spans": [[float(start), c["tq"]]], "diagnostics": diagnostics,
        "diagnostic_rationale": {"volatility": "The answer depends on visibility changes recorded in the simulator trace.",
                                 "uncertainty": rationale},
        "change_intensity": c["intensity"],
        "release_meta": {"version": "query_time_pairs", "source": "ai2thor_execution_trace"},
    }


TIME_BINS = list(range(32, 50, 2))  # 2 s query-time bins where both labels are available across many episodes


def select(cands: list[dict], n_uncertain: int, oversample: float = 1.0) -> list[dict]:
    """Equal Uncertain share in every 2 s query-time bin, so prefix length carries no label information.

    Within a bin the same cut point therefore occurs with and without the decisive reappearance (across videos).
    Known items prefer object pairs that also appear as Uncertain items (same question, different evidence).
    Each object pair is used at most once per label.
    """
    pair = lambda c: (c["episode"], c["x"], c["y"])
    bin_of = lambda c: int(c["tq"]) // 2 * 2
    n_uncertain = round(n_uncertain * oversample)
    per_bin_u = [n_uncertain // len(TIME_BINS) + (i < n_uncertain % len(TIME_BINS)) for i in range(len(TIME_BINS))]

    def take(pool, n, exclude: set) -> list[dict]:
        picked, eps = [], Counter()
        ranked = sorted(pool, key=lambda c: _h(f"{c['episode']}:{c['x']}:{c['y']}:{c['tq']}"))
        while len(picked) < n:
            progress = False
            floor = min((eps[c["episode"]] for c in ranked), default=0)
            for c in ranked:
                if len(picked) >= n:
                    break
                if pair(c) in exclude or eps[c["episode"]] > floor:
                    continue
                picked.append(c)
                exclude.add(pair(c))
                eps[c["episode"]] += 1
                progress = True
            if not progress:
                if all(pair(c) in exclude for c in ranked):
                    break
                floor = max(eps.values(), default=0)
                ranked = [c for c in ranked if pair(c) not in exclude]
                if not ranked:
                    break
                for c in ranked:
                    if len(picked) >= n:
                        break
                    if pair(c) not in exclude:
                        picked.append(c)
                        exclude.add(pair(c))
                break
        return picked

    unc, known = [], []
    u_used: set = set()
    for b, n_u in zip(TIME_BINS, per_bin_u):
        unc += take([c for c in cands if bin_of(c) == b and c["status"] == "uncertain"], n_u, u_used)
    k_used: set = set()
    for b, n_u in zip(TIME_BINS, per_bin_u):
        pool = [c for c in cands if bin_of(c) == b and c["status"] == "known"]
        twins = take([c for c in pool if pair(c) in u_used], KNOWN_PER_UNCERTAIN * n_u, k_used)
        known += twins + take(pool, KNOWN_PER_UNCERTAIN * n_u - len(twins), k_used)
    return unc + known


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    parser.add_argument("--source", help="THOR manifest before any trace-based items were added.")
    parser.add_argument("--replace", action="store_true",
                        help="Use the current manifest under --root as source and regenerate its query-time pair items "
                             "(as many Uncertain items as it had); new ids use a known-from-trace tag.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    mode.add_argument("--write", help="Write oversampled candidate rows to this JSONL for frame verification.")
    parser.add_argument("--oversample", type=float, default=1.6)
    args = parser.parse_args(argv)
    if args.replace == bool(args.source):
        parser.error("give exactly one of --source and --replace")
    tag = "trace" if args.replace else "trace_extra"
    if args.replace:
        current = read_jsonl(Path(args.root) / MANIFEST)
        old_pairs = [r for r in current if r.get("question_subtype") == SUBTYPE]
        source = [r for r in current if r.get("question_subtype") != SUBTYPE]
        n_uncertain = sum(r["diagnostics"]["epistemic_status"] == "uncertain" for r in old_pairs)
        horizon_rows = current
    else:
        source = read_jsonl(Path(args.source))
        n_uncertain = None
        horizon_rows = source
    if not args.replace and any((r.get("release_meta") or {}).get("version", "").startswith(("trace", "query_time")) for r in source):
        raise RuntimeError("--source must predate trace-based THOR items")
    template_uncertain = [r for r in source if r["diagnostics"]["epistemic_status"] == "uncertain"]
    kept = [r for r in source if r["diagnostics"]["epistemic_status"] == "known"]
    if n_uncertain is None:
        n_uncertain = len(template_uncertain)
    horizon = defaultdict(float)
    for r in horizon_rows:
        horizon[r["episode_id"]] = max(horizon[r["episode_id"]], float(r["query_time"]), 59.0)
    timeline = json.loads(TRACE.read_text())
    focus = json.loads(FOCUS.read_text())
    cands = candidates(timeline, horizon, focus)
    chosen = [to_row(c) for c in select(cands, n_uncertain, args.oversample if args.write else 1.0)]
    last = defaultdict(int)
    for r in sorted(chosen, key=lambda r: (r["episode_id"], r["query_time"], r["question"])):
        last[r["episode_id"]] += 1
        r["id"] = f"{r['episode_id']}_{tag}_q{last[r['episode_id']]:02d}"
    out = kept + chosen
    report = check_rows("thor", out)
    known_new = [r for r in chosen if r["diagnostics"]["epistemic_status"] == "known"]
    print(json.dumps({
        "candidates": Counter(c["status"] for c in cands),
        "removed_template_uncertain": len(template_uncertain), "kept_known": len(kept),
        "added": Counter(r["diagnostics"]["epistemic_status"] for r in chosen),
        "known_gold_position": Counter(r["answer_index"] for r in known_new),
        "query_time_by_status": {st: sorted(r["query_time"] for r in chosen if r["diagnostics"]["epistemic_status"] == st)[::11]
                                 for st in ("known", "uncertain")},
        "episodes_with_uncertain": len({r["episode_id"] for r in chosen if r["diagnostics"]["epistemic_status"] == "uncertain"}),
        "n": len(out), "errors": report["errors"][:5], "auroc_q": report["text_status_auroc_question"],
        "auroc_qo": report["text_status_auroc_question_options"], "blind_er": report["blind_agent_best_effective_reliability"],
    }, default=dict))
    for r in chosen[:2] + known_new[:2]:
        print(" ", r["id"], r["query_time"], r["diagnostics"]["epistemic_status"], r["question"], r["options"], r["answer_index"], "|", r["video_evidence"])
    if args.write:
        for r in chosen:
            r["needs_frame_check"] = "neither has reappeared" not in r["video_evidence"]
        write_jsonl(Path(args.write), chosen)
        print(f"candidates: {args.write} ({sum(r['needs_frame_check'] for r in chosen)} need a frame check)")
    if args.in_place:
        if report["error_count"]:
            raise RuntimeError(report["errors"])
        path = Path(args.root) / MANIFEST
        backup = Path(args.root) / f"_backup_pre_thor_{tag}_{datetime.now():%Y%m%d_%H%M%S}" / MANIFEST.parent
        backup.mkdir(parents=True)
        shutil.copy2(path, backup / MANIFEST.name)
        write_jsonl(path, out)
        print(f"backup: {backup}")


if __name__ == "__main__":
    main()
