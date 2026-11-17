#!/usr/bin/env python
"""Add trace-grounded Known items to RevisitQA-Sim VHome so each Uncertain template has same-template Known twins.

Phase 1 (--extract, run where the VirtualHome capture is mounted) writes
scripts/data/revisitqa_sim_traces/vhome_timeline.json: per released episode, the operated objects, their
event list (start/end/before/after state/visibility) and per-second target pixels from the instance masks.
Phase 2 (--dry-run / --in-place) generates Known items whose evidence is fully visible:
  count  "Between t1 and t2 seconds, how many times was the X opened or closed / picked up or put down?"
         X has >= MIN_PIXELS at every sampled second of [t1, t2] and no action straddles the interval.
  state  "For the X identified earlier in the video, what was its physical state at T seconds?"
         X is visible at T, was visible earlier, and no action on X is in progress at T.
  pair   "What were the physical states of the X at t1 seconds and t2 seconds, respectively?"
         both times satisfy the state condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TIMELINE = Path(__file__).resolve().parent / "revisitqa_sim_traces" / "vhome_timeline.json"
MANIFEST = Path("revisitqa_vhome") / "vhome_question_answer.jsonl"
RAW = Path(
    os.environ.get(
        "REVISITQA_VHOME_RAW_EPISODES",
        "data/vhome_raw/episodes",
    )
).expanduser()
MIN_PIXELS = 128
KNOWN_PER_UNCERTAIN = 2
# Released episode -> raw capture (matched by offscreen-event timing, object names and exact video duration).
EPISODE_MAP = {
    "RevisitQA_vhome_01": "scene_0_room_11", "RevisitQA_vhome_02": "scene_0_room_205", "RevisitQA_vhome_03": "scene_0_room_335",
    "RevisitQA_vhome_04": "scene_0_room_73", "RevisitQA_vhome_05": "scene_1_room_11", "RevisitQA_vhome_06": "scene_1_room_126",
    "RevisitQA_vhome_07": "scene_1_room_261", "RevisitQA_vhome_08": "scene_1_room_50", "RevisitQA_vhome_09": "scene_2_room_11",
    "RevisitQA_vhome_10": "scene_2_room_183", "RevisitQA_vhome_11": "scene_2_room_241", "RevisitQA_vhome_12": "scene_2_room_285",
    "RevisitQA_vhome_13": "scene_2_room_346", "RevisitQA_vhome_14": "scene_3_room_11", "RevisitQA_vhome_15": "scene_3_room_172",
    "RevisitQA_vhome_16": "scene_3_room_211", "RevisitQA_vhome_17": "scene_3_room_268", "RevisitQA_vhome_18": "scene_4_room_111",
    "RevisitQA_vhome_19": "scene_4_room_11", "RevisitQA_vhome_20": "scene_4_room_214", "RevisitQA_vhome_21": "scene_4_room_276",
    "RevisitQA_vhome_22": "scene_5_room_11", "RevisitQA_vhome_23": "scene_5_room_162", "RevisitQA_vhome_24": "scene_5_room_247",
    "RevisitQA_vhome_25": "scene_5_room_69", "RevisitQA_vhome_26": "scene_6_room_11", "RevisitQA_vhome_27": "scene_6_room_197",
    "RevisitQA_vhome_28": "scene_6_room_282", "RevisitQA_vhome_29": "scene_6_room_56",
}
BASE_NAMES = {"barsoap": "bar of soap", "waterglass": "drinking glass", "washingsponge": "sponge", "toilet": "toilet lid",
              "tablelamp": "table lamp", "lightswitch": "light switch", "facecream": "face cream", "poundcake": "pound cake",
              "wineglass": "wine glass", "kitchencabinet": "kitchen cabinet", "coffeetable": "coffee table",
              "tv": "television"}


def _h(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


# ------------------------------------------------------------------ phase 1: extraction from the capture
def extract() -> None:
    import numpy as np
    from PIL import Image

    if not RAW.is_dir():
        raise SystemExit(
            "Set REVISITQA_VHOME_RAW_EPISODES to the VirtualHome raw episode directory "
            f"(missing: {RAW})"
        )
    out = {}
    for released, raw_name in EPISODE_MAP.items():
        d = RAW / raw_name
        events = json.loads((d / "events.json").read_text())
        frames = [json.loads(line) for line in (d / "frame_manifest.jsonl").read_text().splitlines() if line.strip()]
        colors = {}
        for f in frames:
            colors.setdefault(f["target_id"], [round(c * 255) for c in f["color"]])
        by_frame = {f["frame"]: f for f in frames}
        duration = len(frames) / 20
        pixels = defaultdict(list)
        for t in range(0, int(math.floor(duration)) + 1):
            f = by_frame.get(20 * t) or by_frame.get(20 * t - 1)
            if f is None:
                for tid in colors:
                    pixels[tid].append(None)
                continue
            mask = np.asarray(Image.open(d / f["mask"]).convert("RGB")).astype(int)
            for tid, color in colors.items():
                pixels[tid].append(int((np.abs(mask - np.array(color)).sum(-1) <= 3).sum()))
        objects = {}
        for e in events:
            obj = objects.setdefault(str(e["object_id"]), {"raw": e["object"], "events": []})
            obj["events"].append({k: e[k] for k in ("verb", "start", "end", "before_state", "after_state", "visibility")})
        for tid, obj in objects.items():
            obj["pixels"] = pixels.get(int(tid), [])
        out[released] = {"raw_episode": raw_name, "duration": duration, "objects": objects}
        print(released, raw_name, {k: (v["raw"], sum(p is not None and p >= MIN_PIXELS for p in v["pixels"])) for k, v in objects.items()}, flush=True)
    TIMELINE.parent.mkdir(parents=True, exist_ok=True)
    TIMELINE.write_text(json.dumps(out, indent=1) + "\n")


# ------------------------------------------------------------------ phase 2: generation
FAMILY = {"open": ("open", "closed"), "closed": ("open", "closed"), "on": ("on", "off"), "off": ("on", "off")}
STATE_TEXT = {"open": "Fully open.", "closed": "Fully closed.", "on": "On.", "off": "Off.",
              "hand": "Held by a person.", "surface": "Resting on a surface.", "floor": "Lying on the floor."}
DISTRACTOR = {"openclose": "Partly open.", "onoff": "Being switched between on and off.", "hold": None}
PAIR_TEXT = {"open": "fully open", "closed": "fully closed", "on": "on", "off": "off",
             "hand": "held by a person", "surface": "resting on a surface", "floor": "lying on the floor"}


def state_key(value: str) -> str:
    value = value.lower()
    if value in ("open", "closed", "on", "off"):
        return value
    if "hand" in value:
        return "hand"
    if "floor" in value:
        return "floor"
    return "surface"


def family(obj: dict) -> str:
    keys = {state_key(e["before_state"]) for e in obj["events"]} | {state_key(e["after_state"]) for e in obj["events"]}
    if keys <= {"open", "closed"}:
        return "openclose"
    if keys <= {"on", "off"}:
        return "onoff"
    return "hold"


def phrase_for(obj: dict, rows: list[dict]) -> str | None:
    base = BASE_NAMES.get(obj["raw"], obj["raw"])
    found = set()
    for r in rows:
        text = r["question"].split("Question: ")[-1].lower()
        # Only state/action templates refer to operated objects, so their phrases identify this object.
        if not re.search(r"physical state|opened or closed|picked up or put down|switched|action caused", text):
            continue
        for m in re.finditer(rf"the ({re.escape(base)}(?: (?:closest to|farthest from) the [a-z]+)?)(?= identified| at | between|[,?.]| was| were| and| return| leave| visible| opened| picked| switched|$)", text):
            found.add(m.group(1).strip())
    return found.pop() if len(found) == 1 else None


def state_at(obj: dict, t: float) -> str | None:
    """State at time t if no action on the object is in progress; None while an action is ongoing."""
    events = sorted(obj["events"], key=lambda e: e["start"])
    current = state_key(events[0]["before_state"])
    for e in events:
        if e["start"] - 0.5 <= t <= e["end"] + 0.5:
            return None
        if e["end"] < t:
            current = state_key(e["after_state"])
    return current


def visible(obj: dict, t: int) -> bool:
    px = obj["pixels"]
    return 0 <= t < len(px) and px[t] is not None and px[t] >= MIN_PIXELS


PAIR_FAMILIES = {
    "onoff": ["On at both times.", "On, then off.", "Off, then on.", "Off at both times."],
    "openclose": ["Fully open at both times.", "Fully open, then fully closed.", "Fully closed, then fully open.", "Fully closed at both times."],
    "hold": ["Held by a person at both times.", "Held by a person, then resting on a surface.",
             "Resting on a surface, then held by a person.", "Resting on a surface at both times."],
}


def pair_subset(combos: list[str], gold: str | None, key: str) -> list[str]:
    """Two (Known) or three (Uncertain) of the four state combinations, chosen by hash, so every 3-subset occurs for both labels."""
    pool = [c for c in combos if c != gold]
    ranked = sorted(pool, key=lambda c: _h(f"{key}:{c}"))
    return ranked[:2] if gold is not None else ranked[:3]


def reoption_pairs(rows: list[dict]) -> int:
    """Re-draw the candidate set of existing two-time state items (Known and Uncertain) with the shared rule."""
    changed = 0
    for r in rows:
        q = r["question"].split("Question: ")[-1]
        if not ("physical states of" in q and "respectively" in q):
            continue
        fam = next((f for f, combos in PAIR_FAMILIES.items() if set(r["options"][:3]) <= set(combos)), None)
        if fam is None:
            raise ValueError(f"{r['id']}: unknown two-time option family {r['options'][:3]}")
        combos = PAIR_FAMILIES[fam]
        key = f"{r['episode_id']}:{q}"
        if r["diagnostics"]["epistemic_status"] == "known":
            gold = r["options"][r["answer_index"]]
            subst = [gold] + pair_subset(combos, gold, key)
        else:
            subst = pair_subset(combos, None, key)
        order = sorted(range(3), key=lambda i: _h(f"{key}:order:{i}"))
        new = [subst[i] for i in order]
        if new != r["options"][:3]:
            prev = r.get("release_meta") or {}
            r["release_meta"] = {**prev, "version": prev.get("version", "text_debias"),
                               "source_options": prev.get("source_options", r["options"]),
                               "edits": prev.get("edits", []) + [{"field": "option", "reason": "shared_two_time_candidate_rule"}]}
            r["options"] = new + r["options"][3:]
            if r["diagnostics"]["epistemic_status"] == "known":
                r["answer_index"] = new.index(subst[0])
            changed += 1
    return changed


def align_uncertain_pair_sets(rows: list[dict]) -> list[str]:
    """Give every Uncertain two-time item a candidate set that some Known item of the same family also uses."""
    is_pair = lambda r: "physical states of" in r["question"] and "respectively" in r["question"]
    family = lambda r: next(f for f, combos in PAIR_FAMILIES.items() if set(r["options"][:3]) <= set(combos))
    known_sets = defaultdict(set)
    for r in rows:
        if is_pair(r) and r["diagnostics"]["epistemic_status"] == "known":
            known_sets[family(r)].add(tuple(sorted(r["options"][:3])))
    changed = []
    for r in rows:
        if not is_pair(r) or r["diagnostics"]["epistemic_status"] != "uncertain":
            continue
        fam = family(r)
        if tuple(sorted(r["options"][:3])) in known_sets[fam]:
            continue
        target = sorted(known_sets[fam], key=lambda t: _h(f"{r['id']}:{t}"))[0]
        order = sorted(range(3), key=lambda i: _h(f"{r['id']}:order:{i}"))
        prev = r.get("release_meta") or {}
        r["release_meta"] = {**prev, "version": prev.get("version", "text_debias"),
                           "source_options": prev.get("source_options", r["options"]),
                           "edits": prev.get("edits", []) + [{"field": "option", "reason": "uncertain_pair_set_seen_in_known"}]}
        r["options"] = [list(target)[i] for i in order] + r["options"][3:]
        changed.append(r["id"])
    return changed


def stem_preamble(rows: list[dict]) -> str:
    stem = next(r["question_stem"] for r in rows if r.get("question_stem"))
    return stem.split("Question: ")[0]


def with_preamble(preamble: str, tq: float, question: str) -> str:
    head = re.sub(r"at or before [0-9.]+ seconds", f"at or before {tq:.2f} seconds", preamble)
    return head + "Question: " + question


def make_row(episode, qtype, subtype, question, substantive, gold_text, tq, span_start, intensity, evidence, preamble):
    from scripts.data.revisitqa_unify_abstention import ABSTENTION, ABSTENTION_INDEX
    key = f"{episode}:{question}:{tq}"
    order = sorted(range(3), key=lambda i: _h(f"{key}:{i}"))
    options = [substantive[i] for i in order]
    return {
        "id": None, "episode_id": episode, "query_time": round(tq, 2), "question_type": qtype,
        "question_subtype": subtype, "video_evidence": evidence, "question": question,
        "question_stem": with_preamble(preamble, tq, question),
        "options": options + [ABSTENTION], "answer_index": options.index(gold_text), "abstention_index": ABSTENTION_INDEX,
        "evidence_spans": [[float(span_start), round(tq, 2)]],
        "diagnostics": {"epistemic_status": "known", "uncertainty_sources": []},
        "diagnostic_rationale": {"volatility": "Operator actions change the object's physical state.",
                                 "uncertainty": "The object is visible at every sampled second the answer depends on."},
        "change_intensity": int(intensity), "qa_schema": "virtualhome_qa290_eval_ready_20260922",
        "release_meta": {"version": "trace_known", "source": "virtualhome_events_and_instance_masks"},
    }


def generate(rows: list[dict], timeline: dict) -> dict[str, list[dict]]:
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r)
    preamble = stem_preamble(rows)
    pools = defaultdict(list)
    for episode, info in sorted(timeline.items()):
        duration = info["duration"]
        starts = sorted(e["start"] for o in info["objects"].values() for e in o["events"])
        intensity = lambda tq: min(10, max(1, sum(s <= tq for s in starts)))
        raw_names = Counter(o["raw"] for o in info["objects"].values())
        for oid, obj in info["objects"].items():
            if raw_names[obj["raw"]] > 1:
                continue  # two objects of one category: the release disambiguates them by phrase we cannot map to ids
            name = phrase_for(obj, by_ep[episode])
            if not name:
                continue
            fam = family(obj)
            last = int(math.floor(duration)) - 1
            vis = [t for t in range(0, last + 1) if visible(obj, t)]
            # --- count: maximal fully-visible runs, then windows inside them
            runs, run = [], []
            for t in range(0, last + 1):
                if visible(obj, t):
                    run.append(t)
                else:
                    if run:
                        runs.append(run)
                    run = []
            if run:
                runs.append(run)
            if fam in ("openclose", "hold"):
                verb = "opened or closed" if fam == "openclose" else "picked up or put down"
                for run in runs:
                    if len(run) < 5:
                        continue
                    t1, t2 = run[0], run[-1]
                    if any(e["start"] - 1 < t1 < e["end"] + 1 or e["start"] - 1 < t2 < e["end"] + 1 for e in obj["events"]):
                        continue
                    inside = [e for e in obj["events"] if t1 <= e["start"] and e["end"] <= t2]
                    # Every counted action must be fully observed; a partly observed action cannot be counted from 1 FPS views.
                    if any(e["visibility"] != "fully_observed" for e in inside):
                        continue
                    n = len(inside)
                    gold = ["Zero times.", "Once.", "Two or more times."][min(n, 2)]
                    tq = t2 + 0.05
                    pools["count"].append(make_row(
                        episode, "history_aggregation", "trace_visible_action_count",
                        f"Between {t1} and {t2} seconds, how many times was the {name} {verb}?",
                        ["Zero times.", "Once.", "Two or more times."], gold, tq, t1, intensity(tq),
                        f"The {name} is visible at every sampled second from {t1} to {t2} s; {n} {verb.split(' or ')[0]}/{verb.split(' or ')[1]} action(s) on it occur in that interval.",
                        preamble))
            # --- state at T (and pairs)
            stable = [t for t in vis if state_at(obj, t) is not None and any(v < t for v in vis)]
            options = ([STATE_TEXT["closed"], STATE_TEXT["open"], DISTRACTOR["openclose"]] if fam == "openclose" else
                       [STATE_TEXT["off"], STATE_TEXT["on"], DISTRACTOR["onoff"]] if fam == "onoff" else
                       [STATE_TEXT["surface"], STATE_TEXT["hand"], STATE_TEXT["floor"]])
            for t in stable:
                gold = STATE_TEXT[state_at(obj, t)]
                if gold not in options:
                    continue
                tq = min(duration - 0.5, t + 1 + _h(f"{episode}:{oid}:{t}") % 8) + 0.05
                tq = math.floor(tq) + 0.05
                first_seen = next(v for v in vis if v < t)
                pools["state"].append(make_row(
                    episode, "current_state", "trace_visible_state",
                    f"For the {name} identified earlier in the video, what was its physical state at {t} seconds?",
                    options, gold, tq, first_seen, intensity(tq),
                    f"The {name} is visible at {t} s with no action on it in progress; its state is {PAIR_TEXT[state_at(obj, t)]}.",
                    preamble))
            for i, t1 in enumerate(stable):
                for t2 in stable[i + 1:]:
                    if not 3 <= t2 - t1 <= 15:
                        continue
                    s1, s2 = state_at(obj, t1), state_at(obj, t2)
                    if fam == "hold":
                        continue  # no Uncertain two-time item uses the holding family; adding Known ones would create a label cue
                    a_key, b_key = ("open", "closed") if fam == "openclose" else ("on", "off") if fam == "onoff" else ("surface", "hand")
                    cap = lambda s: s[0].upper() + s[1:]
                    txt = lambda x, y: cap(f"{PAIR_TEXT[x]} at both times.") if x == y else cap(f"{PAIR_TEXT[x]}, then {PAIR_TEXT[y]}.")
                    combos = [txt(x, y) for x in (a_key, b_key) for y in (a_key, b_key)]
                    gold = txt(s1, s2)
                    others = pair_subset(combos, gold, f"{episode}:{oid}:{t1}:{t2}")
                    tq = t2 + 0.05
                    pools["pair"].append(make_row(
                        episode, "state_change", "trace_visible_state_pair",
                        f"What were the physical states of the {name} at {t1} seconds and {t2} seconds, respectively?",
                        [gold] + others, gold, tq, t1, intensity(tq),
                        f"The {name} is visible at {t1} s ({PAIR_TEXT[s1]}) and at {t2} s ({PAIR_TEXT[s2]}) with no action on it in progress at either time.",
                        preamble))
    return pools


def uncertain_counts(rows: list[dict]) -> dict[str, int]:
    u = [r["question"].split("Question: ")[-1] for r in rows if r["diagnostics"]["epistemic_status"] == "uncertain"]
    k = [r["question"].split("Question: ")[-1] for r in rows if r["diagnostics"]["epistemic_status"] == "known"]
    count = lambda qs, pred: sum(map(pred, qs))
    is_count = lambda q: "how many times was" in q
    is_state = lambda q: q.startswith("For the") or q.startswith("What was the physical state")
    is_pair = lambda q: "physical states of" in q and "respectively" in q
    return {
        "count": max(0, KNOWN_PER_UNCERTAIN * count(u, is_count) - count(k, is_count)),
        "state": max(0, KNOWN_PER_UNCERTAIN * count(u, is_state) - count(k, is_state)),
        "pair": max(0, KNOWN_PER_UNCERTAIN * count(u, is_pair) - count(k, is_pair)),
    }


def select(pools: dict[str, list[dict]], need: dict[str, int]) -> list[dict]:
    chosen = []
    for kind, n in need.items():
        pool = sorted(pools.get(kind, []), key=lambda r: _h(r["question_stem"]))
        if kind == "count":
            # Fully observed actions are rare; cap "Zero times." so the count prior cannot answer Known items.
            # Match the Uncertain count items one-to-one so the template carries no label prior; a blind
            # "Zero times." policy then scores <= 0 under effective reliability.
            n = n // KNOWN_PER_UNCERTAIN
            nonzero = [r for r in pool if r["options"][r["answer_index"]] != "Zero times."]
            zero = [r for r in pool if r["options"][r["answer_index"]] == "Zero times."]
            pool = nonzero + zero[: max(0, n - len(nonzero))]
            n = min(n, len(pool))
            chosen.extend(pool[:n])
            continue
        by_gold = defaultdict(list)
        for r in pool:
            by_gold[r["options"][r["answer_index"]]].append(r)
        per_ep = Counter()
        picked = []
        # round-robin over gold answers, at most 2 items per episode per kind
        while len(picked) < n and any(by_gold.values()):
            for gold in sorted(by_gold):
                while by_gold[gold] and per_ep[by_gold[gold][0]["episode_id"]] >= 2:
                    by_gold[gold].pop(0)
                if by_gold[gold] and len(picked) < n:
                    r = by_gold[gold].pop(0)
                    picked.append(r)
                    per_ep[r["episode_id"]] += 1
        chosen.extend(picked)
    return chosen


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extract", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--in-place", action="store_true")
    parser.add_argument("--source", default=None, help="Pre-trace VHome manifest to rebuild from.")
    args = parser.parse_args(argv)
    if args.extract:
        extract()
        return
    from scripts.data.revisitqa_unify_abstention import check_rows, read_jsonl, write_jsonl

    path = Path(args.root) / MANIFEST
    rows = read_jsonl(Path(args.source) if args.source else path)
    if any((r.get("release_meta") or {}).get("version") == "trace_known" for r in rows):
        raise RuntimeError("trace Known items already added; pass --source with the pre-trace manifest")
    timeline = json.loads(TIMELINE.read_text())
    # "What action caused the change" is Uncertain-only: every change in VirtualHome is operated by a person,
    # so a Known twin would always have the same answer. Drop the template instead of leaving a label cue.
    dropped_cause = [r["id"] for r in rows if "what action caused" in r["question"]]
    rows = [r for r in rows if r["id"] not in set(dropped_cause)]
    reoptioned = reoption_pairs(rows)
    pools = generate(rows, timeline)
    need = uncertain_counts(rows)
    chosen = select(pools, need)
    last = defaultdict(int)
    for r in rows:
        last[r["episode_id"]] = max(last[r["episode_id"]], int(r["id"].rsplit("_q", 1)[1]))
    for r in sorted(chosen, key=lambda r: (r["episode_id"], r["query_time"])):
        last[r["episode_id"]] += 1
        r["id"] = f"{r['episode_id']}_q{last[r['episode_id']]:02d}"
    out = rows + chosen
    aligned = align_uncertain_pair_sets(out)
    report = check_rows("vhome", out)
    print(json.dumps({"aligned_uncertain_pair_sets": aligned, "dropped_cause_items": dropped_cause, "reoptioned_existing_pairs": reoptioned, "need": need, "pool": {k: len(v) for k, v in pools.items()}, "added": len(chosen),
                      "gold_dist": {k: Counter(r["options"][r["answer_index"]] for r in chosen if r["question_subtype"].startswith(f"trace_visible_{k}")) for k in ("action_count", "state_v30", "state_pair")},
                      "n": len(out), "errors": report["errors"][:5], "auroc_q": report["text_status_auroc_question"],
                      "auroc_qo": report["text_status_auroc_question_options"], "blind_er": report["blind_agent_best_effective_reliability"]}, default=dict))
    for r in chosen[:2] + chosen[-2:]:
        print(" ", r["id"], r["query_time"], r["question"], r["options"][:3], "gold", r["answer_index"], "|", r["video_evidence"])
    if args.in_place:
        if report["error_count"]:
            raise RuntimeError(report["errors"])
        backup = Path(args.root) / f"_backup_pre_vhome_{datetime.now():%Y%m%d_%H%M%S}" / MANIFEST.parent
        backup.mkdir(parents=True)
        shutil.copy2(path, backup / MANIFEST.name)
        write_jsonl(path, out)
        print(f"backup: {backup}")


if __name__ == "__main__":
    main()
