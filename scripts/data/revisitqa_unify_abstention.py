#!/usr/bin/env python
"""Rebuild RevisitQA options so every item has three substantive choices plus one fixed abstention option.

Usage:
  python scripts/data/revisitqa_unify_abstention.py --root data/eval_video/RevisitQA --out-root /tmp/revisitqa_v2   # dry run
  python scripts/data/revisitqa_unify_abstention.py --root data/eval_video/RevisitQA --in-place              # backup + overwrite
  python scripts/data/revisitqa_unify_abstention.py --root data/eval_video/RevisitQA --check                 # validate + leak audit
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
from pathlib import Path

ABSTENTION = "Cannot be determined from the video up to this point."
ABSTENTION_INDEX = 3
VERSION = "shared_abstention"
SUBSET_FILES = {
    "cook": ("revisitqa_cook", "cook_question_answer.jsonl"),
    "bike": ("revisitqa_bike", "bike_question_answer.jsonl"),
    "healthy": ("revisitqa_healthy", "health_question_answer.jsonl"),
    "music": ("revisitqa_music", "music_question_answer.jsonl"),
    "sports": ("revisitqa_sports", "sports_question_answer.jsonl"),
    "thor": ("revisitqa_thor", "thor_question_answer.jsonl"),
    "vhome": ("revisitqa_vhome", "vhome_question_answer.jsonl"),
}
# Options that state the evidence is insufficient rather than describing a state of the world.
META = re.compile(
    r"(cannot be (determined|established|resolved|confirmed|inferred|told|identified)|can't be determined"
    r"|cannot (determine|tell|confirm)|not enough (visual )?evidence|enough (identity cues|discriminative evidence)"
    r"|insufficient|uncertain|unreadable|undetermin|not determinable"
    r"|do(es)? not (establish|reveal|uniquely|determine|allow|confirm|provide|preserve)|no definite|no unique"
    r"|no concrete alternative|none of the concrete|remains? (ambiguous|unresolved|unknown|unclear)"
    r"|ambiguous (in|across)|leaves? .*unresolved|absent from the visual record|not (be |visibly )?established"
    r"|not visible enough|contents not visible|not observed( clearly enough)?|lacks enough"
    r"|cannot be recovered|cannot rule out|not sufficiently observed|not shown in the available)",
    re.I,
)
# Known items whose gold answer itself is a non-determinability statement conflict with the fixed abstention option.
DROP_IDS = {"RevisitQA_cook_118_q10"}


def _hash(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def _length_rank(option: str, group: list[str]) -> int:
    ordered = sorted(group, key=lambda text: (len(text), text))
    return ordered.index(option)


def rebuild_rows(subset: str, rows: list[dict]) -> tuple[list[dict], Counter]:
    stats = Counter()
    out = []
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["id"] in DROP_IDS:
            stats["dropped_item"] += 1
            continue
        strata[(row["diagnostics"]["epistemic_status"], row.get("question_type"))].append(row)
    for (status, qtype), group in sorted(strata.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        group.sort(key=lambda row: row["id"])
        for k, row in enumerate(group):
            options = [str(option).strip() for option in row["options"]]
            gold = int(row["answer_index"])
            meta = [i for i, option in enumerate(options) if option == ABSTENTION or META.search(option)]
            dropped, reason = None, None
            if status == "uncertain":
                if gold not in meta:
                    # The Uncertain gold is the abstention choice by construction; log unusual wording for review.
                    stats["uncertain_gold_unmatched_wording"] += 1
                    print(f"[review] {row['id']}: Uncertain gold wording {options[gold]!r}", file=sys.stderr)
                extra = [i for i in meta if i != gold]
                if extra:
                    raise ValueError(f"{row['id']}: Uncertain item keeps extra abstention-type options {extra}")
                substantive = [option for i, option in enumerate(options) if i != gold]
                dropped, reason = options[gold], "uncertain_gold_replaced_by_fixed_abstention"
                new_gold = ABSTENTION_INDEX
                target_pos = target_rank = None
            else:
                if gold in meta:
                    raise ValueError(f"{row['id']}: Known gold is an abstention-type option: {options[gold]!r}")
                distractors = [i for i in range(len(options)) if i != gold]
                if len(options) != 4:
                    raise ValueError(f"{row['id']}: expected 4 options, got {len(options)}")
                target_pos, target_rank = k % 3, (k // 3) % 3
                meta_distractors = [i for i in distractors if i in meta]
                if len(meta_distractors) > 1:
                    raise ValueError(f"{row['id']}: Known item has {len(meta_distractors)} abstention-type distractors")
                if meta_distractors:
                    drop = meta_distractors[0]
                    reason = "known_abstention_type_distractor"
                else:
                    def cost(index: int) -> tuple:
                        kept = [options[j] for j in range(4) if j != index]
                        return (abs(_length_rank(options[gold], kept) - target_rank), _hash(f"{row['id']}:{index}"))
                    drop = min(distractors, key=cost)
                    reason = "known_length_rank_balanced_drop"
                dropped = options[drop]
                kept_distractors = [options[i] for i in distractors if i != drop]
                substantive = list(kept_distractors)
                substantive.insert(target_pos, options[gold])
                new_gold = target_pos
            if len(substantive) != 3 or len({option.lower() for option in substantive}) != 3:
                raise ValueError(f"{row['id']}: substantive options are not 3 unique strings: {substantive}")
            new_row = dict(row)
            new_row["options"] = substantive + [ABSTENTION]
            new_row["answer_index"] = new_gold
            new_row["abstention_index"] = ABSTENTION_INDEX
            new_row["option_balance"] = {
                "method": f"{VERSION}_position_and_length_rank_balanced",
                "stratum": f"{subset}:{qtype}:{status}",
                "target_position": target_pos,
                "target_length_rank": target_rank,
            }
            new_row["release_meta"] = {
                "version": VERSION,
                "source_options": row["options"],
                "source_answer_index": gold,
                "dropped_option": dropped,
                "drop_reason": reason,
            }
            stats[reason] += 1
            out.append(new_row)
    order = {row["id"]: index for index, row in enumerate(rows)}
    out.sort(key=lambda row: order[row["id"]])
    stats["renamed_id"] = _neutralize_ids(out)
    return out, stats


def _neutralize_ids(rows: list[dict]) -> int:
    """Rename ids such as `<episode>_current_state_uncertain_supp_q01` to `<episode>_qNN` so ids carry no label."""
    canonical = re.compile(r"_q(\d+)$")
    next_index: dict[str, int] = defaultdict(int)
    for row in rows:
        suffix = row["id"][len(row["episode_id"]):]
        match = canonical.fullmatch(suffix) if row["id"].startswith(row["episode_id"]) else None
        if match:
            next_index[row["episode_id"]] = max(next_index[row["episode_id"]], int(match.group(1)))
    renamed = 0
    pending = [row for row in rows if not canonical.fullmatch(row["id"][len(row["episode_id"]):])
               and not re.fullmatch(r"_v\d+e\d+_q\d+", row["id"][len(row["episode_id"]):])]
    for row in sorted(pending, key=lambda row: (row["episode_id"], float(row["query_time"]), row["id"])):
        next_index[row["episode_id"]] += 1
        row["release_meta"]["source_id"] = row["id"]
        row["id"] = f"{row['episode_id']}_q{next_index[row['episode_id']]:02d}"
        renamed += 1
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("id neutralization produced duplicates")
    return renamed


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


# ---------------------------------------------------------------- validation and leak audit

def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def _auroc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank = (i + j + 1) / 2
        rank_sum += rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def _nb_status_auroc(rows: list[dict], field) -> float:
    scores, labels = _nb_status_scores(rows, field)
    return _auroc([scores[row["id"]] for row in rows], labels)


def _nb_status_scores(rows: list[dict], field) -> tuple[dict, list[int]]:
    """Episode-grouped 5-fold Bernoulli-style NB scores for Uncertain from text only (out-of-fold)."""
    folds = defaultdict(list)
    for row in rows:
        folds[int.from_bytes(_hash(row["episode_id"])[:4], "big") % 5].append(row)
    scores: dict[str, float] = {}
    for fold in range(5):
        train = [row for f, group in folds.items() if f != fold for row in group]
        test = folds.get(fold, [])
        counts = {0: Counter(), 1: Counter()}
        prior = Counter()
        for row in train:
            label = int(row["diagnostics"]["epistemic_status"] == "uncertain")
            prior[label] += 1
            counts[label].update(set(_tokens(field(row))))
        vocab = set(counts[0]) | set(counts[1])
        totals = {label: sum(counts[label].values()) + len(vocab) for label in (0, 1)}
        for row in test:
            words = set(_tokens(field(row))) & vocab
            score = math.log((prior[1] + 1) / (prior[0] + 1))
            for word in words:
                score += math.log((counts[1][word] + 1) / totals[1]) - math.log((counts[0][word] + 1) / totals[0])
            scores[row["id"]] = score
    labels = [int(row["diagnostics"]["epistemic_status"] == "uncertain") for row in rows]
    return scores, labels


def check_rows(subset: str, rows: list[dict]) -> dict:
    errors = []
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")
    errors.extend(f"{i}: id carries a status word" for i in ids if re.search(r"known|uncertain|supp", i, re.I))
    known = [row for row in rows if row["diagnostics"]["epistemic_status"] == "known"]
    uncertain = [row for row in rows if row["diagnostics"]["epistemic_status"] == "uncertain"]
    if len(known) + len(uncertain) != len(rows):
        errors.append("unknown epistemic status")
    for row in rows:
        options, gold, status = row["options"], row["answer_index"], row["diagnostics"]["epistemic_status"]
        if len(options) != 4 or options[ABSTENTION_INDEX] != ABSTENTION or row.get("abstention_index") != ABSTENTION_INDEX:
            errors.append(f"{row['id']}: abstention option missing or misplaced")
        if len({option.strip().lower() for option in options}) != 4 or any(not option.strip() for option in options):
            errors.append(f"{row['id']}: empty or duplicate options")
        if any(META.search(option) for option in options[:ABSTENTION_INDEX]):
            errors.append(f"{row['id']}: abstention-type wording in substantive options")
        if status == "uncertain" and gold != ABSTENTION_INDEX:
            errors.append(f"{row['id']}: Uncertain gold is not the abstention option")
        if status == "known" and not 0 <= gold < ABSTENTION_INDEX:
            errors.append(f"{row['id']}: Known gold is not substantive")
        sources = row["diagnostics"].get("uncertainty_sources") or []
        if (status == "uncertain") != bool(sources):
            errors.append(f"{row['id']}: uncertainty_sources inconsistent with status")
        if any(span[0] < 0 or span[1] > float(row["query_time"]) + 1e-6 for span in row.get("evidence_spans") or []):
            errors.append(f"{row['id']}: evidence span after query_time")
        if not 1 <= int(row.get("change_intensity", 0)) <= 10:
            errors.append(f"{row['id']}: change_intensity out of range")

    def rate(items, predicate):
        return round(100 * sum(map(predicate, items)) / max(len(items), 1), 2)

    def length_rank(row):
        return _length_rank(row["options"][row["answer_index"]], row["options"][:3])

    def strictly_longest(row):
        lengths = [len(option) for option in row["options"][:3]]
        gold = lengths[row["answer_index"]]
        return all(gold > x for i, x in enumerate(lengths) if i != row["answer_index"])

    unique_max = [r for r in known if sorted(len(o) for o in r["options"][:3])[-1] > sorted(len(o) for o in r["options"][:3])[-2]]

    # Blind "agent" that knows the construction: abstain when the text classifier says Uncertain,
    # otherwise pick the longest substantive option. Threshold is tuned on the test labels (upper bound).
    nb, _ = _nb_status_scores(rows, lambda row: row["question"] + " " + " ".join(row["options"][:3]))
    best = (-1.0, float("inf"))
    for threshold in sorted(set(nb.values())) + [float("inf")]:
        total = 0
        for row in rows:
            if nb[row["id"]] >= threshold:
                pick = ABSTENTION_INDEX
            else:
                pick = max(range(3), key=lambda i: (len(row["options"][i]), row["options"][i]))
            # Effective Reliability (Whitehead et al., 2022), c = 1: abstain 0, correct answer +1, wrong answer -1.
            total += 0 if pick == ABSTENTION_INDEX else (1 if pick == row["answer_index"] else -1)
        best = max(best, (total / len(rows), threshold))
    base_uncertain = round(100 * len(uncertain) / len(rows), 2)
    report = {
        "subset": subset,
        "n": len(rows),
        "known": len(known),
        "uncertain": len(uncertain),
        "errors": errors[:20],
        "error_count": len(errors),
        # Structure cue: with a fixed abstention option on every item, any rule on option structure sees identical layouts.
        "items_with_abstention_option_pct": rate(rows, lambda row: ABSTENTION in row["options"]),
        "known_gold_position_pct": {p: rate(known, lambda row, p=p: row["answer_index"] == p) for p in range(3)},
        "known_gold_longest_pct": rate(known, lambda row: length_rank(row) == 2),
        # Length cue: among Known items with a unique longest option, how often is it the answer (chance 33.3%).
        "known_gold_strict_longest_given_unique_pct": rate(unique_max, strictly_longest),
        "known_gold_shortest_pct": rate(known, lambda row: length_rank(row) == 0),
        "known_gold_middle_pct": rate(known, lambda row: length_rank(row) == 1),
        "uncertain_base_rate_pct": base_uncertain,
        "blind_agent_best_effective_reliability": round(best[0], 4),
        "text_status_auroc_question": round(_nb_status_auroc(rows, lambda row: row["question"]), 3),
        "text_status_auroc_question_options": round(
            _nb_status_auroc(rows, lambda row: row["question"] + " " + " ".join(row["options"][:3])), 3
        ),
    }
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    parser.add_argument("--subsets", default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--out-root", default=None, help="Dry run: write rebuilt manifests here.")
    mode.add_argument("--in-place", action="store_true", help="Back up the manifests, then overwrite them.")
    mode.add_argument("--check", action="store_true", help="Validate current manifests and print the leak audit.")
    args = parser.parse_args(argv)
    root = Path(args.root)
    subsets = list(SUBSET_FILES) if args.subsets == "all" else args.subsets.split(",")
    if args.check:
        reports = [check_rows(subset, read_jsonl(root / d / f)) for subset in subsets for d, f in [SUBSET_FILES[subset]]]
        for report in reports:
            print(json.dumps(report, ensure_ascii=False))
        if any(report["error_count"] for report in reports):
            sys.exit(1)
        return
    if not args.in_place and args.out_root is None:
        parser.error("choose --out-root, --in-place or --check")
    backup = root / f"_backup_pre_unify_{datetime.now():%Y%m%d_%H%M%S}"
    for subset in subsets:
        directory, filename = SUBSET_FILES[subset]
        path = root / directory / filename
        rows = read_jsonl(path)
        if any(row.get("release_meta") for row in rows):
            raise RuntimeError(f"{path} is already rebuilt; refusing to rebuild twice")
        rebuilt, stats = rebuild_rows(subset, rows)
        report = check_rows(subset, rebuilt)
        if report["error_count"]:
            raise RuntimeError(f"{subset}: rebuilt manifest fails validation: {report['errors']}")
        if args.in_place:
            (backup / directory).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup / directory / filename)
            write_jsonl(path, rebuilt)
            target = path
        else:
            target = Path(args.out_root) / directory / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            write_jsonl(target, rebuilt)
        print(json.dumps({"subset": subset, "in": len(rows), "out": len(rebuilt), "target": str(target), **stats}))
    if args.in_place:
        print(f"backup: {backup}")


if __name__ == "__main__":
    main()
