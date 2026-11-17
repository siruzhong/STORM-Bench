#!/usr/bin/env python
"""Apply reviewed text edits to the RevisitQA v2 manifests (question stems and substantive options only).

Edits live in scripts/data/revisitqa_text_edits/*.json as a list of
  {"id", "field": "question" | "option", "old": str | null, "new": str, "reason": str}
`old` must match exactly (null replaces the whole question). Labels, answers and option order never change.
A deterministic pass also strips intensifiers (definitely, clearly, exact/exactly) from substantive options.

Usage:
  python scripts/data/revisitqa_apply_text_edits.py --out-root /tmp/revisitqa_v3   # dry run + audit
  python scripts/data/revisitqa_apply_text_edits.py --in-place                 # backup + overwrite
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.revisitqa_unify_abstention import ABSTENTION_INDEX, SUBSET_FILES, check_rows, read_jsonl, write_jsonl  # noqa: E402

EDIT_DIR = Path(__file__).resolve().parent / "revisitqa_text_edits"
OPTION_RULES = (
    (re.compile(r"\b(definitely|clearly|certainly) ", re.I), ""),
    (re.compile(r"\bexactly (?=(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d))", re.I), ""),
    (re.compile(r"\bthe exact (?=(same|moment|time))", re.I), "the "),
)


NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
NUMBER_PATTERN = re.compile(r"^(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I)


def _number_to_digit(text: str) -> str | None:
    """Spell-out counts differ in length ("Three" vs "Two"); digits remove that cue when every option is a count."""
    if re.match(r"^\d+\b", text):
        return text
    match = NUMBER_PATTERN.match(text)
    return NUMBER_WORDS[match.group(1).lower()] + text[match.end():] if match else None


def _clean_option(text: str) -> str:
    new = text
    for pattern, repl in OPTION_RULES:
        new = pattern.sub(repl, new)
    new = re.sub(r"\s+", " ", new).strip()
    if new and text[:1].isupper() and new[:1].islower():
        new = new[:1].upper() + new[1:]
    return new


def load_edits(only: str | None = None) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    for path in sorted(EDIT_DIR.glob("*.json")):
        if only and path.name != only:
            continue
        for edit in json.loads(path.read_text(encoding="utf-8")):
            edit["source_file"] = path.name
            by_id.setdefault(edit["id"], []).append(edit)
    return by_id


def apply_row(row: dict, edits: list[dict], stats: Counter) -> dict:
    new = dict(row)
    new["options"] = list(row["options"])
    applied = []
    # Normalize counts first so reviewed edits written against digit options match.
    numbers = [_number_to_digit(new["options"][i]) for i in range(ABSTENTION_INDEX)]
    if all(n is not None for n in numbers) and numbers != new["options"][:ABSTENTION_INDEX]:
        new["options"][:ABSTENTION_INDEX] = numbers
        applied.append({"field": "option", "reason": "number_words_to_digits"})
        stats["number_words_to_digits"] += 1
    for edit in edits:
        if edit["field"] == "question":
            fields = [key for key in ("question", "question_stem") if key in new]
            old = edit["old"]
            hits = 0
            for key in fields:
                if old is None and key == "question" and "question_stem" not in new:
                    new[key] = edit["new"]
                    hits += 1
                elif old is not None and old in new[key]:
                    new[key] = new[key].replace(old, edit["new"])
                    hits += 1
            if not hits:
                raise ValueError(f"{row['id']}: question edit does not match: {edit['old']!r}")
        elif edit["field"] == "option":
            index = new["options"].index(edit["old"]) if edit["old"] in new["options"] else -1
            if index < 0 or index == ABSTENTION_INDEX:
                raise ValueError(f"{row['id']}: option edit does not match a substantive option: {edit['old']!r}")
            new["options"][index] = edit["new"]
        else:
            raise ValueError(f"{row['id']}: unknown field {edit['field']}")
        applied.append({"field": edit["field"], "reason": edit["reason"], "source_file": edit["source_file"]})
        stats[edit["reason"]] += 1
    for index in range(ABSTENTION_INDEX):
        cleaned = _clean_option(new["options"][index])
        if cleaned != new["options"][index]:
            new["options"][index] = cleaned
            applied.append({"field": "option", "reason": "strip_option_intensifier"})
            stats["strip_option_intensifier"] += 1
    if applied:
        previous = row.get("release_meta") or {}
        new["release_meta"] = {
            **previous,
            "version": previous.get("version", "text_edits"),
            "source_question": previous.get("source_question", row["question"]),
            "source_question_stem": previous.get("source_question_stem", row.get("question_stem")),
            "source_options": previous.get("source_options", row["options"]),
            "edits": previous.get("edits", []) + applied,
        }
    return new


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/eval_video/RevisitQA")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out-root")
    mode.add_argument("--in-place", action="store_true")
    parser.add_argument("--only", default=None, help="Apply a single edit file (incremental batch).")
    args = parser.parse_args(argv)
    root = Path(args.root)
    edits = load_edits(args.only)
    seen = set()
    backup = root / f"_backup_pre_{(args.only or 'text_edits').replace('.json', '')}_{datetime.now():%Y%m%d_%H%M%S}"
    for subset, (directory, filename) in SUBSET_FILES.items():
        path = root / directory / filename
        rows = read_jsonl(path)
        stats = Counter()
        out = []
        for row in rows:
            out.append(apply_row(row, edits.get(row["id"], []), stats))
            seen.add(row["id"])
        report = check_rows(subset, out)
        if report["error_count"]:
            raise RuntimeError(f"{subset}: {report['errors']}")
        if args.in_place:
            (backup / directory).mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup / directory / filename)
            target = path
        else:
            target = Path(args.out_root) / directory / filename
            target.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(target, out)
        print(json.dumps({"subset": subset, "edited_rows": sum("release_meta" in r for r in out), **stats,
                          "auroc_q": report["text_status_auroc_question"],
                          "auroc_qo": report["text_status_auroc_question_options"],
                          "gold_longest_given_unique": report["known_gold_strict_longest_given_unique_pct"],
                          "blind_er": report["blind_agent_best_effective_reliability"]}))
    missing = sorted(set(edits) - seen)
    if missing:
        raise RuntimeError(f"edits reference unknown ids: {missing[:10]}")
    if args.in_place:
        print(f"backup: {backup}")


if __name__ == "__main__":
    main()
