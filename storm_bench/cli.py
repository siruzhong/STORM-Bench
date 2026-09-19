from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import load_questions, read_jsonl
from .metrics import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(prog="storm-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("questions")
    score = sub.add_parser("evaluate")
    score.add_argument("questions")
    score.add_argument("predictions")
    score.add_argument("--output", default="results.json")
    args = parser.parse_args()
    if args.command == "validate":
        rows = load_questions(args.questions)
        print(f"valid: {len(rows)} questions")
    else:
        result = evaluate(load_questions(args.questions), read_jsonl(args.predictions))
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
