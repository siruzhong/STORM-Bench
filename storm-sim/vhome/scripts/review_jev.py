#!/usr/bin/env python3
"""Run optional Jev reviews over QA records or a recovery status snapshot."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.judgment import digest, failure_request, plan_request, qa_request, review


def load_rows(path, kind):
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return value
    if kind == 'plan':
        return [value]
    return value['questions' if kind == 'qa' else 'latest_rooms']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['qa', 'failure', 'plan'], required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--original', type=Path, help='Original QA file, joined by unique ID')
    parser.add_argument('--model', default='jev-latest')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    if args.original and args.kind != 'qa':
        parser.error('--original only applies to QA')
    if not args.dry_run and not os.environ.get('TYPESAFE_API_KEY'):
        parser.error('Set TYPESAFE_API_KEY for live calls, or use --dry-run')
    rows = load_rows(args.input, args.kind)
    original_rows = load_rows(args.original, 'qa') if args.original else None
    if original_rows is not None:
        originals = {row['id']: row for row in original_rows}
        if len(originals) != len(original_rows) or len({row['id'] for row in rows}) != len(rows):
            parser.error('QA IDs must be unique')
        if set(originals) != {row['id'] for row in rows}:
            parser.error('Original and revised QA IDs differ')
        for row in rows:
            old = originals[row['id']]
            for key in ('options', 'answer_index', 'query_time', 'evidence_spans', 'question_type'):
                if row.get(key) != old.get(key):
                    parser.error(f'Protected QA field changed: {row["id"]}: {key}')
            row['original_question'] = old['question']
    if args.kind == 'failure':
        rows = [row for row in rows if row.get('failure')]
    builder = {'qa': qa_request, 'failure': failure_request, 'plan': plan_request}[args.kind]
    requests = [builder(row, args.model) for row in rows]
    args.output.mkdir(parents=True, exist_ok=False)

    def run(item):
        index, request = item
        result = review(request, dry_run=args.dry_run)
        result['record_index'] = index
        result['source_id'] = rows[index].get('id', rows[index].get('output'))
        (args.output / f'{index:05d}.json').write_text(json.dumps(result, indent=2))
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, enumerate(requests)))
    summary = dict(mode='shadow', automatic_action='none', kind=args.kind,
                   input_path=str(args.input.resolve()), input_sha256=digest(rows),
                   model_requested=args.model, records=len(results),
                   statuses=dict(Counter(row['status'] for row in results)),
                   semantics_compared=sum('preserves_time' in req['questions'] for req in requests),
                   total_request_seconds=sum(row.get('elapsed_seconds', 0) for row in results),
                   scope='Text review only; no video evidence or physics verification')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if any(row['status'] == 'unavailable' for row in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
