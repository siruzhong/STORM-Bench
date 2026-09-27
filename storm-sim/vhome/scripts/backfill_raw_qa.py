#!/usr/bin/env python3
"""Attach draft annotations to accepted captures, including older live workers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.raw_qa import write_raw_questions
from recovery_status import summarize


def run_once(ledger):
    recovery = summarize(ledger)
    rows = []
    for path in recovery['accepted_episodes']:
        root = Path(path)
        try:
            status_path = root / 'qa_status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            digest = hashlib.sha256((root / 'events.json').read_bytes()).hexdigest()
            if status.get('status') != 'raw_ready' or status.get('events_sha256') != digest:
                if (root / 'qa.json').exists():
                    raise ValueError('Existing annotations require explicit migration; refusing to overwrite')
                status = write_raw_questions(root)
            qa = json.loads((root / 'qa.json').read_text())
            if len(qa['questions']) != 10:
                raise ValueError('Draft must contain ten questions')
            rows.append(dict(capture=path, status='raw_ready', questions=10))
        except (OSError, ValueError, KeyError) as error:
            rows.append(dict(capture=path, status='failed', error=str(error)))
    report = dict(checked_at=time.time(), accepted_videos=recovery['accepted_videos'],
                  raw_qa_videos=sum(row['status'] == 'raw_ready' for row in rows),
                  raw_questions=sum(row.get('questions', 0) for row in rows), captures=rows)
    output = ledger.parent / 'raw_qa_status.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', required=True, type=Path)
    parser.add_argument('--watch-seconds', type=int, default=0)
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()
    if args.watch_seconds < 0 or args.interval < 1:
        parser.error('Watch duration must be nonnegative and interval must be positive')
    deadline = time.monotonic() + args.watch_seconds
    previous = None
    while True:
        report = run_once(args.ledger)
        signature = json.dumps(report['captures'], sort_keys=True)
        if signature != previous:
            print(json.dumps(report), flush=True)
            previous = signature
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))


if __name__ == '__main__':
    main()
