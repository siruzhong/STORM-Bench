#!/usr/bin/env python3
"""Summarize batch progress, failure causes, and measured attempt durations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('batch', type=Path)
a = p.parse_args()
s = json.loads((a.batch / 'status.json').read_text())
attempts = [item for c in s['candidates'] for item in c['attempts']]
reasons = Counter()
for item in attempts:
    if item['status'] == 'failed':
        path = a.batch / 'logs' / (item['name'] + '.log')
        lines = path.read_text(errors='replace').splitlines() if path.exists() else []
        reasons[item.get('reason', lines[-1] if lines else 'Missing log')] += 1
elapsed = time.time() - s['created_at']
passed = [item for item in attempts if item['status'] in ('captured', 'publishing', 'accepted')]
finished = [item for item in attempts if item['status'] in ('failed', 'captured', 'publishing', 'accepted')]
report = {'status': s['status'], 'published_videos': len(s['accepted']), 'capture_passed': len(passed),
          'selected_qa': s['qa_count'], 'current_attempt': s.get('current_attempt'),
          'elapsed_minutes': elapsed / 60, 'attempt_states': dict(Counter(x['status'] for x in attempts)),
          'observed_attempt_pass_rate': len(passed) / len(finished) if finished else None,
          'failure_reasons': dict(reasons), 'source_revision': s.get('source_dir', 'source')}
for name, group in [('failed', [x for x in attempts if x['status'] == 'failed']), ('passed', passed)]:
    durations = [x['ended_at'] - x['started_at'] for x in group if 'ended_at' in x]
    report[f'{name}_attempt_mean_seconds'] = statistics.mean(durations) if durations else None
if s['accepted']:
    report['observed_wall_minutes_per_published_video'] = elapsed / 60 / len(s['accepted'])
    report['eta_note'] = 'Too few diverse completed seeds for a reliable completion estimate.' if len(s['accepted']) < 5 else 'Observed rate includes retries and pauses; it is not a guaranteed future rate.'
print(json.dumps(report, indent=2))
