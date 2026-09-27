#!/usr/bin/env python3
"""Write a separate frame-level separation report for an existing capture."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.separation import audit_separation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit_separation(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('missing_samples', 'violations')}))
