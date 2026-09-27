#!/usr/bin/env python3
"""Create a seeded event program from a saved scene graph without rendering."""
import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.planning import create_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config['seed'] = args.seed
    program = create_program(json.loads(args.graph.read_text()), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(program, stream, indent=2)
    print(json.dumps({'plan_id': program['plan_id'], 'events': len(program['events']), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
