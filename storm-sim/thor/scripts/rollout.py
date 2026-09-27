#!/usr/bin/env python3
"""Generate four-room rollouts and export rule-based QA."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New run directory')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/benchgen_qa_4scene_v1.yaml')
    parser.add_argument('--seed', type=int, default=20260820)
    parser.add_argument('--qa-seed', type=int, default=20260825)
    parser.add_argument('--profile', choices=['balanced-v25', 'reference'], default='balanced-v25')
    parser.add_argument('--episodes-per-room', type=int, default=7)
    parser.add_argument('--question-count', type=int, default=300)
    parser.add_argument('--reuse-run', action='append', default=[], metavar='ROOM=PATH')
    args = parser.parse_args(argv)
    if args.profile == 'balanced-v25' and (args.episodes_per_room, args.question_count) != (7, 300):
        parser.error('balanced-v25 preserves the original 28-video / 300-QA contract; use reference for other sizes')
    from tools.storm.benchgen.stream_eqa.four_scene_rollout import _balanced_source_order, ROOM_RANGES
    if not 1 <= args.episodes_per_room <= 30 or args.question_count <= 0:
        parser.error('episodes-per-room must be 1..30; question-count must be positive')
    # Check quota feasibility before launching the simulator.
    _balanced_source_order({r: [Path(f'{r}/{i}') for i in range(args.episodes_per_room)] for r in ROOM_RANGES}, args.question_count)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f'output already exists: {output}; choose a new path')
    command = [sys.executable, '-m', 'tools.storm.benchgen.stream_eqa.four_scene_rollout',
               '--config', str(args.config.expanduser().resolve()), '--work-root', str(output / 'raw'),
               '--output', str(output / 'qa_reference'), '--seed', str(args.seed),
               '--episodes-per-room', str(args.episodes_per_room), '--question-count', str(args.question_count)]
    for reuse in args.reuse_run:
        room, sep, path = reuse.partition('=')
        if not sep or room not in ROOM_RANGES:
            parser.error(f'invalid --reuse-run: {reuse}')
        command += ['--reuse-run', f'{room}={Path(path).expanduser().resolve()}']
    child_env = dict(os.environ)
    child_env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + child_env.get('PYTHONPATH', '')
    subprocess.run(command, cwd=ROOT, env=child_env, check=True)
    if args.profile == 'balanced-v25':
        subprocess.run([sys.executable, str(ROOT / 'scripts/export_qa.py'),
                        '--source-dataset', str(output / 'qa_reference'), '--output', str(output / 'qa_rule'),
                        '--seed', str(args.qa_seed)], cwd=ROOT, env=child_env, check=True)
    print(f'Completed: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
