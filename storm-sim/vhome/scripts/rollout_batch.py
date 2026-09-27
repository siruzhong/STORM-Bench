#!/usr/bin/env python3
"""Freeze candidate programs, resume bounded capture attempts, and publish accepted episodes."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.planning import create_program
from storm_virtualhome.qa import TYPES


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_questions(pool, slot):
    """Cover each event once, balance types across three episodes, and retain uncertainty."""
    lookup = {(q['event_index'], q['question_type']): q for q in pool}
    events = sorted({q['event_index'] for q in pool})
    if events != list(range(10)) or len(lookup) != 60:
        raise ValueError('Expected ten events with all six QA types')
    kinds = [TYPES[(index + slot * 4) % 6] for index in events]
    uncertain = [i for i in events if lookup[i, 'current_state']['diagnostics']['epistemic_status'] == 'uncertain']
    if not uncertain:
        raise ValueError('No observer-uncertainty question is available')
    if not any(kinds[i] == 'current_state' for i in uncertain):
        source = kinds.index('current_state')
        target = uncertain[slot % len(uncertain)]
        kinds[source], kinds[target] = kinds[target], kinds[source]
    return [lookup[i, kind] for i, kind in enumerate(kinds)]


def accepted(episode, program):
    try:
        read = lambda name: json.loads((episode / name).read_text())
        observer, debug = read('observer_validation.json'), read('debug_validation.json')
        qa = read('qa.json')
        return (read('event_program.json') == program and read('plan_execution.json')['accepted']
                and read('camera_continuity.json')['accepted']
                and observer['event_count'] == 10 and 5 <= observer['offscreen_event_count'] <= 6
                and 60 <= observer['duration_sec'] <= 70 and observer['camera_overlap_samples'] == 0
                and debug['frames'] == observer['frames'] and len(qa['questions']) == 60
                and all((episode / name).stat().st_size > 0 for name in ('video.mp4', 'debug.mp4')))
    except (OSError, ValueError, KeyError):
        return False


def publish(root, episode, slot, program, source_dir="source"):
    started = time.monotonic()
    destination = root / 'dataset' / f'episode_{slot + 1:04d}'
    if destination.exists():
        record = json.loads((destination / 'record.json').read_text())
        if record['plan_id'] != program['plan_id']:
            raise ValueError('Published slot belongs to a different plan')
        return record
    stage = root / 'dataset' / f'.episode_{slot + 1:04d}.staging'
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in ('video.mp4', 'debug.mp4', 'event_program.json', 'events.json', 'observer_validation.json',
                 'camera_continuity.json', 'motion_validation.json', 'debug_validation.json', 'plan_execution.json'):
        shutil.copy2(episode / name, stage / name)
    qa = json.loads((episode / 'qa.json').read_text())
    save(stage / 'qa_pool_private.json', qa)
    qa['questions'] = select_questions(qa['questions'], slot)
    save(stage / 'qa.json', qa)
    subprocess.run([sys.executable, str(root / source_dir / 'scripts/export_model_inputs.py'), str(stage),
                    '--output', str(stage / 'evaluation')], check=True)
    record = {'slot': slot, 'seed': program['seed'], 'plan_id': program['plan_id'],
              'source_episode': str(episode.relative_to(root)), 'path': str(destination.relative_to(root)),
              'duration_sec': qa['duration_sec'], 'questions': 10,
              'publish_wall_seconds': time.monotonic() - started, 'source_revision': source_dir,
              'qa_types': dict(Counter(q['question_type'] for q in qa['questions'])),
              'sha256': {name: sha(stage / name) for name in ('video.mp4', 'debug.mp4', 'qa.json')}}
    save(stage / 'record.json', record)
    stage.rename(destination)
    return record


def aggregate(root, state):
    questions, inputs, labels = [], [], []
    for item in state['accepted']:
        packet = root / item['path']
        questions.extend(json.loads((packet / 'qa.json').read_text())['questions'])
        for name, output in [('model_inputs.jsonl', inputs), ('evaluation_labels.jsonl', labels)]:
            rows = [json.loads(line) for line in (packet / 'evaluation' / name).read_text().splitlines()]
            if name == 'model_inputs.jsonl':
                for row in rows:
                    row['video_path'] = str(Path(item['path']) / 'evaluation' / row['video_path'])
            output.extend(rows)
    for name, rows in [('qa_private.jsonl', questions), ('model_inputs.jsonl', inputs), ('evaluation_labels.jsonl', labels)]:
        path = root / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        temporary.replace(path)
    counts = Counter(q['question_type'] for q in questions)
    state['qa_count'] = len(questions)
    state['qa_types'] = dict(counts)
    attempts = [a for c in state.get('candidates', []) for a in c['attempts']]
    state['capture_passed_count'] = sum(a['status'] in ('captured', 'publishing', 'accepted') for a in attempts)
    state['publishing_count'] = sum(a['status'] == 'publishing' for a in attempts)
    state['updated_at'] = time.time()
    if len(state['accepted']) == state['contract']['videos']:
        expected = state['contract']['videos'] * 10
        if len(questions) != expected or len({q['id'] for q in questions}) != expected:
            raise ValueError('Duplicate or missing questions in the final dataset')
        if len(set(counts.values())) != 1 or set(counts) != set(TYPES):
            raise ValueError('Final QA type counts are not balanced')
        from rebuild_batch_qa import rebuild, materialize_prefixes
        qa_report = rebuild(root, root / 'structured_qa')
        materialize_prefixes(root, root / 'structured_qa')
        state['recommended_qa'] = 'structured_qa/model_inputs.jsonl'
        state['qa_uniform_chance'] = qa_report['uniform_chance']
        state['qa_release_status'] = qa_report['status']
        state['status'] = 'complete'
    save(root / 'status.json', state)


def initialize(args):
    root = args.output.resolve()
    if (root / 'status.json').exists():
        state = json.loads((root / 'status.json').read_text())
        for name, expected in state['frozen_files'].items():
            if sha(root / name) != expected:
                raise ValueError(f'Frozen input or source was modified: {name}')
        return root, state
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1]
    shutil.copytree(source, root / 'source', ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.egg-info', '.runtime', '.venv', 'outputs', 'plans'))
    graph = json.loads(args.graph.read_text())
    config = yaml.safe_load(args.config.read_text())
    (root / 'inputs').mkdir()
    save(root / 'inputs/graph.json', graph)
    shutil.copy2(args.config, root / 'inputs/base_config.yaml')
    candidates = []
    for seed in range(args.start_seed, args.start_seed + args.candidates):
        options = dict(config, seed=seed)
        program = create_program(graph, options)
        folder = root / 'inputs' / f'seed_{seed:06d}'
        folder.mkdir()
        (folder / 'config.yaml').write_text(yaml.safe_dump(options, sort_keys=False))
        save(folder / 'program.json', program)
        candidates.append({'seed': seed, 'input': str(folder.relative_to(root)), 'attempts': [], 'accepted': False})
    frozen = {str(p.relative_to(root)): sha(p) for folder in ('source', 'inputs') for p in sorted((root / folder).rglob('*')) if p.is_file()}
    state = {'status': 'prepared', 'created_at': time.time(), 'contract': {
        'videos': args.videos, 'qa_per_video': 10, 'max_attempts': args.max_attempts,
        'attempt_timeout_seconds': args.attempt_timeout, 'min_free_gb': args.min_free_gb,
        'executable': str(args.executable.resolve()), 'gpu_index': args.gpu_index,
        'xorg_root': str(args.xorg_root.resolve()),
        'runtime_guard_sha256': sha(args.executable.resolve().parent / 'camera_guard.json')}, 'frozen_files': frozen,
        'candidates': candidates, 'accepted': [], 'qa_count': 0}
    save(root / 'status.json', state)
    return root, state


def run(root, state):
    with ThreadPoolExecutor(max_workers=1) as publisher:
        run_pipeline(root, state, publisher)


def run_pipeline(root, state, publisher):
    contract = state['contract']
    source_dir = state.get('source_dir', 'source')
    pending = None

    def harvest():
        nonlocal pending
        if pending is None:
            return
        future, candidate, record = pending
        item = future.result()
        state['accepted'].append(item)
        candidate['accepted'] = True
        record.update(status='accepted', published_at=time.time())
        pending = None
        aggregate(root, state)
        print(f'Published {record["name"]}; total {len(state["accepted"])}/{contract["videos"]}', flush=True)
    for candidate in state['candidates']:
        if pending is not None and (pending[0].done() or len(state['accepted']) + 1 >= contract['videos']):
            harvest()
        if len(state['accepted']) >= contract['videos']:
            break
        if candidate['accepted']:
            continue
        inputs = root / candidate['input']
        program = json.loads((inputs / 'program.json').read_text())
        for index in range(contract['max_attempts']):
            name = f'seed_{candidate["seed"]:06d}_attempt_{index + 1:02d}'
            episode = root / 'attempts' / name
            record = next((item for item in candidate['attempts'] if item['name'] == name), None)
            if record and record['status'] in ('failed', 'interrupted'):
                continue
            if record and record['status'] in ('running', 'publishing'):
                record['status'] = 'captured' if accepted(episode, program) else 'interrupted'
                if record['status'] == 'interrupted':
                    save(root / 'status.json', state)
                    continue
            if not record:
                guard = Path(contract['executable']).parent / 'camera_guard.json'
                if sha(guard) != contract['runtime_guard_sha256']:
                    raise ValueError('The camera runtime manifest changed during the batch')
                if shutil.disk_usage(root).free < contract['min_free_gb'] * 1024 ** 3:
                    harvest()
                    state['status'] = 'paused_low_disk'
                    save(root / 'status.json', state)
                    return
                episode.parent.mkdir(exist_ok=True)
                (root / 'logs').mkdir(exist_ok=True)
                record = {'name': name, 'status': 'running', 'started_at': time.time(), 'source_revision': source_dir}
                candidate['attempts'].append(record)
                state['status'] = 'running'
                state['current_attempt'] = name
                save(root / 'status.json', state)
                command = [sys.executable, str(root / source_dir / 'scripts/rollout.py'), '--executable', contract['executable'],
                           '--config', str(inputs / 'config.yaml'), '--plan', str(inputs / 'program.json'),
                           '--output', str(episode), '--gpu-index', str(contract['gpu_index']), '--xorg-root', contract['xorg_root']]
                print(f'Capture {name}; accepted {len(state["accepted"])}/{contract["videos"]}', flush=True)
                with (root / 'logs' / f'{name}.log').open('w') as stream:
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        deadline = time.monotonic() + contract['attempt_timeout_seconds']
                        while True:
                            try:
                                code = child.wait(timeout=min(5, max(0.01, deadline - time.monotonic())))
                                break
                            except subprocess.TimeoutExpired:
                                if pending is not None and pending[0].done():
                                    harvest()
                                if time.monotonic() >= deadline:
                                    raise
                    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                        os.killpg(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        record.update(status='interrupted', ended_at=time.time(), reason=type(error).__name__)
                        save(root / 'status.json', state)
                        if isinstance(error, KeyboardInterrupt):
                            raise
                        continue
                record.update(exit_code=code, ended_at=time.time(), status='captured' if code == 0 and accepted(episode, program) else 'failed')
                if code != 0:
                    lines = (root / 'logs' / f'{name}.log').read_text(errors='replace').splitlines()
                    record['reason'] = lines[-1] if lines else 'No log output'
                save(root / 'status.json', state)
            if record['status'] == 'captured':
                harvest()
                record.update(status='publishing', publishing_started_at=time.time())
                pending = (publisher.submit(publish, root, episode, len(state['accepted']), program, source_dir), candidate, record)
                aggregate(root, state)
                break
    harvest()
    if len(state['accepted']) < contract['videos']:
        state['status'] = 'exhausted'
    state.pop('current_attempt', None)
    aggregate(root, state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revision-source', type=Path)
    p.add_argument('--revision-note')
    p.add_argument('--graph', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--executable', type=Path)
    p.add_argument('--xorg-root', type=Path)
    p.add_argument('--videos', type=int, default=30)
    p.add_argument('--start-seed', type=int, default=42)
    p.add_argument('--candidates', type=int, default=150)
    p.add_argument('--max-attempts', type=int, default=3)
    p.add_argument('--attempt-timeout', type=int, default=1800)
    p.add_argument('--min-free-gb', type=float, default=100)
    p.add_argument('--gpu-index', type=int, default=3)
    args = p.parse_args()
    if not (args.output / 'status.json').exists():
        if any(getattr(args, key) is None for key in ('graph', 'config', 'executable', 'xorg_root')):
            p.error('A new batch requires graph, config, executable, and xorg-root')
        if args.videos < 3 or args.videos % 3 or args.candidates < args.videos or args.max_attempts < 1:
            p.error('Videos must be a positive multiple of three; candidates must cover videos; attempts must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        root, state = initialize(args)
        if args.revision_source:
            if not args.revision_note:
                p.error('A source revision requires an explanation')
            name = f'revisions/revision_{len(state.get("revisions", [])) + 1:03d}'
            shutil.copytree(args.revision_source.resolve(), root / name,
                            ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache', '*.egg-info', '.runtime', '.venv', 'outputs', 'plans'))
            state['frozen_files'].update({str(path.relative_to(root)): sha(path) for path in (root / name).rglob('*') if path.is_file()})
            state.setdefault('revisions', []).append({'source_dir': name, 'created_at': time.time(), 'note': args.revision_note})
            state['source_dir'] = name
            save(root / 'status.json', state)
        run(root, state)
        if state['status'] != 'complete':
            raise SystemExit(2)


if __name__ == '__main__':
    main()
