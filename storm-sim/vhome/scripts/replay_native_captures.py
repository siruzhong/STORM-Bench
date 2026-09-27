#!/usr/bin/env python3
"""Replay selected frozen captures with a new implementation in isolated sessions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import yaml

from rollout_rooms import snapshot_source
from rollout_rooms_native import capture_accepted


def frozen_source(original):
    """Locate and verify the exact implementation recorded for an earlier attempt."""
    original=Path(original).resolve()
    for parent in original.parents:
        status_path=parent/'status.json'
        if not status_path.exists():continue
        status=json.loads(status_path.read_text())
        for room in status.get('rooms',[]):
            for attempt in room.get('attempts',[]):
                if Path(attempt.get('path','')).resolve()!=original.parent:continue
                revision=attempt['source_revision']
                candidates=[parent/'sources'/revision,original.parent/'source',parent/'source']
                for source in candidates:
                    manifest=source/'source_hashes.json'
                    if not manifest.exists():continue
                    expected=json.loads(manifest.read_text())
                    actual={str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest()
                            for name in ('src','scripts') for path in (source/name).rglob('*.py')}
                    digest=hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest()[:12]
                    if actual==expected and digest==revision:return source,revision
                raise ValueError(f'Frozen source is missing or changed for {original}')
    raise ValueError(f'No attempt provenance was found for {original}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True, help='JSON array of original capture paths')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--xorg-root', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--viewpoint', choices=['first_person'],
                        help='Derive a camera-only plan while preserving the source event program')
    parser.add_argument('--preserve-source',action='store_true',
                        help='Replay each capture with its verified original implementation')
    args = parser.parse_args()
    if not 1 <= args.workers <= 29:
        parser.error('Use one to twenty-nine workers')
    if args.preserve_source and args.viewpoint:
        parser.error('Source preservation and viewpoint conversion are separate replay modes')
    inputs = [Path(value) for value in json.loads(args.inputs.read_text())]
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / 'source'
    snapshot_source(Path(__file__).resolve().parents[1], source)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for name in ('scripts', 'src') for p in (source / name).rglob('*.py')}
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    (source / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    state = dict(status='running', source_revision=revision, rooms=[])
    lock = threading.Lock()

    def save():
        temporary = args.output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(args.output / 'status.json')

    def run(item):
        index, original = item
        config_path = original.parent / 'config.yaml'
        config = (yaml.safe_load(config_path.read_text()) if config_path.exists()
                  else json.loads((original / 'config.json').read_text()))
        config.get('camera', {}).pop('recording_camera', None)
        folder = args.output / 'attempts' / f'{index:03d}'
        folder.mkdir(parents=True)
        implementation,implementation_revision=source,revision
        if args.preserve_source:
            original_source,implementation_revision=frozen_source(original)
            implementation=folder/'source'
            shutil.copytree(original_source,implementation,
                            ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copy2(original / 'event_program.json', folder / 'event_program.json')
        program = json.loads((folder / 'event_program.json').read_text())
        if args.viewpoint:
            from storm_virtualhome.viewpoint import first_person_plan
            from storm_virtualhome.planning import validate_program
            config, program = first_person_plan(config, program)
            graph = json.loads((original / 'planning_graph.json').read_text())
            validate_program(program, graph, config)
            (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
            (folder / 'event_program.json').write_text(json.dumps(program, indent=2))
        attempt = dict(path=str(folder), original_capture=str(original), source_revision=implementation_revision,
                       implementation_source=str(implementation),preserved_source=args.preserve_source,
                       status='running', started_at=time.time())
        room = dict(scene=config['scene'], room_id=config['room_id'], room=config['room_name'],
                    status='running', attempts=[attempt])
        command = [sys.executable, '-u', str(implementation / 'scripts/rollout_native_room.py'),
                   '--config', str(folder / 'config.yaml'), '--plan', str(folder / 'event_program.json'),
                   '--output', str(folder / 'capture'), '--executable', args.executable,
                   '--xorg-root', args.xorg_root, '--gpu-index', str(index % 4), '--defer-qa']
        with (folder / 'render.log').open('w') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            attempt['child_pid'] = process.pid
            with lock:
                state['rooms'].append(room)
                save()
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
        if code==0:
            capture=folder/'capture'
            runtime=json.loads((capture/'runtime_manifest.json').read_text())
            if runtime.get('per_frame_actor_positions') and not (capture/'actor_separation.json').exists():
                from storm_virtualhome.separation import audit_separation
                report=audit_separation(capture)
                (capture/'actor_separation.json').write_text(json.dumps(report,indent=2))
                if report['status']!='separation_verified' or not report['exact_root_evidence']:
                    code=1
                    with (folder/'render.log').open('a') as log:
                        log.write('\nExternal actor-separation audit failed: '+report['status']+'\n')
        accepted = code == 0 and capture_accepted(folder / 'capture', program,require_qa=False)
        with lock:
            attempt.update(status='accepted' if accepted else 'rejected', capture_gate=accepted,
                           exit_code=code, ended_at=time.time(),
                           failure='' if accepted else (folder / 'render.log').read_text()[-3500:])
            room['status'] = attempt['status']
            save()

    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, enumerate(inputs)))
    state.update(status='complete', accepted=sum(room['status'] == 'accepted' for room in state['rooms']))
    save()
    print(json.dumps(dict(captures=len(inputs), accepted=state['accepted'])))


if __name__ == '__main__':
    main()
