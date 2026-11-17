#!/usr/bin/env python3
"""Revalidate finished captures in new directories while preserving original evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rollout_rooms import snapshot_source
from rollout_rooms_native import capture_accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--defer-qa', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('Use one to sixteen workers')
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    source = args.output / 'source'
    snapshot_source(project, source)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for name in ('scripts', 'src') for p in (source / name).rglob('*.py')}
    revision = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:12]
    (source / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    jobs = []
    seen = set()
    for path in sorted(args.input.glob('*/status.json')):
        for room in json.loads(path.read_text()).get('rooms', []):
            for attempt in room.get('attempts', []):
                capture = Path(attempt['path']) / 'capture'
                if attempt['status'] != 'rejected' or str(capture) in seen:
                    continue
                seen.add(str(capture))
                try:
                    events = json.loads((capture / 'events.json').read_text())
                    if len(events) != 10 or not (capture / 'video.mp4').is_file():
                        continue
                except (OSError, ValueError):
                    continue
                jobs.append((room, capture))
    state = dict(status='running', source_revision=revision, rooms=[])

    def save():
        temporary = args.output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(args.output / 'status.json')

    def run(index, room, original):
        folder = args.output / 'attempts' / f'{index:03d}'
        capture = folder / 'capture'
        capture.mkdir(parents=True)
        for item in original.iterdir():
            destination = capture / item.name
            if item.is_dir():
                destination.symlink_to(item.resolve(), target_is_directory=True)
            elif item.suffix == '.mp4':
                if item.name == 'video.mp4':
                    destination.symlink_to(item.resolve())
            else:
                shutil.copy2(item, destination)
        (capture / 'revalidation_provenance.json').write_text(json.dumps(dict(
            original_capture=str(original), source_revision=revision,
            policy='Rerun full audits; preserve original video and raw evidence.'), indent=2))
        started = time.time()
        command = [sys.executable, '-c',
                   'import sys; from pathlib import Path; '
                   'sys.path.insert(0,sys.argv[1]); from rollout_mixed import finalize; '
                   'finalize(Path(sys.argv[2]),defer_qa=sys.argv[3]=="1")',
                   str(source / 'scripts'), str(capture), '1' if args.defer_qa else '0']
        with (folder / 'render.log').open('w') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        program = json.loads((capture / 'event_program.json').read_text())
        accepted = code == 0 and capture_accepted(capture, program,require_qa=not args.defer_qa)
        record = dict(path=str(folder), source_revision=revision, started_at=started,
                      status='accepted' if accepted else 'rejected', capture_gate=accepted,
                      ended_at=time.time(), exit_code=code,
                      failure='' if accepted else (folder / 'render.log').read_text()[-3500:])
        return dict(scene=room['scene'], room_id=room['room_id'], room=room['room'],
                    status=record['status'], attempts=[record])

    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, i, room, capture) for i, (room, capture) in enumerate(jobs)]
        for future in as_completed(futures):
            state['rooms'].append(future.result())
            save()
    state.update(status='complete', accepted=sum(r['status'] == 'accepted' for r in state['rooms']))
    save()
    print(json.dumps(dict(candidates=len(jobs), accepted=state['accepted']), indent=2))


if __name__ == '__main__':
    main()
