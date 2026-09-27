#!/usr/bin/env python3
"""Probe independent targets concurrently and merge native evidence by room."""
import argparse
import collections
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.room_program import native_probe_ready
from rollout_rooms_native import merge_probe_entry


def pending_targets(rooms, evidence, require_animated=False):
    """Probe untested targets and revisit logical-only results under animation checks."""
    queues = []
    for index, room in rooms:
        previous = {item['id']: item for item in evidence.get(index, {}).get('targets', [])}
        fresh, retry = [], []
        for candidate in room['candidates']:
            old = previous.get(candidate['id'])
            if (require_animated and old is not None and not old.get('animation_attempted')
                    and not any(op.get('accepted') and op.get('animation_verified')
                                for op in old.get('operations', []))):
                fresh.append((index, candidate['id']))
            elif old is None:
                fresh.append((index, candidate['id']))
            elif not any(op.get('accepted') for op in old.get('operations', [])) and old.get('infrastructure_failure'):
                retry.append((index, candidate['id']))
        queues.append(collections.deque(fresh + retry))
    result = []
    while any(queues):
        for queue in queues:
            if queue:
                result.append(queue.popleft())
    return result


def available_gib():
    values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(values['MemAvailable'].split()[0]) / 1024 ** 2


def owned_simulators(folder, executable):
    """Locate Unity and Xorg processes with this task's exact log directory."""
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [item.decode() for item in (entry / 'cmdline').read_bytes().split(b'\0') if item]
            if not args:
                continue
            if args[0] == str(executable):
                log = Path(args[args.index('-logFile') + 1])
            elif Path(args[0]).name == 'Xorg':
                log = Path(args[args.index('-config') + 1])
            else:
                continue
            if folder in log.parents:
                result.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return result


def stop_task(task, executable):
    process = task['process']
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    # Unity starts its own session, so terminating only the Python group is insufficient.
    for pid in owned_simulators(task['folder'], executable):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if owned_simulators(task['folder'], executable):
        time.sleep(1)
        for pid in owned_simulators(task['folder'], executable):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--probes', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--xorg-root', type=Path, required=True)
    parser.add_argument('--room-index', type=int, action='append', required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--gpu-indices', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--unity-workers', type=int, default=12)
    parser.add_argument('--reserve-memory-gib', type=float, default=128)
    parser.add_argument('--task-timeout', type=float, default=600)
    parser.add_argument('--require-animated',action='store_true')
    parser.add_argument('--complete-pool',action='store_true',help='Continue after minimum room readiness')
    parser.add_argument('--target-classes',nargs='+',help='Restrict new probes to these native classes')
    parser.add_argument('--time-scale',type=float,default=8.0)
    args = parser.parse_args()
    if args.workers < 1 or args.reserve_memory_gib < 0 or args.task_timeout <= 0:
        parser.error('Worker count and timeout must be positive; memory reserve cannot be negative')
    if not 1 <= args.unity_workers <= 64 or any(gpu < 0 for gpu in args.gpu_indices):
        parser.error('Unity worker count must be 1..64; GPU indices cannot be negative')
    args.output = args.output.resolve()
    args.executable = args.executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lease = (args.output / 'worker.lock').open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output / 'status.json').exists():
        parser.error('Use a new output directory; completed target reports can be supplied through --probes')
    catalog = json.loads((args.preflight / 'status.json').read_text())['rooms']
    if any(index < 0 or index >= len(catalog) for index in args.room_index):
        parser.error('Room indices must refer to entries in the preflight catalog')
    rooms = [(index, catalog[index]) for index in sorted(set(args.room_index))]
    graphs = {room['scene']: json.loads((args.preflight / f"scene_{room['scene']}.json").read_text()) for _, room in rooms}
    evidence = {index: dict(scene=room['scene'], room_id=room['room_id'], room=room['room'], targets=[]) for index, room in rooms}
    for path in args.probes:
        for incoming in json.loads(path.read_text()).get('rooms', []):
            for index, room in rooms:
                if (room['scene'], room['room_id']) == (incoming['scene'], incoming['room_id']):
                    evidence[index] = merge_probe_entry(evidence[index], incoming)
    def ready(index):
        checked=dict(evidence[index],targets=[dict(target,operations=[operation
            for operation in target['operations']
            if not args.require_animated or operation.get('animation_verified')])
            for target in evidence[index]['targets']])
        return native_probe_ready(graphs[catalog[index]['scene']], checked)
    pending = collections.deque(item for item in pending_targets(rooms, evidence,args.require_animated)
        if (args.complete_pool or not ready(item[0])) and (not args.target_classes or
            next(node['class_name'] for node in graphs[catalog[item[0]]['scene']]['nodes']
                 if node['id']==item[1]) in args.target_classes))
    state = dict(status='running', pid=os.getpid(), started_at=time.time(), worker_limit=args.workers,
                 unity_workers=args.unity_workers, memory_reserve_gib=args.reserve_memory_gib,
                 gpu_indices=args.gpu_indices, tasks=[], rooms=[])
    state['require_animated']=args.require_animated
    active = {}
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save():
        for index, _ in rooms:
            evidence[index]['status'] = 'complete' if ready(index) else ('running' if active or pending else 'blocked')
        state.update(updated_at=time.time(), active=len(active), pending=len(pending),
                     available_memory_gib=available_gib(), rooms=list(evidence.values()))
        temporary = args.output / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(args.output / 'status.json')

    save()
    try:
        while (pending or active) and not stopping:
            for key, task in list(active.items()):
                process = task['process']
                timeout = time.monotonic() - task['started'] >= args.task_timeout
                if process.poll() is None and not timeout:
                    continue
                stop_task(task, args.executable)
                task['log'].close()
                index, target = key
                report_path = task['folder'] / 'status.json'
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                for incoming in report.get('rooms', []):
                    evidence[index] = merge_probe_entry(evidence[index], incoming)
                target_data = next((item for item in evidence[index]['targets'] if item['id'] == target), {})
                accepted = any(op.get('accepted') and (not args.require_animated or op.get('animation_verified'))
                               for op in target_data.get('operations', []))
                task['record'].update(status='accepted' if accepted else 'rejected',
                                      exit_code=process.returncode, timed_out=timeout, ended_at=time.time())
                del active[key]
            while pending and len(active) < args.workers and available_gib() >= args.reserve_memory_gib:
                index, target = pending.popleft()
                if ready(index) and not args.complete_pool:
                    continue
                counts = collections.Counter(task['record']['gpu_index'] for task in active.values())
                gpu = min(args.gpu_indices, key=lambda item: counts[item])
                folder = args.output / f'room_{index:02d}' / f'target_{target}'
                folder.mkdir(parents=True)
                command = [sys.executable, '-u', str(Path(__file__).with_name('probe_room_actions.py')),
                           '--preflight', str(args.preflight), '--output', str(folder),
                           '--executable', str(args.executable), '--xorg-root', str(args.xorg_root),
                           '--gpu-index', str(gpu), '--room-index', str(index), '--target-id', str(target),
                           '--action-timeout', '75', '--time-scale', str(args.time_scale)]
                if args.require_animated:command.append('--require-animated')
                env = os.environ.copy()
                env.update(STORM_UNITY_JOB_WORKERS=str(args.unity_workers), OPENBLAS_NUM_THREADS='1',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                log = (folder / 'worker.log').open('w')
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record = dict(room_index=index, target_id=target, gpu_index=gpu, pid=process.pid,
                              output=str(folder), status='running', started_at=time.time())
                state['tasks'].append(record)
                active[(index, target)] = dict(process=process, log=log, folder=folder,
                                              record=record, started=time.monotonic())
            save()
            time.sleep(2)
    except BaseException as exc:
        stopping = True
        state['failure'] = repr(exc)
        raise
    finally:
        for task in active.values():
            stop_task(task, args.executable)
            task['log'].close()
            task['record'].update(status='interrupted', ended_at=time.time())
        active.clear()
        state['status'] = 'interrupted' if stopping else ('complete' if all(ready(i) for i, _ in rooms) else 'completed_with_errors')
        save()


if __name__ == '__main__':
    main()
