#!/usr/bin/env python3
"""Probe rooms in isolated Unity sessions with bounded recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storm_virtualhome.room_preflight import probe_result_ready
from storm_virtualhome.room_program import native_probe_ready


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--executable',required=True)
    parser.add_argument('--xorg-root',required=True)
    parser.add_argument('--gpu-index',type=int,default=3)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--room-timeout',type=int,default=900)
    parser.add_argument('--attempts',type=int,default=4)
    parser.add_argument('--limit',type=int,default=14)
    parser.add_argument('--action-timeout',type=int,default=75)
    parser.add_argument('--time-scale',type=float,default=8.0)
    parser.add_argument('--room-index',type=int,action='append')
    parser.add_argument('--target-id',type=int,action='append')
    args=parser.parse_args()
    if args.attempts<1:parser.error('--attempts must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lease=(args.output/'worker.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    catalog=json.loads((args.preflight/'status.json').read_text())
    path=args.output/'status.json'
    state=json.loads(path.read_text()) if path.exists() else dict(rooms=[],started_at=time.time())
    state.update(status='running',pid=__import__('os').getpid())
    def save():
        state['updated_at']=time.time()
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
    save()
    for index,room in enumerate(catalog['rooms']):
        if index%args.shards!=args.shard:continue
        if args.room_index is not None and index not in args.room_index:continue
        entry=next((r for r in state['rooms'] if (r['scene'],r['room_id'])==(room['scene'],room['room_id'])),None)
        if entry and entry.get('status')=='complete':
            graph=json.loads((args.preflight/f"scene_{room['scene']}.json").read_text())
            if probe_result_ready(entry.get('targets',[])) and native_probe_ready(graph,entry):
                continue
        if entry and entry.get('status')=='complete':
            entry['status']='pending_describable_target'
            save()
        if entry is None:
            entry={key:room[key] for key in ('scene','room_id','room')}
            entry.update(status='pending',targets=[],attempts=[]);state['rooms'].append(entry)
        folder=args.output/f'room_{index:02d}';folder.mkdir(exist_ok=True)
        for attempt in range(len(entry['attempts']),args.attempts):
            command=[sys.executable,str(Path(__file__).with_name('probe_room_actions.py')),
                     '--preflight',str(args.preflight),'--output',str(folder),
                     '--executable',args.executable,'--xorg-root',args.xorg_root,
                     '--gpu-index',str(args.gpu_index),'--room-index',str(index),'--limit',str(args.limit)]
            command.extend(['--action-timeout',str(args.action_timeout),'--time-scale',str(args.time_scale)])
            for target_id in args.target_id or ():
                command.extend(['--target-id',str(target_id)])
            if attempt:command.extend(['--skip-infrastructure-targets','--skip-failed-targets'])
            record=dict(started_at=time.time(),status='running');entry['attempts'].append(record)
            entry['status']='running';save()
            with (folder/f'attempt_{attempt}.log').open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                record['pid']=process.pid;save()
                try:code=process.wait(timeout=args.room_timeout)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    code=124
            child=folder/'status.json'
            if child.exists():
                data=json.loads(child.read_text())
                if data.get('rooms'):entry['targets']=data['rooms'][0]['targets']
            record.update(ended_at=time.time(),exit_code=code,status='complete' if code==0 else 'interrupted')
            entry['status']='complete' if code==0 else 'interrupted';save()
            if code==0:break
        if entry['status']!='complete':entry['status']='blocked';save()
    state['status']='complete' if all(r['status']=='complete' for r in state['rooms']) else 'completed_with_errors'
    save()


if __name__=='__main__':main()
