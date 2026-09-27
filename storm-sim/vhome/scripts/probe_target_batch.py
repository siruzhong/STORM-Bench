#!/usr/bin/env python3
"""Probe explicit room targets in isolated sessions and publish one merged status."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def parse_specs(values):
    specs=[]
    for value in values:
        room_text,target_text=value.split(':',1)
        targets=tuple(int(item) for item in target_text.split(',') if item)
        if not targets:raise ValueError(f'No target IDs in spec: {value}')
        specs.append((int(room_text),targets))
    return specs


def accepted_targets(report):
    return {target['id'] for room in report.get('rooms',[]) for target in room.get('targets',[])
            if any(operation.get('accepted') for operation in target.get('operations',[]))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--executable',required=True)
    parser.add_argument('--xorg-root',required=True)
    parser.add_argument('--gpu-index',type=int,required=True)
    parser.add_argument('--spec',action='append',required=True,metavar='ROOM_INDEX:TARGET_ID,...')
    parser.add_argument('--attempts',type=int,default=3)
    parser.add_argument('--action-timeout',type=int,default=75)
    parser.add_argument('--time-scale',type=float,default=8.0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    specs=parse_specs(args.spec)
    state=dict(status='running',started_at=time.time(),gpu_index=args.gpu_index,rooms=[])
    path=args.output/'status.json'
    def save():
        state['updated_at']=time.time();temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2));temporary.replace(path)
    save()
    worker=Path(__file__).with_name('probe_room_actions.py')
    for room_index,targets in specs:
        entry=dict(room_index=room_index,target_ids=list(targets),status='running',attempts=[])
        state['rooms'].append(entry);folder=args.output/f'room_{room_index:02d}';folder.mkdir(exist_ok=True);save()
        for attempt in range(args.attempts):
            command=[sys.executable,str(worker),'--preflight',str(args.preflight),'--output',str(folder),
                     '--executable',args.executable,'--xorg-root',args.xorg_root,
                     '--gpu-index',str(args.gpu_index),'--room-index',str(room_index),
                     '--action-timeout',str(args.action_timeout),'--time-scale',str(args.time_scale)]
            for target in targets:command.extend(['--target-id',str(target)])
            started=time.time()
            with (folder/f'attempt_{attempt}.log').open('w') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            report=json.loads((folder/'status.json').read_text())
            accepted=accepted_targets(report)
            entry['attempts'].append(dict(started_at=started,ended_at=time.time(),exit_code=result.returncode,
                                          accepted_target_ids=sorted(accepted & set(targets))))
            if set(targets)<=accepted:
                entry['status']='complete';break
            save()
        if entry['status']!='complete':entry['status']='blocked'
        save()
    state['status']='complete' if all(room['status']=='complete' for room in state['rooms']) else 'completed_with_errors'
    save()
    if state['status']!='complete':raise SystemExit(2)


if __name__=='__main__':main()
