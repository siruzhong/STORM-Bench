#!/usr/bin/env python3
"""Summarize owned recovery jobs without counting partial captures as deliveries."""
import argparse
import collections
import json
from pathlib import Path
import time
from rollout_rooms_native import capture_accepted


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return {}


def process_matches(job):
    try:
        command=(Path('/proc')/str(job['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        return job['owned_command_fragment'] in command
    except (OSError,KeyError):return False


def summarize(ledger_path):
    ledger=read_json(ledger_path);latest={};accepted={};live=[];probe_progress=[]
    seen=set()
    for job in reversed(ledger.get('jobs',[])):
        output=Path(job['output'])
        if str(output) in seen:continue
        seen.add(str(output))
        data=read_json(output/'status.json')
        if process_matches(job):
            live.append(dict(kind=job['kind'],pid=job['pid'],output=str(output),status=data.get('status')))
        if 'tasks' in data:
            counts=collections.Counter(task['status'] for task in data['tasks'])
            if data.get('active') or data.get('pending'):
                probe_progress.append(dict(output=str(output),active=data.get('active'),
                    pending=data.get('pending'),counts=dict(counts)))
        for room in data.get('rooms',[]):
            key=(room['scene'],room['room_id'])
            attempts=room.get('attempts',[])
            for attempt in attempts:
                if attempt.get('started_at',0)>=latest.get(key,{}).get('started_at',0):
                    path=Path(attempt.get('path',''))/'capture'
                    events=read_json(path/'events.json')
                    latest[key]=dict(scene=key[0],room_id=key[1],room=room['room'],
                        started_at=attempt.get('started_at',0),status=attempt['status'],
                        completed_events=len(events) if isinstance(events,list) else 0,
                        failure=(attempt.get('failure','').splitlines() or [''])[-1],
                        output=str(path),source_revision=attempt.get('source_revision'))
                if attempt.get('capture_gate') and attempt.get('status')=='accepted':
                    path=Path(attempt['path'])/'capture'
                    program=read_json(path/'event_program.json')
                    if capture_accepted(path,program,require_qa=False):accepted.setdefault(key,str(path))
    deadline=read_json(Path(ledger_path).parent/'delivery_deadline.json')
    flagged=[];root_verified=[]
    for key,path in accepted.items():
        exact=read_json(Path(path)/'actor_separation.json')
        if exact.get('status')=='separation_verified' and exact.get('exact_root_evidence'):
            root_verified.append(path)
        supplemental=read_json(Path(ledger_path).parent/'separation_audits'/f'scene_{key[0]}_room_{key[1]}.json')
        if supplemental.get('capture')==path and supplemental.get('status')=='clearance_violation':
            flagged.append(path)
    return dict(checked_at=time.time(),deadline_at=deadline.get('deadline_at'),
        seconds_remaining=round(deadline.get('deadline_at',time.time())-time.time()),
        accepted_videos=len(accepted),accepted_episodes=list(accepted.values()),
        accepted_by_viewpoint=dict(collections.Counter(read_json(Path(path)/'config.json').get('viewpoint','third_person') for path in accepted.values())),
        supplementary_clearance_flagged_episodes=flagged,
        synchronized_root_verified_episodes=root_verified,
        final_qa_status='Verify the merged release separately from per-episode candidate questions',
        live_owned_jobs=live,probe_progress=probe_progress,
        latest_rooms=sorted(latest.values(),key=lambda row:(row['scene'],row['room_id'])))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=summarize(args.ledger)
    if args.output:args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
