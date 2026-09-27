#!/usr/bin/env python3
"""Atomically publish a complete 29-room rollout from verified GPU shards."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rollout_rooms_native import capture_accepted


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    os.replace(temporary,path)


def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20260817)
    parser.add_argument('--render-prefixes',action='store_true')
    args=parser.parse_args()
    catalog=json.loads((args.preflight/'status.json').read_text())['rooms']
    expected=set(range(len(catalog)))
    if len(catalog)!=29:raise ValueError(f'Expected 29 rooms, found {len(catalog)}')
    episodes={};source_revisions=set();shard_contracts=[]
    for root in args.inputs:
        state=json.loads((root/'status.json').read_text())
        if state.get('status')!='complete' or state.get('accepted')!=state.get('target_rooms'):
            raise ValueError(f'Incomplete rollout shard: {root}')
        shard_contracts.append((state.get('shard',0),state.get('shards',1)))
        for room in state['rooms']:
            index=room['catalog_index']
            if index in episodes:raise ValueError(f'Duplicate room index: {index}')
            if room['status']!='accepted':raise ValueError(f'Room {index} was not accepted')
            capture=Path(room['episode']).resolve()
            program=json.loads((capture/'event_program.json').read_text())
            if not capture_accepted(capture,program):
                raise ValueError(f'Capture gate no longer passes for room {index}')
            accepted_attempts=[a for a in room['attempts'] if a.get('status')=='accepted']
            if len(accepted_attempts)!=1:raise ValueError(f'Room {index} has an ambiguous accepted attempt')
            source_revisions.add(accepted_attempts[0]['source_revision'])
            episodes[index]=(room,capture,program,accepted_attempts[0])
    if set(episodes)!=expected:
        raise ValueError(f'Room coverage differs: missing={sorted(expected-set(episodes))}, extra={sorted(set(episodes)-expected)}')
    if len(source_revisions)!=1:raise ValueError(f'Shards use different source revisions: {sorted(source_revisions)}')
    if len(set(shard_contracts))!=len(shard_contracts):raise ValueError('Duplicate shard contract')
    if args.output.exists():raise FileExistsError(args.output)
    stage=args.output.with_name('.'+args.output.name+'.staging')
    if stage.exists():shutil.rmtree(stage)
    (stage/'dataset').mkdir(parents=True)
    manifest=[]
    try:
        for index in sorted(episodes):
            room,capture,program,attempt=episodes[index]
            destination=stage/'dataset'/f'episode_{index:04d}'
            destination.symlink_to(capture,target_is_directory=True)
            manifest.append(dict(catalog_index=index,scene=room['scene'],room_id=room['room_id'],
                room=room['room'],seed=attempt['seed'],plan_id=program['plan_id'],capture=str(capture),
                sha256={name:sha256(capture/name) for name in ('video.mp4','debug.mp4','qa.json','event_program.json')}))
        write_json(stage/'manifest.json',dict(schema_version=1,rooms=29,qa_per_room=10,
            source_revision=next(iter(source_revisions)),shards=sorted(shard_contracts),episodes=manifest))
        command=[sys.executable,str(Path(__file__).with_name('rebuild_batch_qa.py')),
                 '--input',str(stage),'--output',str(stage/'structured_qa'),'--seed',str(args.seed)]
        if args.render_prefixes:command.append('--render-prefixes')
        with (stage/'qa_export.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=[json.loads(line) for line in (stage/'structured_qa/model_inputs.jsonl').read_text().splitlines()]
        counts=Counter(row['episode_id'] for row in rows)
        if len(rows)!=290 or len({row['id'] for row in rows})!=290 or len(counts)!=29 or set(counts.values())!={10}:
            raise ValueError(f'Final QA contract failed: rows={len(rows)}, episodes={dict(counts)}')
        write_json(stage/'status.json',dict(status='complete',accepted=29,target_rooms=29,qa_count=290,
            source_revision=next(iter(source_revisions)),created_at=time.time(),
            prefixes_materialized=args.render_prefixes))
        os.replace(stage,args.output)
    except Exception:
        write_json(stage/'failure.json',dict(status='failed',failed_at=time.time()))
        raise


if __name__=='__main__':main()
