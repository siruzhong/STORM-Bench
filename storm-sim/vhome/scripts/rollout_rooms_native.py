#!/usr/bin/env python3
"""Consume native room probes and render independently verified room episodes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import yaml
import os
import re

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storm_virtualhome.room_program import build_native_profile, native_probe_ready
from storm_virtualhome.planning import create_program
from storm_virtualhome.room_preflight import probe_result_ready
from rollout_rooms import snapshot_source


def capture_accepted(episode,program,require_qa=True):
    try:
        read=lambda name:json.loads((episode/name).read_text())
        observer=read('observer_validation.json')
        debug=read('debug_validation.json')
        motion=read('motion_validation.json')
        qa=read('qa.json') if require_qa else None
        runtime=read('runtime_manifest.json') if (episode/'runtime_manifest.json').exists() else {}
        if runtime.get('per_frame_actor_positions'):
            separation=read('actor_separation.json')
            if (separation['status']!='separation_verified'
                    or not separation['exact_root_evidence']
                    or separation['covered_frames']!=observer['frames']):return False
        active=(motion['active_frame_fraction'] if motion.get('acceptance_mode')=='locomotion_and_turning'
                else motion['moving_frame_fraction'])
        longest=(motion['longest_inactive_seconds'] if motion.get('acceptance_mode')=='locomotion_and_turning'
                 else motion['longest_stationary_seconds'])
        return (read('event_program.json')==program
                and read('plan_execution.json')['accepted']
                and read('camera_continuity.json')['accepted']
                and observer['event_count']==10
                and 5<=observer['offscreen_event_count']<=6
                and 60<=observer['duration_sec']<=70
                and observer['camera_overlap_samples']==0
                and active>=.8 and longest<=2.0
                and debug['frames']==observer['frames']
                and (not require_qa or (qa.get('qa_stage') == 'raw'
                     and len(qa['questions']) == 10
                     and {q['event_index'] for q in qa['questions']} == set(range(10)))
                     or (require_qa and qa.get('qa_stage') != 'raw' and len(qa['questions']) == 60))
                and all((episode/name).stat().st_size>0 for name in ('video.mp4','debug.mp4')))
    except (OSError,ValueError,KeyError):
        return False


def publish_link(destination,capture):
    destination.parent.mkdir(exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve()==capture.resolve():return
        raise ValueError(f'Published episode already points elsewhere: {destination}')
    temporary=destination.with_name('.'+destination.name+'.staging')
    if temporary.exists() or temporary.is_symlink():temporary.unlink()
    temporary.symlink_to(capture.resolve(),target_is_directory=True)
    os.replace(temporary,destination)


def merge_probe_entry(existing, incoming):
    """Combine independently verified targets without losing earlier successes."""
    if existing is None:
        return copy.deepcopy(incoming)
    merged=copy.deepcopy(existing)
    targets={target['id']:target for target in merged.get('targets',[])}
    for candidate in incoming.get('targets',[]):
        current=targets.get(candidate['id'])
        if current is None:
            targets[candidate['id']]=copy.deepcopy(candidate)
            continue
        operations={(item.get('first'),item.get('second')):item
                    for item in current.get('operations',[])}
        for operation in candidate.get('operations',[]):
            key=(operation.get('first'),operation.get('second'))
            if key not in operations or operation.get('accepted'):
                operations[key]=copy.deepcopy(operation)
        current['operations']=list(operations.values())
        current['animation_attempted']=current.get('animation_attempted',False) or candidate.get('animation_attempted',False)
        if candidate.get('infrastructure_failure') and not current.get('infrastructure_failure'):
            current['infrastructure_failure']=candidate['infrastructure_failure']
    merged['targets']=list(targets.values())
    merged['status']='complete' if probe_result_ready(merged['targets']) else incoming.get('status',existing.get('status'))
    if merged['status']=='complete':merged.pop('failure',None)
    return merged


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--probes',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--executable',required=True)
    p.add_argument('--xorg-root',type=Path)
    p.add_argument('--gpu-index',type=int,default=3)
    p.add_argument('--attempts',type=int,default=3)
    p.add_argument('--attempt-timeout',type=int,default=1800)
    p.add_argument('--seed',type=int,default=20260920)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--defer-qa',action='store_true',
                   help='Defer batch curation and prefix export; per-video draft QA is always generated')
    p.add_argument('--require-animated-probes',action='store_true')
    p.add_argument('--native-navigation',action='store_true')
    args=p.parse_args()
    if args.shards<1 or not 0<=args.shard<args.shards:raise ValueError('Invalid shard selection')
    args.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lease=(args.output/'worker.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    project=Path(__file__).resolve().parents[1]
    hashes={str(f.relative_to(project)):hashlib.sha256(f.read_bytes()).hexdigest() for name in ('src','scripts') for f in (project/name).rglob('*.py')}
    revision=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
    source=args.output/'sources'/revision
    source.parent.mkdir(exist_ok=True)
    if not source.exists():snapshot_source(project,source)
    (source/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    template=yaml.safe_load((source/'configs/randomized.yaml').read_text())
    catalog=json.loads((args.preflight/'status.json').read_text())
    selected=[(index,r) for index,r in enumerate(catalog['rooms']) if index%args.shards==args.shard]
    state=dict(status='running',started_at=time.time(),target_rooms=len(selected),catalog_rooms=len(catalog['rooms']),
        shard=args.shard,shards=args.shards,accepted=0,
        rooms=[dict(catalog_index=index,scene=r['scene'],room_id=r['room_id'],room=r['room'],
                    status='waiting_for_native_probe',attempts=[]) for index,r in selected])
    if (args.output/'status.json').exists():
        state=json.loads((args.output/'status.json').read_text())
        if (state.get('shard',0),state.get('shards',1))!=(args.shard,args.shards):
            raise ValueError('The saved rollout uses a different shard contract')
        state.update(status='running',resumed_at=time.time())
    last_progress=None
    def save():
        nonlocal last_progress
        signature=json.dumps(state['rooms'],sort_keys=True)
        if signature!=last_progress:
            state['progress_at']=time.time();last_progress=signature
        state['pid']=os.getpid()
        state['accepted']=sum(room['status']=='accepted' for room in state['rooms'])
        state['updated_at']=time.time();temporary=args.output/'status.tmp'
        temporary.write_text(json.dumps(state,indent=2));temporary.replace(args.output/'status.json')
    save()
    while True:
        if (args.output/'pause_requested').exists():
            state['status']='paused_for_repair';save();return
        probes={};complete=True
        for path in args.probes:
            if not path.exists():complete=False;continue
            report=json.loads(path.read_text());complete&=report['status'] in ('complete','completed_with_errors','interrupted','blocked','paused_for_repair','superseded_by_enriched_scene_run')
            for room in report['rooms']:
                key=(room['scene'],room['room_id'])
                probes[key]=merge_probe_entry(probes.get(key),room)
        worked=False
        for entry in state['rooms']:
            index=entry['catalog_index']
            if entry['status']=='accepted' or sum(a.get('source_revision')==revision for a in entry['attempts'])>=args.attempts:continue
            probe=probes.get((entry['scene'],entry['room_id']))
            if not probe:continue
            graph=json.loads((args.preflight/f"scene_{entry['scene']}.json").read_text())
            excluded={attempt['excluded_target_id'] for attempt in entry['attempts']
                      if attempt.get('source_revision')==revision and 'excluded_target_id' in attempt}
            if excluded:
                filtered=copy.deepcopy(probe)
                filtered['targets']=[target for target in probe['targets'] if target['id'] not in excluded]
                if native_probe_ready(graph,filtered):
                    probe=filtered
            if args.require_animated_probes:
                probe=copy.deepcopy(probe)
                for target in probe['targets']:
                    target['operations']=[operation for operation in target['operations']
                                          if operation.get('animation_verified')]
            graph=json.loads((args.preflight/f"scene_{entry['scene']}.json").read_text())
            attempt=len(entry['attempts'])
            seed=args.seed+index*100+attempt
            try:
                config=build_native_profile(graph,template,probe,seed)
                if args.native_navigation:config['native_navigation']=True
                additions_path=args.preflight/f"additions_{entry['scene']}.json"
                if additions_path.exists():
                    config['scene_additions']=json.loads(additions_path.read_text())
                program=create_program(graph,config)
            except ValueError as exc:
                entry.update(status='waiting_for_native_targets' if probe['status']!='complete' else 'insufficient_native_targets',reason=str(exc));save();continue
            name=f"scene_{entry['scene']}_{entry['room']}_{entry['room_id']}"
            folder=args.output/'attempts'/name/f'{attempt+1:02d}';folder.mkdir(parents=True)
            (folder/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
            (folder/'event_program.json').write_text(json.dumps(program,indent=2))
            record=dict(source_revision=revision,seed=seed,plan_id=program['plan_id'],status='running',started_at=time.time(),path=str(folder))
            entry['attempts'].append(record);entry['status']='rendering';save()
            cmd=[sys.executable,str(source/'scripts/rollout_native_room.py'),'--config',str(folder/'config.yaml'),
                '--plan',str(folder/'event_program.json'),'--output',str(folder/'capture'),
                '--executable',args.executable,'--gpu-index',str(args.gpu_index)]
            if args.xorg_root:cmd+=['--xorg-root',str(args.xorg_root)]
            if args.defer_qa:cmd+=['--defer-qa']
            with (folder/'render.log').open('w') as log:
                process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                record['child_pid']=process.pid;save()
                try:code=process.wait(timeout=args.attempt_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGINT)
                    try:process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    code=124
                result=subprocess.CompletedProcess(cmd,code)
            passed=result.returncode==0 and capture_accepted(folder/'capture',program,require_qa=not args.defer_qa)
            record.update(status='accepted' if passed else 'rejected',exit_code=result.returncode,
                          capture_gate=passed,ended_at=time.time())
            if passed:
                destination=args.output/'dataset'/f'episode_{index:04d}';destination.parent.mkdir(exist_ok=True)
                publish_link(destination,folder/'capture')
                entry.update(status='accepted',episode=str(destination))
            else:
                record['failure']=(folder/'render.log').read_text()[-3500:]
                entry['status']='rejected'
                failure=record['failure']
                event_index=None
                if 'No frozen approach provides a verified initial target view' in failure:
                    event_index=0
                # A timing failure does not establish that the target is unusable.
                if event_index is not None:
                    record['excluded_target_id']=program['events'][event_index]['object_id']
                    record['exclusion_scope']='initial_target_visibility'
                if 'Capture initialization failed:' in record['failure'] or 'entirely black frame' in record['failure']:
                    state['status']='blocked_renderer'
                    save()
                    return
            worked=True;save();print(json.dumps(entry),flush=True)
        if (args.output/'reload_requested').exists():
            (args.output/'reload_requested').unlink()
            os.execv(sys.executable,[sys.executable,*sys.argv])
        if complete and not worked:break
        if not worked:time.sleep(10)
    state['status']='complete' if state['accepted']==state['target_rooms'] else 'completed_with_rejections'
    state['qa_status']=('raw_ready_batch_curation_deferred' if args.defer_qa and state['accepted']==state['target_rooms']
                        else 'pending' if state['accepted'] else 'no_accepted_captures')
    save()
    if state['accepted']==state['target_rooms'] and not args.defer_qa:
        with (args.output/'qa_export.log').open('w') as log:
            result=subprocess.run([sys.executable,str(source/'scripts/rebuild_batch_qa.py'),'--input',str(args.output),'--output',str(args.output/'structured_qa'),'--render-prefixes'],stdout=log,stderr=subprocess.STDOUT)
        state['qa_status']='exported_development_candidate' if result.returncode==0 else 'selection_failed'
        save()


if __name__=='__main__':main()
