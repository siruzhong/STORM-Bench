#!/usr/bin/env python3
"""Test native approach and reversible operations before planning captures."""
import argparse
import json
import math
import shutil
from pathlib import Path
import sys
import time
import requests
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.room_enrichment import reset_enriched_scene
from storm_virtualhome.room_preflight import (
    spawn_points, probe_candidates, probe_result_ready, switch_animation_ready,
)
from storm_virtualhome.room_program import native_probe_ready
from storm_virtualhome.scene import action, position, room_id_of, support_of
from storm_virtualhome.mixed import check_effect


SAFE_ANIMATED_SWITCHES={
    'computer','tv','lightswitch','radio','faucet',
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--executable',required=True)
    p.add_argument('--xorg-root',type=Path)
    p.add_argument('--gpu-index',type=int,default=3)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--limit',type=int,default=14)
    p.add_argument('--room-index',type=int)
    p.add_argument('--target-id',type=int,action='append')
    p.add_argument('--skip-infrastructure-targets',action='store_true')
    p.add_argument('--skip-failed-targets',action='store_true')
    p.add_argument('--action-timeout',type=int,default=75)
    p.add_argument('--time-scale',type=float,default=8.0)
    p.add_argument('--direct-attempts',type=int,default=4)
    p.add_argument('--solver-attempts',type=int,default=2)
    p.add_argument('--record-probes',action='store_true')
    p.add_argument('--allow-visual-switch-fallback',action='store_true')
    p.add_argument('--require-animated',action='store_true')
    p.add_argument('--character-resource',default='Chars/Male1')
    p.add_argument('--max-operation-seconds',type=float,default=5.0)
    args=p.parse_args()
    if args.direct_attempts<1 or args.solver_attempts<1:
        p.error('attempt limits must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((args.preflight/'status.json').read_text())
    status=dict(status='running',started_at=time.time(),rooms=[])
    if (args.output/'status.json').exists():
        status=json.loads((args.output/'status.json').read_text())
        status.update(status='running',resumed_at=time.time())
    def save():
        status['updated_at']=time.time()
        temp=args.output/'status.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(args.output/'status.json')
    save()
    blocked=[]
    for index,room in enumerate(catalog['rooms']):
        if index%args.shards!=args.shard:continue
        if args.room_index is not None and index!=args.room_index:continue
        with UnityProcess(args.executable,args.output/'logs'/f'room_{index}_{int(time.time())}',gpu_index=args.gpu_index,xorg_root=args.xorg_root) as c:
            c.timeout=args.action_timeout
            source=json.loads((args.preflight/f"scene_{room['scene']}.json").read_text())
            additions_path=args.preflight/f"additions_{room['scene']}.json"
            additions=json.loads(additions_path.read_text()) if additions_path.exists() else None
            points=spawn_points(source,room['room_id'])
            entry=next((r for r in status['rooms'] if (r['scene'],r['room_id'])==(room['scene'],room['room_id'])),None)
            if entry is None:
                entry={k:room[k] for k in ('scene','room_id','room')};entry.update(targets=[])
                status['rooms'].append(entry)
            entry['status']='running';save()
            def ready():
                checked=dict(entry,targets=[dict(t,operations=[o for o in t['operations']
                    if not args.require_animated or o.get('animation_verified')]) for t in entry['targets']])
                return probe_result_ready(checked['targets']) and native_probe_ready(source,checked)
            candidates=probe_candidates(room,max(args.limit,len(room['candidates'])))
            if args.target_id is not None:
                requested=set(args.target_id)
                candidates=[candidate for candidate in candidates if candidate['id'] in requested]
                missing=requested-{candidate['id'] for candidate in candidates}
                if missing:raise ValueError(f'Requested probe targets are absent: {sorted(missing)}')
            else:
                candidates=candidates[:args.limit]
            for candidate in candidates:
                target=candidate['id']
                old=next((t for t in entry['targets'] if t['id']==target),None)
                if old and any(o['accepted'] and (not args.require_animated or o.get('animation_verified'))
                               for o in old['operations']):continue
                if old and old.get('infrastructure_failure') and args.skip_infrastructure_targets:continue
                if (old and args.skip_failed_targets and old.get('operations')
                        and not old.get('infrastructure_failure')):continue
                if old:
                    entry.setdefault('previous_trials',[]).append(old)
                    entry['targets'].remove(old)
                node=next(n for n in source['nodes'] if n['id']==target)
                trial=dict(id=target,name=node['class_name'],probe_family=candidate.get('probe_family'),operations=[])
                trial['animation_attempted']=args.require_animated
                entry['targets'].append(trial);save()
                if not points:
                    trial['failure']='No geometric spawn proposal';save();continue
                origins=sorted(points,key=lambda point:math.dist(point,position(node)))[:8]
                try:
                    def run(lines,animate=False,solve=False):
                        folder=args.output/'native'
                        folder.mkdir(exist_ok=True)
                        started=time.monotonic()
                        old_timeout=c.timeout
                        if animate:c.timeout=max(c.timeout,90)
                        try:
                            response=c.render(lines,folder,'probe',record=args.record_probes or animate,width=160,height=120,
                                              time_scale=args.time_scale,
                                              skip_animation=not (args.record_probes or animate),find_solution=solve,
                                              modalities=['normal','seg_inst'] if animate else None)
                        finally:
                            c.timeout=old_timeout
                        duration=time.monotonic()-started
                        frames=sorted(folder.rglob('*_normal.png'))
                        frame_delta=0.0
                        target_delta=0.0
                        shared_pixels=0
                        if len(frames)>=2:
                            with Image.open(frames[0]) as image:
                                first=np.asarray(image.convert('RGB'),dtype=float)
                            with Image.open(frames[-1]) as image:
                                last=np.asarray(image.convert('RGB'),dtype=float)
                            frame_delta=float(np.abs(first-last).mean())
                            first_mask_path=frames[0].with_name(frames[0].name.replace('_normal.png','_seg_inst.png'))
                            last_mask_path=frames[-1].with_name(frames[-1].name.replace('_normal.png','_seg_inst.png'))
                            if first_mask_path.exists() and last_mask_path.exists():
                                with Image.open(first_mask_path) as image:
                                    first_mask=np.asarray(image.convert('RGB'),dtype=np.int16)
                                with Image.open(last_mask_path) as image:
                                    last_mask=np.asarray(image.convert('RGB'),dtype=np.int16)
                                color=np.asarray(c.colors()[str(target)],dtype=float)
                                if color.max()<=1:color*=255
                                color=np.rint(color).astype(np.int16)
                                shared=(np.all(np.abs(first_mask-color)<=1,axis=-1)
                                        & np.all(np.abs(last_mask-color)<=1,axis=-1))
                                shared_pixels=int(shared.sum())
                                if shared_pixels:
                                    target_delta=float(np.abs(first-last)[shared].mean())
                        timing=dict(actions=lines,
                            animation='full' if animate else 'skipped',find_solution=solve,
                            duration_seconds=duration,recorded_frames=len(frames),mean_frame_delta=frame_delta,
                            target_shared_pixels=shared_pixels,target_mean_frame_delta=target_delta)
                        timing['video_seconds']=len(frames)/20.0
                        trial.setdefault('action_timings',[]).append(timing)
                        shutil.rmtree(folder)
                        statuses=json.loads(response.get('message','{}'))
                        if not statuses or any(s.get('message')!='Success' for s in statuses.values()):
                            raise RuntimeError(f'{lines}: {statuses}')
                        return timing
                    operation_pairs=[('Open','Close'),('SwitchOn','SwitchOff'),('Grab','PutBack')]
                    preferred=candidate.get('probe_family')
                    operation_pairs.sort(key=lambda pair:pair[0]!=preferred)
                    for first,second in operation_pairs:
                        if first not in candidate['operations']:continue
                        result=dict(first=first,second=second,accepted=False)
                        trial['operations'].append(result)
                        failures=[]
                        attempts=[]
                        for origin in origins[:2]:
                            reset_enriched_scene(c,room['scene'],additions)
                            initial=c.graph();initial_nodes={n['id']:n for n in initial['nodes']}
                            initial_target=initial_nodes[target]
                            initial_support=support_of(initial,target)
                            anchors=[initial_target]
                            if initial_support is not None and initial_support['class_name'] not in ('floor','wall','ceiling'):
                                anchors.append(initial_support)
                            nearby=[n for n in initial['nodes'] if room_id_of(initial,n['id'])==room['room_id']
                                    and n['id'] not in {a['id'] for a in anchors}
                                    and n['class_name'] not in ('floor','wall','ceiling')
                                    and n.get('category','').lower()!='rooms'
                                    and 'GRABBABLE' not in n.get('properties',[])
                                    and {'SURFACES','CAN_OPEN','SITTABLE'} & set(n.get('properties',[]))
                                    and max(n.get('bounding_box',{}).get('size',[0,0,0]))>=.45]
                            anchors.extend(sorted(nearby,key=lambda n:math.dist(position(n),position(initial_target)))[:5])
                            attempts.extend((origin,anchor['id']) for anchor in anchors)
                        def execute(origin,approach_id,animate,solve=False):
                            reached=False;approach_name=None
                            try:
                                reset_enriched_scene(c,room['scene'],additions)
                                c.follow_camera(position=[0,2.4,-2.4],rotation=[28,0,0],fov=75)
                                c.character(resource=args.character_resource,room=room['room'],position=origin)
                                before=c.graph();nodes={n['id']:n for n in before['nodes']}
                                current=nodes[target]
                                actual_first,actual_second=first,second
                                if actual_first=='Open' and 'OPEN' in current['states']:
                                    actual_first,actual_second=actual_second,actual_first
                                if actual_first=='SwitchOn' and 'ON' in current['states']:
                                    actual_first,actual_second=actual_second,actual_first
                                support=support_of(before,target)
                                if actual_second=='PutBack' and support is None:
                                    raise RuntimeError('Portable target has no native support surface')
                                approach=nodes[approach_id]
                                approach_name=approach['class_name']
                                event=dict(object_id=target,verb=actual_first,injector_node_id=1,
                                           surface_id=support['id'] if support else None)
                                walk=action('Walk',approach,character=0)
                                operation=action(actual_first,current,character=0)
                                if animate:
                                    approach_timing=run([walk],args.require_animated,solve);reached=True
                                    reached_position=next(n for n in c.graph()['nodes'] if n['id']==1)['obj_transform']['position']
                                    forward=run([operation],True,solve)
                                else:
                                    forward=run([walk,operation],False,solve);reached=True
                                after=c.graph();verification='graph_state'
                                try:check_effect(before,after,event)
                                except ValueError:
                                    if (actual_first not in ('SwitchOn','SwitchOff') or not animate
                                            or not args.allow_visual_switch_fallback
                                            or not switch_animation_ready(forward)):
                                        raise
                                    verification='native_recorded_animation'
                                approach_position=(reached_position if animate else
                                    next(n for n in after['nodes'] if n['id']==1)['obj_transform']['position'])
                                reverse=action(actual_second,current,*([support] if actual_second=='PutBack' else []),character=0)
                                if animate:
                                    if not args.require_animated:run([walk],False,solve)
                                    backward=run([reverse],True,solve)
                                else:
                                    backward=run([walk,reverse],False,solve)
                                event['verb']=actual_second
                                try:check_effect(after,c.graph(),event)
                                except ValueError:
                                    if (actual_second not in ('SwitchOn','SwitchOff') or not animate
                                            or not args.allow_visual_switch_fallback
                                            or not switch_animation_ready(backward)):
                                        raise
                                    verification='native_recorded_animation'
                                if args.require_animated:
                                    durations=[forward['video_seconds'],backward['video_seconds']]
                                    if min(durations)<=0 or max(durations)>args.max_operation_seconds:
                                        raise ValueError(f'Animated operation durations exceed the planning budget: {durations}')
                                result.update(first=actual_first,second=actual_second,accepted=True,
                                              character_resource=args.character_resource,
                                              spawn=origin,approach_position=approach_position,
                                              approach_id=approach['id'],find_solution=solve,
                                              state_verification=verification)
                                if args.require_animated:
                                    result.update(animation_verified=True,
                                        verification_scope='single_operator_full_animation',
                                        animation_time_scale=args.time_scale,animation_fps=20,
                                        approach_video_seconds=approach_timing['video_seconds'],
                                        operation_video_seconds={actual_first:forward['video_seconds'],
                                                                 actual_second:backward['video_seconds']})
                                trial.setdefault('spawn',origin)
                                trial.setdefault('approach_position',approach_position)
                                trial.setdefault('approach_id',approach['id'])
                                return True,reached
                            except requests.RequestException:
                                raise
                            except Exception as exc:
                                failures.append(dict(spawn=origin,approach_id=approach_id,
                                                     approach=approach_name,
                                                     animation='full' if animate else 'skipped',
                                                     find_solution=solve,
                                                     failure=str(exc)))
                                return False,reached
                        reachable=[]
                        for origin,approach_id in attempts[:args.direct_attempts]:
                            passed,reached=execute(origin,approach_id,args.require_animated)
                            if reached and (origin,approach_id) not in reachable:
                                reachable.append((origin,approach_id))
                            if passed or (first=='SwitchOn' and len(reachable)>=2):break
                        if (not args.require_animated and first=='SwitchOn' and not result['accepted']
                                and (node['class_name'] in SAFE_ANIMATED_SWITCHES
                                     or args.allow_visual_switch_fallback)):
                            for origin,approach_id in reachable[:2]:
                                passed,_=execute(origin,approach_id,True)
                                if passed:break
                        if not args.require_animated and first!='SwitchOn' and not result['accepted']:
                            for origin,approach_id in attempts[:args.solver_attempts]:
                                passed,_=execute(origin,approach_id,False,True)
                                if passed:break
                        if not result['accepted']:
                            result['failure']=failures[-1]['failure'] if failures else 'No operation trial ran'
                            result['trials']=failures
                        save()
                    if ready():
                        break
                except requests.RequestException as exc:
                    trial['infrastructure_failure']=str(exc)
                    entry['status']='interrupted'
                    status['status']='interrupted'
                    save()
                    raise
                except Exception as exc:trial['failure']=str(exc)
                save()
                print(json.dumps(dict(scene=room['scene'],room_id=room['room_id'],target=trial)),flush=True)
            if ready():
                entry['status']='complete'
            else:
                entry['status']='blocked'
                entry['failure']='Fewer than three targets or four reversible verbs passed native execution'
                blocked.append((room['scene'],room['room_id']))
            save()
    status['status']='completed_with_errors' if blocked else 'complete';save()
    if blocked:raise SystemExit(2)


if __name__=='__main__':main()
