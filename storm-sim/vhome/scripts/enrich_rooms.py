#!/usr/bin/env python3
"""Validate natural prop additions and publish a new native preflight catalog."""
import argparse
import json
import math
import ast
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.room_enrichment import propose_additions,apply_additions
from preflight_rooms import room_candidates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--executable',required=True)
    p.add_argument('--xorg-root',type=Path)
    p.add_argument('--gpu-index',type=int,default=0)
    p.add_argument('--surface-props-per-room',type=int,default=0)
    p.add_argument('--replay-passes',type=int,default=3)
    p.add_argument('--room-props',action='append',default=[],metavar='SCENE:ROOM:CLASS,...')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.surface_props_per_room<0:p.error('surface prop count cannot be negative')
    if args.replay_passes<1:p.error('replay pass count must be positive')
    targeted={}
    for value in args.room_props:
        try:
            scene_text,room_text,class_text=value.split(':',2)
            key=(int(scene_text),int(room_text))
            classes=tuple(name.strip() for name in class_text.split(',') if name.strip())
        except (ValueError,TypeError):
            p.error(f'invalid targeted prop request: {value}')
        if not classes:p.error(f'targeted prop request has no classes: {value}')
        if key in targeted:p.error(f'duplicate targeted prop request: {scene_text}:{room_text}')
        targeted[key]=classes
    templates={}
    for path in sorted(args.source.glob('scene_*.json')):
        for node in json.loads(path.read_text())['nodes']:templates.setdefault(node['class_name'],node)
    report=dict(status='running',started_at=time.time(),rooms=[],scenes=[])
    def save():(args.output/'status.json').write_text(json.dumps(report,indent=2))
    save()
    with UnityProcess(args.executable,args.output/'logs',gpu_index=args.gpu_index,xorg_root=args.xorg_root) as c:
        for scene in range(7):
            c.reset(scene);before=c.graph()
            room_props={room_id:classes for (target_scene,room_id),classes in targeted.items()
                        if target_scene==scene} if targeted else None
            additions=propose_additions(before,templates,per_room=args.surface_props_per_room,
                                        room_prop_classes=room_props)
            (args.output/f'additions_{scene}.json').write_text(json.dumps(additions,indent=2))
            rejected=[]
            for retry in range(4):
                try:
                    graph,response=apply_additions(c,additions,allow_partial=True)
                    break
                except ValueError as exc:
                    failed=c.graph()
                    (args.output/f'failed_scene_{scene}.json').write_text(json.dumps(failed))
                    if not str(exc).startswith('Unexpected scene objects: ') or retry==3:
                        raise
                    unexpected=set(ast.literal_eval(str(exc).split(': ',1)[1]))
                    blocked=set()
                    for node in failed['nodes']:
                        if node['id'] not in unexpected:continue
                        nearby=[n for n in additions['nodes'] if math.dist(n['bounding_box']['center'],node['bounding_box']['center'])<.4]
                        if not nearby:raise
                        blocked.add(min(nearby,key=lambda n:math.dist(n['bounding_box']['center'],node['bounding_box']['center']))['id'])
                    rejected.extend(dict(node=n,reason='Prototype cloned additional scene objects') for n in additions['nodes'] if n['id'] in blocked)
                    additions['nodes']=[n for n in additions['nodes'] if n['id'] not in blocked]
                    additions['edges']=[e for e in additions['edges'] if e['from_id'] not in blocked]
                    additions['placements']=[e for e in additions['placements'] if e['object_id'] not in blocked]
                    c.reset(scene)
            additions['rejected_proposals']=rejected
            colors=c.colors()
            additions['id_mapping'] = response['id_mapping']
            for placement in additions['placements']:
                placement['native_object_id'] = response['id_mapping'].get(str(placement['object_id']))
            c.reset(scene)
            graph,replay=apply_additions(c,additions,allow_partial=False)
            response['exact_replay']=replay
            (args.output/f'additions_{scene}.json').write_text(json.dumps(additions,indent=2))
            (args.output/f'scene_{scene}.json').write_text(json.dumps(graph))
            (args.output/f'colors_{scene}.json').write_text(json.dumps(colors))
            report['scenes'].append(dict(scene=scene,added=len(additions['id_mapping']),response=response))
            for room in graph['nodes']:
                if room.get('category','').lower()!='rooms':continue
                report['rooms'].append(dict(scene=scene,room_id=room['id'],room=room['class_name'],candidates=room_candidates(graph,colors,room['id'])))
            save();print(json.dumps(report['scenes'][-1]),flush=True)
    for replay_index in range(args.replay_passes):
        log_root=args.output/'independent_replay_logs'/f'pass_{replay_index:02d}'
        with UnityProcess(args.executable,log_root,
                          gpu_index=args.gpu_index,xorg_root=args.xorg_root) as c:
            for scene in range(7):
                additions=json.loads((args.output/f'additions_{scene}.json').read_text())
                c.reset(scene)
                _,replay=apply_additions(c,additions,allow_partial=False)
                response=report['scenes'][scene]['response']
                response.setdefault('independent_replays',[]).append(replay)
                response['independent_replay']=replay
                save()
    placed={(scene['scene'],placement['room_id'],placement['class_name'])
            for scene in report['scenes']
            for placement in json.loads((args.output/f"additions_{scene['scene']}.json").read_text())['placements']}
    missing=[(scene,room,name) for (scene,room),classes in targeted.items() for name in classes
             if (scene,room,name) not in placed]
    report['unplaced_targeted_props']=[dict(scene=scene,room_id=room,class_name=name)
                                       for scene,room,name in missing]
    report['status']='complete';save()


if __name__=='__main__':main()
