#!/usr/bin/env python3
"""Extend one immutable scene recipe with natural props and verify exact replay."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.room_enrichment import apply_additions, propose_additions, reset_enriched_scene
from preflight_rooms import room_candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',type=int,required=True)
    parser.add_argument('--room',type=int,required=True)
    parser.add_argument('--props',nargs='+',required=True)
    parser.add_argument('--executable',required=True)
    parser.add_argument('--xorg-root',type=Path,required=True)
    parser.add_argument('--gpu-index',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('The destination must be new')
    args.output.mkdir(parents=True)
    for path in args.source.glob('*.json'):shutil.copy2(path,args.output/path.name)
    catalog=json.loads((args.source/'status.json').read_text())
    recipe=json.loads((args.source/f'additions_{args.scene}.json').read_text())
    templates={}
    for path in sorted(args.source.glob('scene_*.json')):
        for node in json.loads(path.read_text())['nodes']:
            name=node['class_name']
            footprint=lambda item:max(item['bounding_box']['size'][axis] for axis in (0,2))
            if name not in templates or footprint(node)<footprint(templates[name]):templates[name]=node
    with UnityProcess(args.executable,args.output/'enrichment_logs',gpu_index=args.gpu_index,xorg_root=args.xorg_root) as client:
        reset_enriched_scene(client,args.scene,recipe)
        graph=client.graph()
        proposed=propose_additions(graph,templates,room_prop_classes={args.room:tuple(args.props)})
        keep={item['object_id'] for item in proposed['placements']
              if item['room_id']==args.room and item['class_name'] in args.props}
        if not keep:raise ValueError('No unobstructed support surface is available')
        combined=copy.deepcopy(recipe)
        combined.pop('id_mapping',None)
        combined['nodes'] += [item for item in proposed['nodes'] if item['id'] in keep]
        combined['edges'] += [item for item in proposed['edges'] if item['from_id'] in keep]
        combined['placements'] += [item for item in proposed['placements'] if item['object_id'] in keep]
        (args.output/'proposed_additions.json').write_text(json.dumps(combined,indent=2))
        client.reset(args.scene)
        try:
            graph,response=apply_additions(client,combined)
        except Exception:
            (args.output/'failed_graph.json').write_text(json.dumps(client.graph()))
            raise
        combined['id_mapping']=response['id_mapping']
        for placement in combined['placements']:
            placement['native_object_id']=combined['id_mapping'][str(placement['object_id'])]
        colors=client.colors()
    with UnityProcess(args.executable,args.output/'replay_logs',gpu_index=args.gpu_index,xorg_root=args.xorg_root) as client:
        client.reset(args.scene)
        graph,replay=apply_additions(client,combined)
        colors=client.colors()
    for room in catalog['rooms']:
        if room['scene']==args.scene:
            room['candidates']=room_candidates(graph,colors,room['room_id'])
    for name,data in [('status',catalog),(f'additions_{args.scene}',combined),
                      (f'scene_{args.scene}',graph),(f'colors_{args.scene}',colors),
                      ('enrichment_validation',dict(scene=args.scene,room=args.room,added=len(keep),replay=replay))]:
        (args.output/f'{name}.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(added=len(keep),mapping=combined['id_mapping'])),flush=True)


if __name__=='__main__':main()
