#!/usr/bin/env python3
"""Rebuild a room catalog from exact simulator replay of selected prop recipes."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.room_enrichment import apply_additions
from preflight_rooms import room_candidates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--override',action='append',default=[],metavar='SCENE:CATALOG')
    p.add_argument('--executable',required=True)
    p.add_argument('--xorg-root',type=Path,required=True)
    p.add_argument('--gpu-index',type=int,default=0)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    overrides={int(value.split(':',1)[0]):Path(value.split(':',1)[1]) for value in args.override}
    report=dict(status='running',started_at=time.time(),rooms=[],scenes=[])
    with UnityProcess(args.executable,args.output/'logs',gpu_index=args.gpu_index,xorg_root=args.xorg_root) as client:
        for scene in range(7):
            source=overrides.get(scene,args.source)
            recipe=json.loads((source/f'additions_{scene}.json').read_text())
            client.reset(scene)
            graph,replay=apply_additions(client,recipe)
            colors=client.colors()
            for name,data in [(f'additions_{scene}',recipe),(f'scene_{scene}',graph),(f'colors_{scene}',colors)]:
                (args.output/f'{name}.json').write_text(json.dumps(data,indent=2))
            report['scenes'].append(dict(scene=scene,recipe_source=str(source),replay=replay))
            for node in graph['nodes']:
                if node.get('category','').lower()=='rooms':
                    report['rooms'].append(dict(scene=scene,room_id=node['id'],room=node['class_name'],
                        candidates=room_candidates(graph,colors,node['id'])))
            (args.output/'status.json').write_text(json.dumps(report,indent=2))
    report.update(status='complete',ended_at=time.time())
    (args.output/'status.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(rooms=len(report['rooms']),status=report['status'])))


if __name__=='__main__':main()
