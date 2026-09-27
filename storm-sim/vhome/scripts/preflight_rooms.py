#!/usr/bin/env python3
"""Measure native mask identity before selecting room-local event targets."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.scene import room_id_of, support_of


def room_candidates(graph, colors, room_id):
    members = [n for n in graph['nodes'] if room_id_of(graph, n['id']) == room_id]
    counts = Counter(n['class_name'] for n in members)
    rows = []
    for node in members:
        props = node.get('properties', [])
        operations = []
        if 'CAN_OPEN' in props and node['class_name'] not in ('door', 'window', 'curtains'):
            operations += ['Open', 'Close']
        if 'HAS_SWITCH' in props:
            operations += ['SwitchOn', 'SwitchOff']
        support = support_of(graph, node['id'])
        if 'GRABBABLE' in props and support and max(node['bounding_box']['size']) < .6:
            operations += ['Grab', 'PutBack']
        if not operations:
            continue
        color = colors.get(str(node['id']))
        collisions = [] if color is None else [int(k) for k,v in colors.items()
                    if int(k) != node['id'] and np.allclose(v, color, atol=1e-6)]
        rows.append(dict(id=node['id'], name=node['class_name'], operations=operations,
                         unique_name=counts[node['class_name']] == 1,
                         unique_mask=color is not None and not collisions,
                         mask_collision_ids=collisions, support_id=support['id'] if support else None,
                         states=node['states'], bounds=node['bounding_box']))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, default=3)
    p.add_argument('--xorg-root', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', started_at=time.time(), rooms=[])
    def save():
        status['updated_at'] = time.time()
        tmp=args.output/'status.tmp'
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(args.output/'status.json')
    save()
    with UnityProcess(args.executable, args.output/'logs', gpu_index=args.gpu_index,
                      xorg_root=args.xorg_root) as client:
        for scene in range(7):
            client.reset(scene)
            graph, colors = client.graph(), client.colors()
            (args.output/f'scene_{scene}.json').write_text(json.dumps(graph))
            (args.output/f'colors_{scene}.json').write_text(json.dumps(colors))
            for room in graph['nodes']:
                if room.get('category', '').lower() != 'rooms':
                    continue
                row=dict(scene=scene, room_id=room['id'], room=room['class_name'],
                         candidates=room_candidates(graph, colors, room['id']))
                status['rooms'].append(row)
                save()
                print(json.dumps({k:v for k,v in row.items() if k!='candidates'}), flush=True)
    status['status']='complete'
    save()


if __name__ == '__main__':
    main()
