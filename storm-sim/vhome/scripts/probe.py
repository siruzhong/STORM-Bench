#!/usr/bin/env python3
"""Inspect a scene, save rear-camera frames, and optionally record one walk."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PIL import Image
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.scene import candidates, action


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scene', type=int, default=0)
    p.add_argument('--display')
    p.add_argument('--xorg-root', type=Path, help='Extracted Xorg runtime for NVIDIA rendering')
    p.add_argument('--gpu-index', type=int, default=3)
    p.add_argument('--walk', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with UnityProcess(args.executable, args.output / 'logs', gpu_index=args.gpu_index, display=args.display, xorg_root=args.xorg_root) as c:
        c.reset(args.scene)
        c.follow_camera([0.25, 2.15, -1.7], [25, 0, 0], 75)
        before = c.command('camera_count')['value']
        print('CAMERAS', c.command('character_cameras'), 'STATIC', before, flush=True)
        c.character()
        graph = c.graph()
        (args.output / 'scene_graph.json').write_text(json.dumps(graph, indent=2))
        colors = c.colors()
        (args.output / 'instance_colors.json').write_text(json.dumps(colors, indent=2))
        count = c.command('camera_count')['value']
        for index in range(count - 1, count):
            Image.fromarray(c.image(index)).save(args.output / f'camera_{index}.png')
        containers, objects = candidates(graph)
        print('CONTAINERS', [(n['id'], n['class_name']) for n in containers], flush=True)
        print('TARGETS', [(n['id'], n['class_name'], s['class_name']) for n, s in objects], flush=True)
        (args.output / 'candidates.json').write_text(json.dumps({'containers': containers, 'objects': objects}, indent=2))
        if args.walk and objects:
            print(c.render([action('Walk', objects[0][0])], args.output / 'walk', 'walk'), flush=True)
            for index in range(count - 1, count):
                Image.fromarray(c.image(index)).save(args.output / f'after_walk_{index}.png')
        print('PROBE_DONE', flush=True)


if __name__ == '__main__':
    main()
