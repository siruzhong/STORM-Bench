"""Measure all event targets in every recorded segmentation frame."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import argparse
import os
import time
import numpy as np
from PIL import Image


def measure(ep):
    capture = Path(ep['metadata']['capture'])
    events = ep['events']
    objects = sorted({e['object_id'] for e in events})
    colors = {}
    for event in events:
        if event['object_id'] not in colors:
            p = capture/'evidence'/(event['stage_keys']['before']+'.json')
            data = json.loads(p.read_text())
            color = tuple(round(x*255) for x in data['color'])
            same = [int(i) for i,c in data['instance_colors'].items() if tuple(round(v*255) for v in c)==color]
            assert same == [event['object_id']], (event['object_id'],same)
            colors[event['object_id']] = color
    frames = [json.loads(x) for x in (capture/'frame_manifest.jsonl').read_text().splitlines()]
    counts=[]
    for f in frames:
        arr=np.asarray(Image.open(capture/f['mask']).convert('RGB'),dtype=np.uint32)
        packed=(arr[:,:,0]<<16)|(arr[:,:,1]<<8)|arr[:,:,2]
        counts.append([int(np.count_nonzero(packed==((colors[o][0]<<16)|(colors[o][1]<<8)|colors[o][2]))) for o in objects])
    result=dict(episode_id=ep['metadata']['episode_id'],objects=objects,counts=counts,
                fps=ep['qa']['sample_fps'],frames=len(frames),
                manifest_sha256=hashlib.sha256((capture/'frame_manifest.jsonl').read_bytes()).hexdigest())
    print(json.dumps(dict(episode=result['episode_id'],frames=len(frames))),flush=True)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1:
        parser.error('--workers must be positive')
    bundle = json.loads(args.evidence.read_text())
    if not bundle['episodes']:
        raise ValueError('The evidence bundle contains no episodes')
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(bundle['episodes']))) as pool:
        result = list(pool.map(measure, bundle['episodes']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(episodes=result, elapsed_seconds=time.monotonic()-start)))
