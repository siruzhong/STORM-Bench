"""Render paired state examples from actual evaluator-clock frames."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader, cpu


def render_episode(job):
    episode, visibility, video_root, output = job
    key = episode['metadata']['episode_id']
    capture = Path(episode['metadata']['capture'])
    manifest = [json.loads(s) for s in (capture / 'frame_manifest.jsonl').read_text().splitlines()]
    video = VideoReader(str(Path(video_root) / 'episodes' / (key + '.mp4')), ctx=cpu(0), num_threads=1)
    fps = visibility['fps']
    events = {e['object_id']: e for e in episode['events']}
    options = defaultdict(list)
    for o in episode['observations']:
        if o['state'] is None:
            continue
        state = 'resting on a surface' if o['state'].startswith('on:') else o['state']
        col = visibility['objects'].index(o['object_id'])
        for t in range(int((visibility['frames'] - 1) / fps) + 1):
            frame = round(t * fps)
            if o['state_interval'][0] <= t < o['state_interval'][1]:
                pixels = visibility['counts'][frame][col]
                if pixels >= 40:
                    options[o['object_id'], state].append((pixels, frame, o))
    records = []
    for obj, event in events.items():
        examples = []
        for (oid, state), candidates in sorted(options.items()):
            if oid != obj:
                continue
            pixels, frame, o = max(candidates, key=lambda x: (x[0], -x[1]))
            source = json.loads(Path(o['source_json']).read_text())
            color = tuple(round(x * 255) for x in source['color'])
            mask = np.asarray(Image.open(capture / manifest[frame]['mask']).convert('RGB'))
            ys, xs = np.where(np.all(mask == color, axis=2))
            if not len(xs):
                raise ValueError('The recorded instance color does not match its visibility audit')
            box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            image = Image.fromarray(video[frame].asnumpy())
            context = image.copy()
            ImageDraw.Draw(context).rectangle(box, outline='red', width=3)
            context.thumbnail((170, 140))
            x1, y1, x2, y2 = box
            pad = max(20, int(max(x2 - x1, y2 - y1) * .3))
            crop = image.crop((max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad)))
            crop.thumbnail((290, 140))
            tile = Image.new('RGB', (480, 180), 'white')
            d = ImageDraw.Draw(tile)
            d.text((5, 3), f'{state} | t={frame/fps:.0f}s | {pixels} px', fill='black')
            tile.paste(context, (0, 25)); tile.paste(crop, (180, 25))
            destination = Path(output) / f'{key}_{obj}_{len(examples)}.jpg'
            tile.save(destination)
            examples.append(dict(state=state, frame=frame, time=frame/fps, pixels=pixels,
                                 box=box, tile=destination.name))
        records.append(dict(episode_id=key, object_id=obj, name=event['object_description'], examples=examples))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('visibility', type=Path)
    parser.add_argument('video_root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(args.bundle.read_text())
    visibility = {v['episode_id']: v for v in json.loads(args.visibility.read_text())['episodes']}
    jobs = [(ep, visibility[ep['metadata']['episode_id']], str(args.video_root), str(args.output)) for ep in bundle['episodes']]
    with ProcessPoolExecutor(max_workers=8) as pool:
        records = [row for result in pool.map(render_episode, jobs) for row in result]
    for offset in range(0, len(records), 8):
        selected = records[offset:offset+8]
        canvas = Image.new('RGB', (960, 205*len(selected)), (225, 225, 225))
        draw = ImageDraw.Draw(canvas)
        for row, record in enumerate(selected):
            draw.text((5, row*205+5), f'{offset+row:02d} {record["episode_id"]} | {record["object_id"]}: {record["name"]}', fill='black')
            for col, ex in enumerate(record['examples'][:2]):
                canvas.paste(Image.open(args.output / ex['tile']), (480*col, row*205+25))
        canvas.save(args.output / f'page_{offset//8:02d}.jpg')
    (args.output / 'state_examples.json').write_text(json.dumps(records, indent=2))
    print(json.dumps(dict(objects=len(records), pages=(len(records)+7)//8)))


if __name__ == '__main__':
    main()
