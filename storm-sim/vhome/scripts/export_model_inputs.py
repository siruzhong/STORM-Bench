#!/usr/bin/env python3
"""Export prefix videos and keep model inputs separate from evaluation labels."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preset', choices=('fast', 'veryfast'), default='veryfast')
    args = parser.parse_args()
    data = json.loads((args.episode / 'qa.json').read_text())
    source = (args.episode / data['video_path']).resolve()
    fps = data['sample_fps']
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'prefixes').mkdir()
    inputs, labels, clips = [], [], {}
    for q in data['questions']:
        count = int(math.floor(q['query_time'] * fps + 1e-6))
        if count <= 0:
            raise ValueError(f'Empty video prefix for {q["id"]}')
        if count not in clips:
            path = Path('prefixes') / f'{count:06d}.mp4'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-v', 'error',
                            '-i', str(source), '-frames:v', str(count), '-an', '-c:v', 'libx264',
                            '-preset', args.preset, '-crf', '18', '-threads', '2',
                            str(args.output / path)], check=True)
            clips[count] = path.as_posix()
        inputs.append({key: q[key] for key in ('id', 'episode_id', 'query_time', 'question', 'options')})
        inputs[-1]['video_path'] = clips[count]
        labels.append({key: value for key, value in q.items() if key not in ('question', 'options')})
    for name, records in [('model_inputs.jsonl', inputs), ('evaluation_labels.jsonl', labels)]:
        (args.output / name).write_text(''.join(json.dumps(row) + '\n' for row in records))
    print(json.dumps({'questions': len(inputs), 'prefix_videos': len(clips), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
