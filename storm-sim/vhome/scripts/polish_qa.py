#!/usr/bin/env python3
"""Polish question wording through a user-provided multimodal chat endpoint."""
import argparse
import base64
import io
from importlib.resources import files
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.judgment import qa_request, review, rewrite_supported

import imageio.v2 as imageio
import requests
from PIL import Image

SYSTEM = files('storm_virtualhome').joinpath('prompts/qa_polish.txt').read_text(encoding='utf-8').strip()


def make_request(question, frames, model):
    content = [{'type': 'text', 'text': json.dumps({
        'question': question['question'], 'options': question['options'],
        'query_time': question['query_time']})}]
    for timestamp, frame in frames:
        stream = io.BytesIO()
        Image.fromarray(frame).save(stream, format='JPEG', quality=85)
        data = base64.b64encode(stream.getvalue()).decode('ascii')
        content.extend([{'type': 'text', 'text': f'Frame at {timestamp:.3f} seconds'},
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{data}'}}])
    return {'model': model, 'temperature': 0, 'messages': [
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Episode qa.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', required=True, help='Full URL of a compatible chat endpoint')
    p.add_argument('--model', required=True)
    p.add_argument('--api-key-env', default='VLM_API_KEY')
    p.add_argument('--frames', type=int, default=8)
    p.add_argument('--dry-run', action='store_true', help='Save request bodies without calling the API')
    p.add_argument('--jev-check', action='store_true', help='Keep original wording when Jev review is uncertain')
    p.add_argument('--jev-model', default='jev-latest')
    args = p.parse_args()
    if args.frames < 1:
        p.error('--frames must be positive')
    if args.output.exists():
        p.error('Output already exists')
    if args.jev_check and not args.dry_run and not os.environ.get('TYPESAFE_API_KEY'):
        p.error('Set TYPESAFE_API_KEY for --jev-check')
    source = json.loads(args.input.read_text())
    video = args.input.parent / source['video_path']
    reader = imageio.get_reader(str(video))
    fps = reader.get_meta_data()['fps']
    count = reader.count_frames()
    key = os.environ.get(args.api_key_env)
    if not args.dry_run and not key:
        p.error(f'Set {args.api_key_env} to your API key')
    args.output.mkdir(parents=True)
    session = requests.Session()
    try:
        for q in source['questions']:
            # The frame at query_time belongs to the next prefix, so exclude it.
            last = min(count - 1, max(0, int(q['query_time'] * fps + 1e-6) - 1))
            indices = sorted({round(i * last / max(1, args.frames - 1)) for i in range(args.frames)})
            frames = [(index / fps, reader.get_data(index)) for index in indices]
            body = make_request(q, frames, args.model)
            if args.dry_run:
                (args.output / f'{q["id"]}.request.json').write_text(json.dumps(body))
                continue
            response = session.post(args.endpoint, json=body, headers={'Authorization': f'Bearer {key}'}, timeout=180)
            response.raise_for_status()
            result = response.json()
            text = result['choices'][0]['message']['content']
            rewritten = json.loads(text)
            if set(rewritten) != {'question'} or not isinstance(rewritten['question'], str) or not rewritten['question'].strip():
                raise ValueError(f'Invalid rewrite for {q["id"]}')
            q['original_question'] = q['question']
            q['question'] = rewritten['question'].strip()
            q['wording_source'] = args.model
            if args.jev_check:
                judgment = review(qa_request(q, args.jev_model))
                supported = rewrite_supported(judgment)
                judgment['rewrite_retained'] = supported
                judgment['rewrite_confidence_floor'] = .8
                judgment['automatic_action'] = 'keep_rewrite' if supported else 'restore_original_wording'
                (args.output / f'{q["id"]}.jev.json').write_text(json.dumps(judgment, indent=2))
                if not supported:
                    q['question'] = q['original_question']
                    q['wording_source'] = 'original_retained_after_jev_review'
            (args.output / f'{q["id"]}.response.json').write_text(json.dumps(result, indent=2))
        source['video_path'] = os.path.relpath(video.resolve(), args.output.resolve())
        (args.output / 'qa.json').write_text(json.dumps(source, indent=2))
    finally:
        reader.close()
        session.close()


if __name__ == '__main__':
    main()
