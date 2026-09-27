#!/usr/bin/env python3
"""Edit QA text using sampled video frames and a vision API.

Transport: POST {VLM_BASE_URL}/chat/completions, Bearer VLM_API_KEY.
Only question and video_evidence are editable.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
from importlib.resources import files
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.dataset_io import load_documents, read_json, validate_bundle, write_bundle, write_json, write_jsonl

SYSTEM = files('tools.storm').joinpath('prompts/qa_polish.txt').read_text(encoding='utf-8').strip()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def sample_times(document, question, maximum):
    """Sample evidence boundaries and fill gaps within the query prefix."""
    query = float(question['query_time'])
    candidates = sorted({float(t) for t in document['sample_timestamps']
                         if math.isfinite(float(t)) and 0 <= float(t) <= query
                         and float(t) < float(document['duration_sec'])})
    if not candidates:
        raise ValueError('no sample timestamps inside the query prefix')
    if maximum < 2:
        raise ValueError('max-frames must be at least 2')
    wanted = [candidates[-1]]
    for start, end in question['evidence_spans']:
        wanted.extend([float(start), float(end)])
    wanted.append(candidates[0])
    chosen = []
    for target in wanted:
        nearest = min(candidates, key=lambda t: abs(t - target))
        if nearest not in chosen:
            chosen.append(nearest)
        if len(chosen) == maximum:
            break
    # Fill the largest gaps between selected timestamps.
    while len(chosen) < min(maximum, len(candidates)):
        remaining = [t for t in candidates if t not in chosen]
        chosen.append(max(remaining, key=lambda t: min(abs(t - c) for c in chosen)))
    return sorted(chosen)


def extract_frames(video, timestamps, query_time, max_side=640):
    import imageio.v2 as imageio
    from PIL import Image
    reader = imageio.get_reader(str(video), format='ffmpeg')
    frames = []
    try:
        fps = float(reader.get_meta_data()['fps'])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('video has no valid constant frame rate')
        for timestamp in timestamps:
            if not 0 <= timestamp <= query_time:
                raise ValueError('attempt to sample beyond query_time')
            # Round down to stay within the query time in this constant-frame-rate video.
            index = int(math.floor(timestamp * fps))
            actual = index / fps
            if actual > query_time:
                raise ValueError('decoded frame would be beyond query_time')
            picture = Image.fromarray(reader.get_data(index)).convert('RGB')
            picture.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            picture.save(buffer, format='JPEG', quality=85)
            frames.append({'timestamp': actual, 'jpeg': buffer.getvalue()})
    finally:
        reader.close()
    return frames


def make_messages(question, frames):
    # Editing requests include labels; evaluation prompts exclude them.
    annotation = {key: question[key] for key in (
        'id', 'question', 'options', 'answer_index', 'query_time', 'video_evidence',
        'evidence_spans', 'question_type', 'diagnostics')}
    content = [{'type': 'text', 'text': 'Original immutable task and private annotation:\n' + json.dumps(annotation, ensure_ascii=False)}]
    for frame in frames:
        content.append({'type': 'text', 'text': f"Prefix frame at {frame['timestamp']:.3f} seconds"})
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/jpeg;base64,' + base64.b64encode(frame['jpeg']).decode('ascii')}})
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]


def call_vlm(*, base_url, api_key, model, messages, timeout, retries):
    """Send a Chat Completions request; adapt here for other API formats."""
    url = base_url.rstrip('/') + '/chat/completions'
    payload = json.dumps({'model': model, 'messages': messages, 'temperature': 0,
                          'max_tokens': 1200}).encode()
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=payload, method='POST', headers={
            'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read())
            text = value['choices'][0]['message']['content']
            if not isinstance(text, str):
                raise ValueError('API returned non-text message content')
            return text
        except urllib.error.HTTPError as exc:
            # Log the status code without response bodies or credentials.
            if attempt >= retries or (exc.code != 429 and exc.code < 500):
                raise RuntimeError(f'VLM API HTTP {exc.code}') from None
        except (urllib.error.URLError, TimeoutError):
            if attempt >= retries:
                raise RuntimeError('VLM API network error or timeout') from None
        time.sleep(min(2 ** attempt, 16))
    raise RuntimeError('VLM retry budget exhausted')


def validate_response(question, raw):
    stripped = raw.strip()
    if stripped.startswith('```') and stripped.endswith('```'):
        stripped = re.sub(r'^```(?:json)?\s*', '', stripped)[:-3].strip()
    result = json.loads(stripped)
    if not isinstance(result, dict) or set(result) != {'id', 'decision', 'question', 'video_evidence', 'reason'}:
        raise ValueError('response has missing or extra fields (factual fields are forbidden)')
    if result['id'] != question['id'] or result['decision'] not in ('rewrite', 'keep', 'flag'):
        raise ValueError('response ID or decision mismatch')
    for key in ('question', 'video_evidence', 'reason'):
        if not isinstance(result[key], str) or not result[key].strip() or len(result[key]) > 4000:
            raise ValueError(f'invalid response field: {key}')
    if result['decision'] == 'rewrite':
        # Reject timestamps and label cues in the question text.
        if re.search(r'\d|FloorPlan|answer_index|epistemic|ground.truth|correct (?:answer|option)', result['question'], re.I):
            raise ValueError('question contains timestamp, numeric or private-label cues')
    return result


def apply_response(question, result):
    changed = copy.deepcopy(question)
    if result['decision'] == 'rewrite':
        changed['question'] = result['question'].strip()
        changed['video_evidence'] = result['video_evidence'].strip()
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default=os.environ.get('VLM_BASE_URL', ''))
    parser.add_argument('--model', default=os.environ.get('VLM_MODEL', ''))
    parser.add_argument('--max-frames', type=int, default=12)
    parser.add_argument('--max-side', type=int, default=640)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--limit', type=int, help='Process only first N questions; retain others unchanged')
    parser.add_argument('--resume', action='store_true', help='Reuse validated per-question checkpoints')
    parser.add_argument('--retry-failed', action='store_true', help='With --resume, retry errors and flagged rows')
    parser.add_argument('--dry-run', action='store_true', help='Decode prefix frames and export request JSON; no API calls')
    args = parser.parse_args(argv)
    api_key = os.environ.get('VLM_API_KEY', '')
    if args.max_frames < 2 or args.max_side < 64 or args.timeout <= 0 or args.retries < 0 or (args.limit is not None and args.limit <= 0):
        parser.error('invalid frames, image size, timeout, retry count or limit')
    if args.retry_failed and not args.resume:
        parser.error('--retry-failed requires --resume')
    if not args.dry_run and not all((args.base_url, args.model, api_key)):
        parser.error('set VLM_BASE_URL, VLM_MODEL and VLM_API_KEY (or use --dry-run)')
    if args.base_url:
        parsed = urllib.parse.urlsplit(args.base_url)
        if parsed.scheme not in ('https', 'http') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            parser.error('base-url must be an HTTP(S) API base with no embedded credentials, query or fragment')
    source, output = args.input.expanduser().resolve(), args.output.expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('input and output must be separate, non-nested directories')
    if output.exists() and not args.resume:
        parser.error('output already exists; use a new directory or --resume')
    documents = load_documents(source)
    videos = {str(video): file_digest(video) for _, _, video in documents}
    settings = dict(base_url=args.base_url, model=args.model, max_frames=args.max_frames,
                    max_side=args.max_side, limit=args.limit, dry_run=args.dry_run, system=SYSTEM,
                    script_sha256=file_digest(__file__))
    signature = digest({'documents': [(str(rel), d) for rel, d, _ in documents], 'videos': videos, 'settings': settings})
    manifest = output / 'polish_run.json'
    if output.exists():
        if not manifest.is_file() or read_json(manifest).get('signature') != signature:
            parser.error('resume source, video, prompt or settings differ from this run; choose a new output')
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {'signature': signature, 'source': str(source), 'settings': settings, 'status': 'running'})
    audit = []
    processed = 0
    for relative, document, video in documents:
        for index, original in enumerate(document['questions']):
            if args.limit is not None and processed >= args.limit:
                audit.append({'id': original['id'], 'status': 'not_selected'})
                continue
            processed += 1
            fingerprint = digest({'question': original, 'video': videos[str(video)], 'signature': signature})
            checkpoint = output / '.polish_cache' / f'{fingerprint}.json'
            record = read_json(checkpoint) if checkpoint.exists() else None
            if record and args.retry_failed and record['status'] in ('error', 'flag'):
                record = None
            if record is None:
                record = {'id': original['id'], 'fingerprint': fingerprint}
                try:
                    times = sample_times(document, original, args.max_frames)
                    frames = extract_frames(video, times, float(original['query_time']), args.max_side)
                    record['frame_timestamps'] = [frame['timestamp'] for frame in frames]
                    messages = make_messages(original, frames)
                    if args.dry_run:
                        request_path = output / 'requests' / f'{fingerprint}.json'
                        write_json(request_path, {'model': args.model or 'YOUR_VISION_MODEL', 'messages': messages,
                                                  'temperature': 0, 'max_tokens': 1200})
                        record.update(status='dry_run', request=str(request_path.relative_to(output)))
                    else:
                        raw = call_vlm(base_url=args.base_url, api_key=api_key, model=args.model,
                                       messages=messages, timeout=args.timeout, retries=args.retries)
                        result = validate_response(original, raw)
                        record.update(status=result['decision'], response=result)
                except Exception as exc:
                    # Record the failure and leave the original question unchanged.
                    record.update(status='error', error=f'{type(exc).__name__}: {str(exc)[:240]}')
                write_json(checkpoint, record)
            if 'response' in record:
                result = validate_response(original, json.dumps(record['response']))
                document['questions'][index] = apply_response(original, result)
            audit.append(record)
            print(f"[{processed}] {original['id']}: {record['status']}", flush=True)
    if args.dry_run:
        write_jsonl(output / 'polish_audit.jsonl', audit)
    else:
        write_bundle(output, documents, parent=source, audit=audit)
        validate_bundle(output)
    errors = sum(row['status'] == 'error' for row in audit)
    write_json(manifest, {'signature': signature, 'source': str(source), 'settings': settings,
                         'status': 'completed_with_errors' if errors else 'completed',
                         'processed': processed, 'errors': errors,
                         'semantic_review_required': not args.dry_run})
    print(f'Output: {output}; processed={processed}; errors={errors}', flush=True)
    return 2 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
