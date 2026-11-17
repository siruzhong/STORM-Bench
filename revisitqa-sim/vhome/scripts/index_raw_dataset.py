#!/usr/bin/env python3
"""Publish immutable progress snapshots of accepted videos and their draft QA."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from recovery_status import summarize
from rollout_rooms_native import capture_accepted


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inspect_capture(root, viewpoint):
    config = read(root / 'config.json')
    if config.get('viewpoint', 'third_person') != viewpoint:
        return None
    program = read(root / 'event_program.json')
    if not capture_accepted(root, program, require_qa=True):
        raise ValueError(f'Capture or annotation gate failed: {root}')
    qa = read(root / 'qa.json')
    status = read(root / 'qa_status.json')
    if qa.get('qa_stage') != 'raw' or status.get('status') != 'raw_ready':
        raise ValueError(f'Expected draft annotations: {root}')
    if status['events_sha256'] != digest(root / 'events.json'):
        raise ValueError(f'Annotations refer to different events: {root}')
    observer = read(root / 'observer_validation.json')
    for question in qa['questions']:
        options = question['options']
        if not 2 <= len(options) <= 4 or len(set(options)) != len(options):
            raise ValueError(f'Invalid option domain: {question["id"]}')
        if not 0 <= question['answer_index'] < len(options):
            raise ValueError(f'Invalid answer: {question["id"]}')
        boundary = question['query_time']
        if not 0 < boundary <= observer['duration_sec']:
            raise ValueError(f'Invalid query boundary: {question["id"]}')
        if any(not 0 <= start < end <= boundary for start, end in question['evidence_spans']):
            raise ValueError(f'Evidence extends beyond the query: {question["id"]}')
    return dict(scene=config['scene'], room_id=config['room_id'],
                room=config['room_name'], capture=str(root.resolve()),
                episode_id=qa['episode_id'], viewpoint=viewpoint,
                duration_seconds=observer['duration_sec'], events=observer['event_count'],
                offscreen_events=observer['offscreen_event_count'],
                plan_id=program['plan_id'],
                hashes={name: digest(root / name) for name in (
                    'video.mp4', 'debug.mp4', 'qa.json', 'events.json',
                    'event_program.json', 'runtime_manifest.json')},
                questions=qa['questions'])


def publish(ledger, output, expected_rooms, viewpoint):
    report = summarize(ledger)
    entries = []
    for path in report['accepted_episodes']:
        entry = inspect_capture(Path(path), viewpoint)
        if entry is not None:
            entry['supplementary_clearance_review'] = (
                'flagged_for_review' if path in report.get('supplementary_clearance_flagged_episodes', [])
                else 'synchronized_root_verified' if path in report.get('synchronized_root_verified_episodes', [])
                else 'pending_complete_root_evidence')
            entries.append(entry)
    entries.sort(key=lambda item: (item['scene'], item['room_id']))
    keys = [(item['scene'], item['room_id']) for item in entries]
    if len(keys) != len(set(keys)) or len(keys) > expected_rooms:
        raise ValueError('Room coverage is inconsistent with the requested dataset')
    questions = [q for entry in entries for q in entry['questions']]
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Question identifiers are not unique across captures')
    signature = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()[:16]
    snapshots = output / 'snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    destination = snapshots / signature
    if not destination.exists():
        staging = snapshots / f'.{signature}.{os.getpid()}.staging'
        staging.mkdir()
        (staging / 'episodes').mkdir()
        rows = []
        for entry in entries:
            key = f"scene_{entry['scene']}_room_{entry['room_id']}"
            link = Path('episodes') / key
            (staging / link).symlink_to(entry['capture'], target_is_directory=True)
            for question in entry.pop('questions'):
                rows.append(dict(question, source_video_path=str(link / 'video.mp4'),
                                 requires_query_prefix=True))
            entry['episode_path'] = str(link)
        choices = Counter(len(q['options']) for q in rows)
        manifest = dict(schema_version=1, created_at=time.time(),
                        status=('raw_complete_pending_review' if len(entries) == expected_rooms
                                else 'partial'),
                        accepted_rooms=len(entries), expected_rooms=expected_rooms,
                        raw_questions=len(rows), viewpoint=viewpoint,
                        polishing='pending', evaluation='not_run',
                        release_review='pending',
                        clearance_review_counts=dict(Counter(entry['supplementary_clearance_review'] for entry in entries)),
                        option_counts=dict(choices),
                        uniform_random_accuracy=(sum(1 / len(q['options']) for q in rows) / len(rows)
                                                 if rows else None),
                        question_types=dict(Counter(q['question_type'] for q in rows)),
                        episodes=entries,
                        usage='Annotations contain answers. Export video prefixes through '
                              'export_model_inputs.py before evaluation. This is a progress '
                              'snapshot, not a reviewed benchmark release.')
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (staging / 'raw_questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in rows))
        staging.rename(destination)
    temporary = output / f'.current.{os.getpid()}'
    temporary.symlink_to(destination.resolve(), target_is_directory=True)
    temporary.replace(output / 'current')
    return read(destination / 'manifest.json')


def capture_evidence_complete(manifest):
    """Keep watching when raw room coverage still includes unverified captures."""
    expected = manifest['expected_rooms']
    reviews = manifest.get('clearance_review_counts', {})
    return (manifest['accepted_rooms'] == expected
            and manifest['raw_questions'] == expected * 10
            and reviews.get('synchronized_root_verified', 0) == expected
            and sum(reviews.values()) == expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-rooms', type=int, default=29)
    parser.add_argument('--viewpoint', choices=('first_person', 'third_person'), default='first_person')
    parser.add_argument('--watch-seconds', type=int, default=0)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.expected_rooms < 1 or args.watch_seconds < 0 or args.interval < 1:
        parser.error('Room count and interval must be positive; watch duration must be nonnegative')
    deadline = time.monotonic() + args.watch_seconds
    previous = None
    while True:
        result = publish(args.ledger, args.output, args.expected_rooms, args.viewpoint)
        summary = {key: result[key] for key in (
            'status', 'accepted_rooms', 'raw_questions', 'clearance_review_counts')}
        if summary != previous:
            print(json.dumps(summary), flush=True)
            previous = summary
        remaining = deadline - time.monotonic()
        if remaining <= 0 or capture_evidence_complete(result):
            break
        time.sleep(min(args.interval, remaining))


if __name__ == '__main__':
    main()
