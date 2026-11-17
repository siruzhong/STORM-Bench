"""Check draft publication boundaries and immutable progress snapshots."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import index_raw_dataset as index


def capture(root):
    root.mkdir()
    rows = [dict(id=f'q{i}', event_index=i, options=['open', 'closed'],
                 answer_index=i % 2, query_time=60, evidence_spans=[[1, 2]],
                 question_type='current_state') for i in range(10)]
    values = {
        'config.json': dict(scene=0, room_id=11, room_name='bathroom', viewpoint='first_person'),
        'event_program.json': dict(plan_id='fixed'),
        'plan_execution.json': dict(accepted=True),
        'camera_continuity.json': dict(accepted=True),
        'observer_validation.json': dict(event_count=10, offscreen_event_count=5,
            duration_sec=65, camera_overlap_samples=0, frames=1300),
        'debug_validation.json': dict(frames=1300),
        'motion_validation.json': dict(acceptance_mode='locomotion_and_turning',
            active_frame_fraction=.9, longest_inactive_seconds=1.5),
        'qa.json': dict(qa_stage='raw', episode_id='room11', questions=rows),
        'events.json': [dict(event_index=i) for i in range(10)],
        'runtime_manifest.json': dict(native_navigation=True),
    }
    for name, value in values.items():
        (root / name).write_text(json.dumps(value))
    (root / 'qa_status.json').write_text(json.dumps(dict(status='raw_ready',
        events_sha256=hashlib.sha256((root / 'events.json').read_bytes()).hexdigest())))
    for name in ('video.mp4', 'debug.mp4'):
        (root / name).write_bytes(b'fixture')
    return root


def test_snapshot_is_idempotent_and_marks_prefix_and_review_requirements(tmp_path, monkeypatch):
    root = capture(tmp_path / 'capture')
    monkeypatch.setattr(index, 'summarize', lambda _: dict(accepted_episodes=[str(root)]))
    output = tmp_path / 'index'
    report = index.publish(tmp_path / 'ledger', output, 29, 'first_person')
    snapshot = (output / 'current').resolve()
    assert report['accepted_rooms'] == 1 and report['raw_questions'] == 10
    assert report['status'] == 'partial' and report['release_review'] == 'pending'
    assert report['uniform_random_accuracy'] == .5
    assert (snapshot / 'episodes/scene_0_room_11').resolve() == root
    rows = [json.loads(line) for line in (snapshot / 'raw_questions.jsonl').read_text().splitlines()]
    assert all(q['requires_query_prefix'] for q in rows)
    index.publish(tmp_path / 'ledger', output, 29, 'first_person')
    assert (output / 'current').resolve() == snapshot


def test_stale_answers_and_future_evidence_are_rejected(tmp_path):
    root = capture(tmp_path / 'capture')
    (root / 'events.json').write_text('[]')
    with pytest.raises(ValueError, match='different events'):
        index.inspect_capture(root, 'first_person')
    (root / 'qa_status.json').write_text(json.dumps(dict(status='raw_ready',
        events_sha256=index.digest(root / 'events.json'))))
    qa = index.read(root / 'qa.json')
    qa['questions'][0]['evidence_spans'] = [[1, 61]]
    (root / 'qa.json').write_text(json.dumps(qa))
    with pytest.raises(ValueError, match='beyond the query'):
        index.inspect_capture(root, 'first_person')


def test_other_viewpoint_is_not_counted(tmp_path):
    root = capture(tmp_path / 'capture')
    assert index.inspect_capture(root, 'third_person') is None


def test_watcher_keeps_updating_after_raw_coverage_until_root_checks_finish(monkeypatch):
    common = dict(status='raw_complete_pending_review', expected_rooms=29,
                  accepted_rooms=29, raw_questions=290)
    pending = dict(common, clearance_review_counts={
        'synchronized_root_verified': 24, 'pending_complete_root_evidence': 4,
        'flagged_for_review': 1})
    complete = dict(common, clearance_review_counts={'synchronized_root_verified': 29})
    reports = iter([pending, complete])
    published = []
    def publish(*args):
        value = next(reports)
        published.append(value)
        return value
    sleeps = []
    monkeypatch.setattr(index, 'publish', publish)
    monkeypatch.setattr(index.time, 'monotonic', lambda: 0)
    monkeypatch.setattr(index.time, 'sleep', sleeps.append)
    monkeypatch.setattr(sys, 'argv', ['index_raw_dataset.py', '--ledger', 'unused',
        '--output', 'unused', '--watch-seconds', '120'])
    index.main()
    assert published == [pending, complete]
    assert sleeps == [30]
