"""Check balanced QA selection and restart safety without running the simulator."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('batch', Path(__file__).parents[1] / 'scripts/rollout_batch.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def pool():
    return [dict(id=f'{i}_{kind}', event_index=i, question_type=kind,
                 diagnostics={'epistemic_status': 'uncertain' if i >= 5 and kind == 'current_state' else 'known'})
            for i in range(10) for kind in batch.TYPES]


def test_30_episodes_cover_events_balance_types_and_keep_uncertainty():
    counts = Counter()
    for slot in range(30):
        selected = batch.select_questions(pool(), slot)
        assert len(selected) == 10
        assert {q['event_index'] for q in selected} == set(range(10))
        assert any(q['diagnostics']['epistemic_status'] == 'uncertain' for q in selected)
        counts.update(q['question_type'] for q in selected)
    assert counts == dict.fromkeys(batch.TYPES, 50)


def test_selection_rejects_missing_types():
    with pytest.raises(ValueError):
        batch.select_questions(pool()[:-1], 0)


def test_resume_rejects_frozen_file_edits(tmp_path):
    source = tmp_path / 'input.json'
    source.write_text('{}')
    batch.save(tmp_path / 'status.json', {'frozen_files': {'input.json': batch.sha(source)}})
    args = SimpleNamespace(output=tmp_path)
    batch.initialize(args)
    source.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='modified'):
        batch.initialize(args)


def test_missing_capture_reports_are_not_accepted(tmp_path):
    assert batch.accepted(tmp_path, {'plan_id': 'x'}) is False


def test_resume_does_not_rerender_failed_attempts(tmp_path):
    folder = tmp_path / 'inputs'
    folder.mkdir()
    batch.save(folder / 'program.json', {'seed': 1})
    state = {'contract': {'videos': 3, 'max_attempts': 1}, 'accepted': [],
             'candidates': [{'seed': 1, 'input': 'inputs', 'accepted': False,
                             'attempts': [{'name': 'seed_000001_attempt_01', 'status': 'failed'}]}]}
    batch.run(tmp_path, state)
    assert state['status'] == 'exhausted'
    assert state['qa_count'] == 0
    assert not (tmp_path / 'attempts').exists()


def test_resume_publishes_captured_episodes_once_in_order(tmp_path, monkeypatch):
    candidates = []
    for seed in (1, 2):
        folder = tmp_path / f'input_{seed}'
        folder.mkdir()
        batch.save(folder / 'program.json', {'seed': seed})
        candidates.append({'seed': seed, 'input': folder.name, 'accepted': False,
                           'attempts': [{'name': f'seed_{seed:06d}_attempt_01', 'status': 'captured'}]})
    state = {'contract': {'videos': 2, 'max_attempts': 1}, 'accepted': [], 'candidates': candidates}
    calls = []

    def publish(root, episode, slot, program, source_dir):
        calls.append((slot, program['seed']))
        return {'slot': slot, 'seed': program['seed']}

    monkeypatch.setattr(batch, 'publish', publish)
    monkeypatch.setattr(batch, 'aggregate', lambda root, state: None)
    monkeypatch.setattr(batch.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Captured work must not be rerendered'))
    batch.run(tmp_path, state)
    assert calls == [(0, 1), (1, 2)]
    assert len(state['accepted']) == 2
    assert all(c['accepted'] for c in candidates)


def test_publication_overlaps_the_next_capture(tmp_path, monkeypatch):
    import threading
    publishing = threading.Event()
    capturing = threading.Event()
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    guard = runtime / 'camera_guard.json'
    guard.write_text('{}')
    candidates = []
    for seed in (1, 2):
        folder = tmp_path / f'input_{seed}'
        folder.mkdir()
        batch.save(folder / 'program.json', {'seed': seed})
        candidates.append({'seed': seed, 'input': folder.name, 'accepted': False,
                           'attempts': [{'name': 'seed_000001_attempt_01', 'status': 'captured'}] if seed == 1 else []})
    state = {'contract': {'videos': 2, 'max_attempts': 1, 'executable': str(runtime / 'simulator'),
                         'runtime_guard_sha256': batch.sha(guard), 'min_free_gb': 0,
                         'gpu_index': 0, 'xorg_root': str(runtime), 'attempt_timeout_seconds': 10},
             'accepted': [], 'candidates': candidates}

    def publish(root, episode, slot, program, source_dir):
        if slot == 0:
            publishing.set()
            assert capturing.wait(2), 'Next capture was blocked by publication'
        return {'slot': slot, 'seed': program['seed']}

    class Child:
        def wait(self, timeout):
            assert publishing.wait(2)
            capturing.set()
            return 0

    monkeypatch.setattr(batch, 'publish', publish)
    monkeypatch.setattr(batch, 'aggregate', lambda root, state: None)
    monkeypatch.setattr(batch, 'accepted', lambda *args: True)
    monkeypatch.setattr(batch.subprocess, 'Popen', lambda *a, **kw: Child())
    batch.run(tmp_path, state)
    assert len(state['accepted']) == 2
    assert capturing.is_set()
