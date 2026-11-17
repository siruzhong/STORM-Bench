"""Check target selection and evidence preservation across parallel probes."""
import importlib.util
from pathlib import Path
import sys

scripts = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('parallel_probe', scripts / 'probe_targets_parallel.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_target_queue_interleaves_rooms_and_does_not_repeat_semantic_failures():
    rooms = [(0, dict(candidates=[dict(id=i) for i in range(1, 7)])),
             (1, dict(candidates=[dict(id=10), dict(id=11)]))]
    evidence = {0: dict(targets=[
        dict(id=1, operations=[dict(accepted=True)], infrastructure_failure='old timeout'),
        dict(id=2, operations=[dict(accepted=False)]),
        dict(id=3, operations=[], infrastructure_failure='timeout'),
    ])}
    assert module.pending_targets(rooms, evidence) == [(0, 4), (1, 10), (0, 5), (1, 11), (0, 6), (0, 3)]


def test_failed_parallel_trial_cannot_replace_accepted_operation():
    good = dict(id=5, operations=[dict(first='Open', second='Close', accepted=True, spawn=[1, 0, 2])])
    failed = dict(id=5, operations=[dict(first='Open', second='Close', accepted=False)])
    existing = dict(targets=[good], status='running')
    merged = module.merge_probe_entry(existing, dict(targets=[failed], status='blocked'))
    assert merged['targets'][0]['operations'][0]['accepted']
    assert merged['targets'][0]['operations'][0]['spawn'] == [1, 0, 2]
    assert existing['targets'] == [good]


def test_animation_queue_rechecks_logical_success_without_repeating_verified_targets():
    rooms=[(0,dict(candidates=[dict(id=1),dict(id=2),dict(id=3)]))]
    evidence={0:dict(targets=[
        dict(id=1,operations=[dict(accepted=True)]),
        dict(id=2,operations=[dict(accepted=True,animation_verified=True)]),
        dict(id=3,animation_attempted=True,operations=[dict(accepted=False)])])}
    assert module.pending_targets(rooms,evidence,require_animated=True)==[(0,1)]
