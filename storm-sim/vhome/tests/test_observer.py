"""Check observer permissions and the knowledge boundary for hidden events."""
import pytest

from storm_virtualhome.observer_qa import build_observer_questions
from storm_virtualhome.observer_validation import check_roles
from storm_virtualhome.qa import TYPES
from storm_virtualhome.scene import action


def fixture_questions(container='fridge'):
    event = {'object': 'dishbowl', 'surface': 'kitchencounter', 'container': container}
    stages = {name: {'time': time} for name, time in
              [('before', 2), ('hidden_unobserved', 7), ('absence_discovered', 10), ('returned', 15)]}
    return build_observer_questions('episode', event, stages, 15)


def test_observer_cannot_manipulate():
    check_roles([{'action': '<char0> [TurnTo] <fridge> (305)'},
                 {'action': '<char0> [TurnLeft]'},
                 {'action': '<char0> [TurnRight]'},
                 {'action': '<char1> [Grab] <dishbowl> (326)'}])
    with pytest.raises(ValueError, match='observer'):
        check_roles([{'action': '<char0> [Grab] <dishbowl> (326)'}])
    assert action('Grab', {'id': 326, 'class_name': 'dishbowl'}, character=1).startswith('<char1>')


def test_hidden_truth_does_not_change_observer_answers():
    assert fixture_questions('fridge') == fixture_questions('cupboard')
    questions = fixture_questions()
    assert {q['question_type'] for q in questions} == set(TYPES)
    for q in questions:
        assert all(end <= q['query_time'] for _, end in q['evidence_spans'])
    hidden = next(q for q in questions if q['question_subtype'] == 'offscreen_location')
    assert hidden['diagnostics']['epistemic_status'] == 'uncertain'
    assert 'cannot be determined' in hidden['options'][hidden['answer_index']]
    assert max(end for _, end in hidden['evidence_spans']) == 7


def test_reject_discovery_before_hidden_checkpoint():
    stages = {name: {'time': time} for name, time in
              [('before', 2), ('hidden_unobserved', 7), ('absence_discovered', 6), ('returned', 15)]}
    with pytest.raises(ValueError, match='ordered'):
        build_observer_questions('episode', {'object': 'dishbowl', 'surface': 'kitchencounter'}, stages, 15)


def test_reject_even_one_visible_frame_during_hidden_injection(tmp_path, monkeypatch):
    import json
    import numpy as np
    from PIL import Image
    from storm_virtualhome import observer_validation as validation

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    (tmp_path / 'evidence').mkdir()
    write('config.json', {'camera': {'recording_camera': '87'}, 'render': {'fps': 1},
                         'evidence': {'min_visible_pixels': 1}, 'monitor_ids': [2, 10, 20]})
    write('events.json', [{'object_id': 3, 'surface_id': 10, 'container_id': 20,
                          'disappear': {'start': 1, 'end': 2, 'discovery_confirmed_at': 3}}])
    write('actions.json', [{'action': '<char1> [Grab] <bowl> (3)'}])
    write('stages.json', {'absence_discovered': {'time': 3}})
    colors = {'3': [1, 0, 0], '10': [0, 1, 0], '20': [0, 0, 1], '2': [1, 1, 0]}
    for phase in ('before', 'hidden_unobserved', 'absence_discovered', 'returned'):
        relation = {'from_id': 3, 'to_id': 20 if phase == 'hidden_unobserved' else 10,
                    'relation_type': 'INSIDE' if phase == 'hidden_unobserved' else 'ON'}
        write(f'evidence/{phase}.json', {'color': colors['3'], 'color_collision_ids': [],
              'instance_colors': colors, 'graph': {'nodes': [{'id': 20, 'states': ['CLOSED'], 'obj_transform': {'position': [1, 0, 0]}},
                                                  {'id': 2, 'obj_transform': {'position': [0, 0, 0]}},
                                                  {'id': 3, 'obj_transform': {'position': [1, 0, 1]}}],
                                                  'edges': [relation]}})
    red = np.full((10, 10, 3), [255, 0, 0], dtype=np.uint8)
    green = np.full((10, 10, 3), [0, 255, 0], dtype=np.uint8)
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    Image.fromarray(red).save(tmp_path / 'evidence/before.seg.png')
    Image.fromarray(red).save(tmp_path / 'evidence/returned.seg.png')
    Image.fromarray(black).save(tmp_path / 'evidence/hidden_unobserved.seg.png')
    Image.fromarray(green).save(tmp_path / 'evidence/absence_discovered.seg.png')
    rows = []
    visible_pair = red.copy()
    visible_pair[:5] = [255, 255, 0]
    for frame, array in enumerate((red, black, green, visible_pair, visible_pair, visible_pair)):
        name = f'frame{frame}.png'
        Image.fromarray(array).save(tmp_path / name)
        rows.append({'frame': frame, 'mask': name, 'color': colors['3'], 'color_collision_ids': [],
                     'monitored_colors': {key: colors[key] for key in ('2', '10', '20')}})
    (tmp_path / 'frame_manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    monkeypatch.setattr(validation, 'camera_audit', lambda root: {'camera_overlap_samples': 0})
    assert validation.audit_observer(tmp_path)['offscreen_injection_frames'] == 1
    leaked = black.copy()
    leaked[0, 0] = [255, 0, 0]
    Image.fromarray(leaked).save(tmp_path / 'frame1.png')
    with pytest.raises(ValueError, match='visible target'):
        validation.audit_observer(tmp_path)
    Image.fromarray(black).save(tmp_path / 'frame1.png')
    before = json.loads((tmp_path / 'evidence/before.json').read_text())
    next(n for n in before['graph']['nodes'] if n['id'] == 2)['obj_transform']['position'] = [100, 0, 100]
    write('evidence/before.json', before)
    with pytest.raises(ValueError, match='did not reach'):
        validation.audit_observer(tmp_path)


def test_concurrent_actions_preserve_observer_role():
    check_roles([{'action': '<char0> [Walk] <desk> (373) | <char1> [Grab] <bowl> (326)'}])
    with pytest.raises(ValueError, match='observer'):
        check_roles([{'action': '<char1> [Walk] <desk> (373) | <char0> [Grab] <bowl> (326)'}])
    with pytest.raises(ValueError, match='Unexpected'):
        check_roles([{'action': '<char0> [Walk] <desk> (373) | <char0> [TurnTo] <bowl> (326)'}])


def test_motion_audit_distinguishes_walks_and_stops():
    import numpy as np
    from storm_virtualhome.motion import movement_metrics
    walk = np.zeros((100, 3))
    walk[:, 0] = np.arange(100) / 20
    moving, _ = movement_metrics(walk, 20)
    assert moving['moving_frame_fraction'] == 1.0
    assert moving['longest_stationary_seconds'] == 0
    stopped, _ = movement_metrics(np.zeros((100, 3)), 20)
    assert stopped['moving_frame_fraction'] == 0
    assert stopped['longest_stationary_seconds'] == 5


def test_viewpoint_registration_tracks_the_original_region():
    import cv2
    import numpy as np
    from storm_virtualhome.registration import align_region
    rng = np.random.default_rng(7)
    before = rng.integers(0, 256, (320, 400, 3), dtype=np.uint8)
    transform = np.array([[1, 0, 12], [0, 1, -8]], dtype=np.float64)
    after = cv2.warpAffine(before, transform, (400, 320))
    region = np.zeros((320, 400), dtype=bool)
    region[150:165, 190:205] = True
    empty = np.zeros_like(region)
    aligned, report = align_region(before, after, region, empty, empty)
    expected = cv2.warpAffine(region.astype(np.uint8), transform, (400, 320)) > 0
    assert (aligned & expected).sum() / expected.sum() > 0.95
    assert report['inliers'] >= 12
