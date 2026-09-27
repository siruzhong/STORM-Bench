"""Reject missing transitions and irregular injection spacing."""
import pytest

from storm_virtualhome.schedule import transition_schedule
from storm_virtualhome.observer_qa import build_observer_questions


def pairs():
    return [{'object_id': 326,
             'disappear': {'start': 20 + i * 56, 'end': 40 + i * 56},
             'appear': {'start': 48 + i * 56, 'end': 67 + i * 56}}
            for i in range(5)]


def config():
    return {'event_count': 10, 'event_interval_seconds': 28,
            'event_interval_tolerance_seconds': 6}


def test_ten_transitions_are_five_pairs():
    report = transition_schedule(pairs(), config())
    assert report['event_count'] == 10
    assert report['start_gaps_seconds'] == [28] * 9
    with pytest.raises(ValueError, match='Expected 10'):
        transition_schedule(pairs()[:-1], config())
    assert transition_schedule(pairs()[:2], config(), complete=False)['event_count'] == 4


def test_irregular_spacing_and_overlap_are_rejected():
    events = pairs()
    events[-1]['appear'] = {'start': 300, 'end': 319}
    with pytest.raises(ValueError, match='outside'):
        transition_schedule(events, config())
    events = pairs()
    events[1]['disappear']['start'] = 60
    with pytest.raises(ValueError, match='overlap'):
        transition_schedule(events, config())


def test_history_counts_accumulate_without_future_evidence():
    all_questions = []
    history = []
    for i in range(5):
        stages = {name: {'time': t + i * 56} for name, t in
                  [('before', 12), ('hidden_unobserved', 40), ('absence_discovered', 48), ('returned', 67)]}
        history.extend(stages[name]['time'] for name in ('before', 'absence_discovered', 'returned'))
        questions = build_observer_questions('episode', {'object': 'dishbowl', 'surface': 'kitchencounter'},
                    stages, 300, cycle_index=i, history_times=history)
        count = next(q for q in questions if q['question_type'] == 'history_aggregation')
        assert count['options'][count['answer_index']] == str(i + 1)
        assert len(count['evidence_spans']) == 3 * (i + 1)
        assert all(end <= q['query_time'] for q in questions for _, end in q['evidence_spans'])
        all_questions.extend(questions)
    assert len({q['id'] for q in all_questions}) == 40


def test_later_cycle_audits_only_its_own_frames(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    from storm_virtualhome.observer_validation import audit_pair

    def write(name, value):
        (tmp_path / name).write_text(json.dumps(value))

    (tmp_path / 'evidence').mkdir()
    write('config.json', {'camera': {'recording_camera': '87'}, 'render': {'fps': 1},
                         'evidence': {'min_visible_pixels': 1}, 'monitor_ids': [2, 10, 20]})
    write('actions.json', [{'action': '<char0> [Walk] <chair> (4) | <char1> [Grab] <bowl> (3)'}])
    phases = ('before', 'hidden_unobserved', 'absence_discovered', 'returned')
    keys = {name: 'cycle_01_' + name for name in phases}
    write('stages.json', {keys[name]: {'time': t} for name, t in zip(phases, (6.5, 8, 9, 12))})
    colors = {'3': [1, 0, 0], '10': [0, 1, 0], '20': [0, 0, 1], '2': [1, 1, 0]}
    for phase in phases:
        hidden = phase == 'hidden_unobserved'
        graph = {'nodes': [{'id': 20, 'states': ['CLOSED'], 'obj_transform': {'position': [1, 0, 0]}},
                           {'id': 2, 'obj_transform': {'position': [0, 0, 0]}},
                           {'id': 3, 'obj_transform': {'position': [1, 0, 1]}}],
                 'edges': [{'from_id': 3, 'to_id': 20 if hidden else 10,
                            'relation_type': 'INSIDE' if hidden else 'ON'}]}
        write(f'evidence/{keys[phase]}.json', {'color': colors['3'], 'color_collision_ids': [],
                                             'instance_colors': colors, 'graph': graph})
        color = [255, 0, 0] if phase in ('before', 'returned') else [0, 255, 0]
        Image.fromarray(np.full((10, 10, 3), color, dtype=np.uint8)).save(tmp_path / f'evidence/{keys[phase]}.seg.png')
    event = {'object_id': 3, 'surface_id': 10, 'container_id': 20, 'stage_keys': keys,
             'disappear': {'start': 7, 'end': 8, 'discovery_confirmed_at': 9}}
    rows = [{'frame': t, 'time': t, 'target_pixels': 100 if t >= 9 or t < 6 else 0,
             'monitored_pixels': {'2': 100 if t >= 9 else 0, '10': 0, '20': 0}}
            for t in range(18)]
    report = audit_pair(tmp_path, event, rows, lower_bound=6)
    assert report['frames'] == 6
    assert report['first_reobserved_target_time'] == 9
    assert report['visible_return_injector_frames'] == 3
    rows[7]['target_pixels'] = 1
    with pytest.raises(ValueError, match='visible target'):
        audit_pair(tmp_path, event, rows, lower_bound=6)


def test_short_video_duration_is_a_hard_bound():
    from storm_virtualhome.schedule import check_duration
    config = {'duration_range_seconds': [60, 70]}
    for duration in (60, 65, 70):
        check_duration(duration, config)
    for duration in (59.95, 70.05, float('nan')):
        with pytest.raises(ValueError, match='outside'):
            check_duration(duration, config)
    with pytest.raises(ValueError, match='Invalid'):
        check_duration(65, {'duration_range_seconds': [70, 60]})


def test_hand_held_questions_use_the_visible_holder():
    stages = {name: {'time': t} for name, t in
              [('before', 1), ('hidden_unobserved', 4), ('absence_discovered', 6), ('returned', 10)]}
    event = {'object': 'dishbowl', 'surface': 'kitchencounter', 'held_by_injector': True}
    questions = build_observer_questions('short', event, stages, 10)
    current = next(q for q in questions if q['question_subtype'] == 'visible_location')
    assert current['options'][current['answer_index']] == "in the other person's hand"
    tracking = next(q for q in questions if q['question_type'] == 'object_tracking')
    assert tracking['options'][tracking['answer_index']] == 'The other person.'
    hidden = next(q for q in questions if q['question_subtype'] == 'offscreen_location')
    assert hidden['diagnostics']['epistemic_status'] == 'uncertain'


def test_holding_requires_an_actor_to_object_relation():
    from storm_virtualhome.scene import held_by
    assert held_by({'edges': [{'from_id': 2, 'to_id': 326, 'relation_type': 'HOLDS_RH'}]}, 326)
    assert not held_by({'edges': [{'from_id': 326, 'to_id': 2, 'relation_type': 'ON'}]}, 326)


def test_turning_is_reported_separately_from_translation():
    import numpy as np
    from storm_virtualhome.motion import active_metrics
    positions = np.zeros((100, 3))
    turn = np.arange(100) / 20
    report, _, _ = active_metrics(positions, turn, 20)
    assert report['moving_frame_fraction'] == 0
    assert report['active_frame_fraction'] == 1
    assert report['longest_inactive_seconds'] == 0
    idle, _, _ = active_metrics(positions, np.sin(np.arange(100)) * 0.002, 20)
    assert idle['active_frame_fraction'] == 0
    assert idle['longest_inactive_seconds'] == 5
