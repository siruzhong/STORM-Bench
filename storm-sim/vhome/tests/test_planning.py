"""Verify frozen random plans and reject replay drift before dataset export."""
import copy

import pytest

from storm_virtualhome.planning import create_program, validate_program, check_execution, digest
from test_mixed import scene, config


def settings(seed=42):
    return dict(config(seed), randomize_event_order=True)


def test_random_order_is_reproducible_and_not_a_fixed_chain():
    orders = set()
    for seed in range(100):
        p = create_program(scene(), settings(seed))
        assert p == create_program(scene(), settings(seed))
        validate_program(p, scene(), settings(seed))
        orders.add(tuple(e['task'] for e in p['events']))
        assert all(e['injector_character'] != 0 for e in p['events'])
        times = [e['planned_start_seconds'] for e in p['events']]
        assert all(abs(t * 20 - round(t * 20)) < 1e-9 for t in times)
        assert all(5.4 - 1e-9 <= b - a <= 6.6 + 1e-9 for a, b in zip(times, times[1:]))
    assert len(orders) >= 12


def test_scene_config_and_plan_drift_are_rejected():
    p = create_program(scene(), settings())
    changed = copy.deepcopy(scene())
    changed['nodes'][4]['states'] = ['OPEN']
    with pytest.raises(ValueError, match='scene'):
        validate_program(p, changed, settings())
    with pytest.raises(ValueError, match='configuration'):
        validate_program(p, scene(), settings(17))
    p['events'][0]['injector_character'] = 0
    with pytest.raises(ValueError, match='modified'):
        validate_program(p, scene(), settings())
    p['plan_id'] = digest({k: v for k, v in p.items() if k != 'plan_id'})
    with pytest.raises(ValueError, match='operating characters'):
        validate_program(p, scene(), settings())


def test_rendering_must_follow_objects_actors_and_start_windows():
    p = create_program(scene(), settings())
    actual = copy.deepcopy(p['events'])
    for e in actual:
        e['start'] = e['planned_start_seconds']
    check_execution(p, actual)
    actual[3]['object_id'] = 999
    with pytest.raises(ValueError, match='changed'):
        check_execution(p, actual)
    actual = copy.deepcopy(p['events'])
    for e in actual:
        e['start'] = e['planned_start_seconds'] + 3
    with pytest.raises(ValueError, match='window'):
        check_execution(p, actual)


def test_failed_execution_leaves_a_rejection_report(tmp_path):
    import json
    from storm_virtualhome.planning import audit_execution
    program = create_program(scene(), settings())
    (tmp_path / 'event_program.json').write_text(json.dumps(program))
    (tmp_path / 'events.json').write_text('[]')
    with pytest.raises(ValueError, match='complete'):
        audit_execution(tmp_path)
    report = json.loads((tmp_path / 'plan_execution.json').read_text())
    assert report['accepted'] is False
    assert report['actual_count'] == 0
    assert report['plan_id'] == program['plan_id']


def test_event_proportions_are_sampled_before_rendering():
    from collections import Counter
    proportions = set()
    for seed in range(100):
        options = dict(settings(seed), randomize_event_mix=True,
                       object_descriptions={235: "cabinet above the sink", 236: "cabinet nearest the refrigerator"})
        program = create_program(scene(), options)
        validate_program(program, scene(), options)
        events = program['events']
        counts = Counter(e['verb'] for e in events)
        proportions.add((counts['Open'], counts['Grab']))
        assert len(events) == 10
        assert len({e['object_id'] for e in events}) == 5
        assert counts['Open'] == counts['Close']
        assert counts['Grab'] == counts['PutBack']
        assert sum(e['requested_visibility'] == 'offscreen' for e in events) == 2
        assert events[0]['verb'] == 'SwitchOn'
    assert proportions == {(3, 1), (2, 2)}


def test_same_class_targets_need_visible_references():
    with pytest.raises(ValueError, match='visible object descriptions'):
        create_program(scene(), dict(settings(42), randomize_event_mix=True))


def test_absolute_cadence_returns_to_the_frozen_schedule_after_delay():
    from storm_virtualhome.planning import event_window_deadline, preferred_start
    event = {'planned_start_seconds': 42.7, 'start_tolerance_seconds': 2.5}
    assert preferred_start(event, 34.8, 6) == pytest.approx(42.7)
    assert preferred_start(event, 30, 6) == pytest.approx(42.7)
    assert preferred_start(event, 38, 6) == pytest.approx(42.7)
    assert preferred_start(event, 43, 6) == pytest.approx(45.2)
    assert event_window_deadline(event, 40.2) == pytest.approx(45.2)
    assert event_window_deadline(event, 40.2, 'frozen_relative_intervals') == pytest.approx(42.7)
    assert event == {'planned_start_seconds': 42.7, 'start_tolerance_seconds': 2.5}


def test_saved_json_configuration_preserves_plan_signature():
    import json
    from storm_virtualhome.planning import planning_config
    config = dict(settings(), object_descriptions={235: "left cabinet", 236: "right cabinet"})
    assert digest(planning_config(config)) == digest(planning_config(json.loads(json.dumps(config))))


def test_missing_source_evidence_writes_a_rejection_report(tmp_path):
    import json
    from storm_virtualhome.planning import audit_execution
    program = create_program(scene(), settings())
    (tmp_path / 'event_program.json').write_text(json.dumps(program))
    (tmp_path / 'events.json').write_text('[]')
    (tmp_path / 'config.json').write_text(json.dumps(settings()))
    with pytest.raises(FileNotFoundError):
        audit_execution(tmp_path)
    assert json.loads((tmp_path / 'plan_execution.json').read_text())['accepted'] is False


def test_five_offscreen_events_are_frozen_for_both_random_mixtures():
    counts = set()
    for seed in range(100):
        options = dict(settings(seed), randomize_event_mix=True, offscreen_event_count=5,
                       object_descriptions={235: "stove-side cabinet", 236: "refrigerator-side cabinet"})
        program = create_program(scene(), options)
        assert program == create_program(scene(), options)
        hidden = [e for e in program['events'] if e['requested_visibility'] == 'offscreen']
        assert len(hidden) == 5
        assert all(e['injector_character'] == 1 for e in hidden)
        assert {'Grab', 'PutBack', 'Close'} <= {e['verb'] for e in hidden}
        counts.add(sum(e['verb'] == 'Grab' for e in program['events']))
        changed = copy.deepcopy(program)
        hidden_index = next(i for i, e in enumerate(changed['events']) if e['requested_visibility'] == 'offscreen')
        changed['events'][hidden_index]['requested_visibility'] = 'in_view'
        changed['plan_id'] = digest({k: v for k, v in changed.items() if k != 'plan_id'})
        with pytest.raises(ValueError, match='offscreen event count'):
            validate_program(changed, scene(), options)
    assert counts == {1, 2}


def test_relative_cadence_uses_each_planned_visibility_interval():
    from storm_virtualhome.mixed import check_event_spacing

    events = [{"start": value} for value in (1.3, 6.9, 10.95, 19.2)]
    program = {
        "events": [
            {},
            {"planned_gap_seconds": 7.8, "start_tolerance_seconds": 3.0},
            {"planned_gap_seconds": 6.5, "start_tolerance_seconds": 3.0},
            {"planned_gap_seconds": 7.8, "start_tolerance_seconds": 3.0},
        ]
    }
    options = {
        "timing_policy": "frozen_relative_intervals",
        "event_interval_seconds": 6.0,
        "event_interval_tolerance_seconds": 3.0,
    }
    assert check_event_spacing(events, options, program) == pytest.approx(
        [5.6, 4.05, 8.25]
    )
    program["events"][2]["planned_gap_seconds"] = 8.0
    with pytest.raises(ValueError, match="Irregular event spacing"):
        check_event_spacing(events, options, program)


def test_measured_offscreen_count_must_stay_near_the_target():
    from storm_virtualhome.mixed import check_offscreen_count
    for count in (5, 6):
        check_offscreen_count(count, {'offscreen_event_count': 5})
    for count in (2, 4, 7, 10):
        with pytest.raises(ValueError):
            check_offscreen_count(count, {'offscreen_event_count': 5})


def test_hidden_transfer_reserves_time_before_rendering():
    options = dict(settings(42), randomize_event_mix=True, offscreen_event_count=5,
                   object_descriptions={235: "stove-side cabinet", 236: "refrigerator-side cabinet"})
    regular = create_program(scene(), options)
    reserved = create_program(scene(), dict(options, offscreen_transfer_interval_seconds=7.0))
    placement = next(i for i, e in enumerate(regular['events']) if e['verb'] == 'PutBack')
    for index, (before, after) in enumerate(zip(regular['events'], reserved['events'])):
        assert before['task'] == after['task']
        assert after['planned_start_seconds'] - before['planned_start_seconds'] == pytest.approx(1 if index >= placement else 0)
        assert after['start_tolerance_seconds'] == before['start_tolerance_seconds']


def test_transition_cadence_is_weighted_without_extending_the_episode():
    options = dict(
        settings(42),
        visibility_adaptive_cadence=True,
        in_view_event_interval_seconds=6.0,
        offscreen_event_interval_seconds=6.0,
        offscreen_transfer_interval_seconds=6.0,
        object_transition_interval_extra_seconds=2.0,
        event_span_seconds=54.0,
    )
    program = create_program(scene(), options)
    gaps = [event['planned_gap_seconds'] for event in program['events'][1:]]
    transitions = [
        gap for previous, event, gap in zip(program['events'], program['events'][1:], gaps)
        if previous['object_id'] != event['object_id']
    ]
    repeated = [
        gap for previous, event, gap in zip(program['events'], program['events'][1:], gaps)
        if previous['object_id'] == event['object_id']
    ]
    assert sum(gaps) == pytest.approx(54.0, abs=.05)
    assert transitions and repeated
    assert min(transitions) > max(repeated)


def test_event_span_must_be_positive_and_finite():
    with pytest.raises(ValueError, match='event span'):
        create_program(scene(), dict(settings(), event_span_seconds=0))


def test_online_cadence_gate_matches_final_spacing_limits():
    from storm_virtualhome.planning import check_injection_start
    event = {'event_index': 5, 'planned_start_seconds': 31.8, 'start_tolerance_seconds': 2.5}
    options = {'event_interval_seconds': 6.0, 'event_interval_tolerance_seconds': 2.5}
    check_injection_start(event, 32.5, 24.0, options)
    with pytest.raises(ValueError, match='invalid start gap'):
        check_injection_start(event, 32.55, 24.0, options)
    with pytest.raises(ValueError, match='frozen window'):
        check_injection_start(event, 34.35, 28.0, options)
    check_injection_start(event, 31.8, None, options)


def test_relative_timing_freezes_intervals_without_accumulated_clock_rejection():
    from storm_virtualhome.planning import check_injection_start
    options=dict(settings(),timing_policy='frozen_relative_intervals')
    program=create_program(scene(),options)
    actual=copy.deepcopy(program['events'])
    actual[0]['start']=actual[0]['planned_start_seconds']
    for previous,event in zip(actual,actual[1:]):
        event['start']=previous['start']+event['planned_gap_seconds']+1.0
    assert actual[-1]['start']-program['events'][-1]['planned_start_seconds']>2.5
    check_execution(program,actual)
    check_injection_start(program['events'][1],actual[1]['start'],actual[0]['start'],options)
    actual[4]['start']+=3.0
    with pytest.raises(ValueError,match='relative interval'):
        check_execution(program,actual)
