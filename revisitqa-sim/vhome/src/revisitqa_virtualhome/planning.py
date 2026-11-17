"""Freeze a seeded event program before simulator action rendering starts."""
import copy
import hashlib
import json
import math
import random

from .mixed import STATE_EFFECTS, plan_routine
from .scene import support_of


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def planning_config(config):
    value = copy.deepcopy(config)
    value.get('camera', {}).pop('recording_camera', None)
    return json.loads(json.dumps(value))


def scene_signature(graph, config=None):
    nodes = sorted(({k: n[k] for k in ('id', 'class_name', 'properties', 'states')} for n in graph['nodes'] if n['class_name'] != 'character'), key=lambda n: n['id'])
    ids = {n['id'] for n in nodes}
    edges = sorted((e for e in graph['edges'] if e['from_id'] in ids and e['to_id'] in ids and e['relation_type'] in ('ON', 'INSIDE')), key=lambda e: (e['from_id'], e['to_id'], e['relation_type']))
    if config and config.get('scene_signature_mode') == 'native_room':
        relevant = {a['id'] for a in config['room_action_pool']} | {a['surface_id'] for a in config['room_action_pool']}
        edges = [e for e in edges if e['relation_type'] == 'INSIDE' or e['from_id'] in relevant]
    if config and config.get('scene_signature_mode') == 'room_recipe':
        # Native startup can recompute support edges for unrelated decorations.
        # Keep all object identities and states, containment, and every support
        # relation used by manipulation, placement or navigation.
        pool = config['mixed_candidates']
        relevant = set(pool['portable'] + pool['cabinets'])
        relevant.update((pool['appliance'], pool['switch'], pool['secondary_surface_id'], config['surface_id']))
        relevant.update(config['navigation'].values())
        edges = [e for e in edges if e['relation_type'] == 'INSIDE' or e['from_id'] in relevant]
    return digest({'nodes': nodes, 'edges': edges})


def create_program(graph, config):
    if config.get('room_action_pool'):
        from .room_program import plan_room_routine
        events = plan_room_routine(graph, config)
    else:
        events = plan_routine(graph, config)
    descriptions = config.get('object_descriptions', {})
    objects_by_class = {}
    for event in events:
        objects_by_class.setdefault(event['object'], set()).add(event['object_id'])
        description = descriptions.get(event['object_id'], descriptions.get(str(event['object_id'])))
        if description:
            event['object_description'] = description
    for keys in objects_by_class.values():
        if len(keys) > 1:
            names = {descriptions.get(key, descriptions.get(str(key))) for key in keys}
            if None in names or len(names) != len(keys):
                raise ValueError('Same-class targets need distinct visible object descriptions')
    rng = random.Random(config['seed'] ^ 0xE73A11)
    interval = config.get('event_interval_seconds', 6.0)
    tolerance = config.get('event_interval_tolerance_seconds', 2.5)
    fps = config.get('render', {}).get('fps', 20)
    nominal = config.get("first_event_seconds", 1.0)
    planned_gaps = []
    for index, event in enumerate(events[1:], 1):
        previous = events[index - 1]
        transfer = (previous['verb'] == 'Grab' and event['verb'] == 'PutBack'
                    and previous['object_id'] == event['object_id']
                    and 'offscreen' in (previous['requested_visibility'], event['requested_visibility']))
        if transfer:
            gap = config.get('offscreen_transfer_interval_seconds', interval)
        elif config.get('visibility_adaptive_cadence'):
            gap = (config.get('offscreen_event_interval_seconds', interval)
                   if event['requested_visibility'] == 'offscreen'
                   else config.get('in_view_event_interval_seconds', interval))
        else:
            gap = interval
        if event['object_id'] != previous['object_id']:
            gap += config.get('object_transition_interval_extra_seconds', 0.0)
        if config.get('operator_transfer_adaptive_cadence'):
            threshold = config.get('operator_transfer_distance_threshold_m', 3.0)
            distance = event.get('operator_transfer_distance_m', 0.0)
            if distance > threshold:
                gap += min(config.get('operator_transfer_max_extra_seconds', 1.5),
                           (distance - threshold) * config.get(
                               'operator_transfer_seconds_per_meter', .75))
        planned_gaps.append(gap)
    target_span = config.get('event_span_seconds')
    if target_span is not None:
        if not math.isfinite(target_span) or target_span <= 0 or not planned_gaps:
            raise ValueError('The configured event span must be positive and finite')
        scale = target_span / sum(planned_gaps)
        planned_gaps = [gap * scale for gap in planned_gaps]
    for index, event in enumerate(events):
        planned_gap=None
        if index:
            planned_gap=round(planned_gaps[index - 1],2)
            nominal += planned_gap
        timestamp = round((nominal + rng.uniform(-0.3, 0.3)) * fps) / fps
        event.update(event_index=index, planned_start_seconds=round(timestamp, 6),
                     start_tolerance_seconds=tolerance)
        if planned_gap is not None:event['planned_gap_seconds']=planned_gap
    timing_policy=config.get('timing_policy','planned_start_windows_without_retiming')
    if timing_policy not in ('planned_start_windows_without_retiming','frozen_relative_intervals'):
        raise ValueError('Unsupported event timing policy')
    program = {'schema_version': 1, 'seed': config['seed'], 'scene_signature': scene_signature(graph, config),
               'config_signature': digest(planning_config(config)), 'events': events,
               'duration_range_seconds': config.get('duration_range_seconds', [60, 70]),
               'timing_policy': timing_policy,
               'supported_event_families': ['movement', 'state_change']}
    program['plan_id'] = digest(program)
    validate_program(program, graph, config)
    return program


def validate_program(program, graph, config):
    payload = {k: v for k, v in program.items() if k != 'plan_id'}
    if program.get('schema_version') != 1 or digest(payload) != program.get('plan_id'):
        raise ValueError('The event program is invalid or has been modified')
    if program['config_signature'] != digest(planning_config(config)):
        raise ValueError('The event program configuration differs from the renderer configuration')
    if program['scene_signature'] != scene_signature(graph, config):
        raise ValueError('The scene differs from the scene used for planning')
    events = program['events']
    if sum(e['requested_visibility'] == 'offscreen' for e in events) != config.get('offscreen_event_count', 2):
        raise ValueError('The event program has the wrong offscreen event count')
    if len(events) != config.get('event_count', 10):
        raise ValueError('The event program has the wrong event count')
    nodes = {n['id']: n for n in graph['nodes']}
    states = {key: set(node['states']) for key, node in nodes.items()}
    held, done = {}, set()
    previous_time = -1.0
    for index, event in enumerate(events):
        if event['event_index'] != index or event['task'] in done:
            raise ValueError('Event indexes and task names must be unique and ordered')
        if not set(event['depends_on']) <= done:
            raise ValueError('An event precedes its dependency')
        timestamp = event['planned_start_seconds']
        tolerance = event['start_tolerance_seconds']
        if not math.isfinite(timestamp) or not previous_time < timestamp < program['duration_range_seconds'][1] or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Invalid planned event time window')
        previous_time = timestamp
        if event['requested_visibility'] not in ('in_view', 'offscreen'):
            raise ValueError('Unsupported visibility intention')
        if config.get('motion_contract'):
            route=config['motion_contract']['event_routes'][index]
            for key in ('observer_approach_waypoints','observer_hidden_waypoints'):
                if event.get(key)!=route[key.replace('observer_','')]:
                    raise ValueError('The event route differs from the frozen motion contract')
            if event.get('observer_action_waypoint')!=route['action_waypoint']:
                raise ValueError('The manipulation waypoint differs from the frozen motion contract')
            if event.get('operator_approach_id')!=route['operator_approach_id']:
                raise ValueError('The operator route differs from the frozen motion contract')
        actor, target, verb = event['injector_character'], event['object_id'], event['verb']
        if actor not in (1, 2) or event['injector_node_id'] != actor + 1:
            raise ValueError('Only the two operating characters may cause events')
        if verb in STATE_EFFECTS:
            old, new, _ = STATE_EFFECTS[verb]
            prop = 'CAN_OPEN' if verb in ('Open', 'Close') else 'HAS_SWITCH'
            if prop not in nodes[target]['properties'] or old not in states[target]:
                raise ValueError('The planned state change has an invalid precondition')
            states[target].remove(old)
            states[target].add(new)
        elif verb == 'Grab':
            if actor in held or target in held.values() or 'GRABBABLE' not in nodes[target]['properties'] or support_of(graph, target) is None:
                raise ValueError('The planned pickup has an invalid precondition')
            held[actor] = target
        elif verb == 'PutBack':
            if held.get(actor) != target or 'SURFACES' not in nodes[event['surface_id']]['properties']:
                raise ValueError('The planned placement has an invalid precondition')
            del held[actor]
        else:
            raise ValueError('Unsupported native event operation')
        done.add(event['task'])
    if held:
        raise ValueError('The episode must finish with empty hands')


def check_execution(program, events):
    if digest({k: v for k, v in program.items() if k != 'plan_id'}) != program.get('plan_id'):
        raise ValueError('The saved event program was modified')
    if len(program['events']) != len(events):
        raise ValueError('The renderer did not execute the complete event program')
    previous_start=None
    relative=program.get('timing_policy')=='frozen_relative_intervals'
    for planned, actual in zip(program['events'], events):
        for key in ('task', 'verb', 'object_id', 'surface_id', 'injector_character', 'requested_visibility'):
            if planned[key] != actual[key]:
                raise ValueError(f'The renderer changed the planned event: {key}')
        if planned.get('object_description') != actual.get('object_description'):
            raise ValueError('The renderer changed the planned object description')
        if not math.isfinite(actual['start']):
            raise ValueError(f'Event {planned["event_index"]} has an invalid start time')
        if relative and previous_start is not None:
            gap=actual['start']-previous_start
            if abs(gap-planned['planned_gap_seconds'])>planned['start_tolerance_seconds']+1e-6:
                raise ValueError(f'Event {planned["event_index"]} missed its planned relative interval')
        elif abs(actual['start']-planned['planned_start_seconds'])>planned['start_tolerance_seconds']+1e-6:
            raise ValueError(f'Event {planned["event_index"]} missed its planned start window')
        previous_start=actual['start']


def audit_execution(root):
    from pathlib import Path
    root = Path(root)
    program = json.loads((root / 'event_program.json').read_text())
    events = json.loads((root / 'events.json').read_text())
    report = {'scope': 'planned_event_execution', 'plan_id': program['plan_id'],
              'accepted': False, 'planned_count': len(program['events']), 'actual_count': len(events),
              'start_deviations_seconds': [round(e['start'] - p['planned_start_seconds'], 6)
                                          for p, e in zip(program['events'], events)]}
    output = root / 'plan_execution.json'
    try:
        if (root / 'config.json').exists() or (root / 'planning_graph.json').exists():
            validate_program(program, json.loads((root / 'planning_graph.json').read_text()),
                             json.loads((root / 'config.json').read_text()))
        check_execution(program, events)
    except (ValueError, KeyError, OSError) as exc:
        report['failure'] = str(exc)
        output.write_text(json.dumps(report, indent=2))
        raise
    report['accepted'] = True
    output.write_text(json.dumps(report, indent=2))
    return report


def preferred_start(event, previous_start, interval, timing_policy='planned_start_windows_without_retiming'):
    """Keep cadence inside the immutable event window as native action times vary."""
    planned = event['planned_start_seconds']
    tolerance = event['start_tolerance_seconds']
    if previous_start is None:
        return planned
    if timing_policy=='frozen_relative_intervals':
        return previous_start+interval
    return min(planned+tolerance,max(planned,previous_start+max(0,interval-tolerance)))


def event_window_deadline(event, preferred, timing_policy='planned_start_windows_without_retiming'):
    """Return the latest valid start without shortening an absolute window."""
    if timing_policy == 'frozen_relative_intervals':
        return preferred + event['start_tolerance_seconds']
    return event['planned_start_seconds'] + event['start_tolerance_seconds']


def check_injection_start(event, timestamp, previous_start, config):
    """Reject an impossible cadence before rendering another native manipulation."""
    relative=config.get('timing_policy')=='frozen_relative_intervals'
    if (previous_start is None or not relative) and abs(timestamp - event['planned_start_seconds']) > event['start_tolerance_seconds'] + 1e-6:
        raise ValueError(f'Event {event["event_index"]} starts outside its frozen window at {timestamp:.2f}s')
    if previous_start is not None:
        gap = timestamp - previous_start
        center = event.get('planned_gap_seconds',config.get('event_interval_seconds',6.0))
        tolerance = config.get('event_interval_tolerance_seconds',event['start_tolerance_seconds'])
        if abs(gap - center) > tolerance + 1e-6:
            raise ValueError(f'Event {event["event_index"]} has an invalid start gap of {gap:.2f}s')


def operation_command_time(event, preferred, previous_start, config):
    """Schedule the native command before its measured manipulation onset."""
    manipulation = {'Open': 3.25, 'Close': 3.25, 'SwitchOn': 1.6,
                    'SwitchOff': 1.6, 'Grab': 1.45, 'PutBack': 1.45}[event['verb']]
    lead = max(0.0, float(event.get('measured_operation_seconds') or manipulation) - manipulation)
    earliest = event['planned_start_seconds'] - event['start_tolerance_seconds']
    if previous_start is not None:
        earliest = max(earliest, previous_start + event.get('planned_gap_seconds',
            config['event_interval_seconds']) - config['event_interval_tolerance_seconds'])
    # Probe durations include approach work that may already be complete here.
    # A command cannot start before the earliest legal event time unless a
    # positive preparation delay is guaranteed, which the native API does not do.
    # An unnecessarily early first event leaves a long gap before a slower
    # second action. Use the middle of the first window's early half.
    command_limit=(event['planned_start_seconds']-event['start_tolerance_seconds']/2
                   if previous_start is None else earliest+.25)
    command = max(earliest, min(preferred - lead - .35, command_limit))
    return command, lead
