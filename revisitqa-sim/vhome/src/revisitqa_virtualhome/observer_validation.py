"""Verify observer roles and visibility from recorded frames."""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .recording import visible_pixels
from .debug_view import camera_audit, target_mask
from .scene import inside_closed, on_surface, held_by


def check_roles(actions, allowed_characters=(0, 1)):
    for row in actions:
        seen = set()
        for command in row['action'].split('|'):
            match = re.fullmatch(r'<char(\d+)> \[(\w+)\].*', command.strip())
            if match is None or int(match[1]) not in allowed_characters or match[1] in seen:
                raise ValueError('Unexpected character action')
            seen.add(match[1])
            if match[1] == '0' and match[2] not in ('Walk', 'WalkTowards', 'TurnTo',
                                                    'TurnLeft', 'TurnRight'):
                raise ValueError('The observer performed a manipulation')


def audit_pair(root, event, rows, lower_bound=0):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    actions = json.loads((root / 'actions.json').read_text())
    all_stages = json.loads((root / 'stages.json').read_text())
    keys = event.get('stage_keys', {name: name for name in
        ('before', 'hidden_unobserved', 'absence_discovered', 'returned')})
    stages = {name: all_stages[key] for name, key in keys.items() if key in all_stages}
    upper_bound = stages.get('returned', {}).get('time', float('inf'))
    def evidence(name, suffix):
        return root / 'evidence' / (keys[name] + suffix)
    check_roles(actions)
    fps = config['render']['fps']
    start, end = (event['disappear'][k] for k in ('start', 'end'))
    snapshots = {name: json.loads((evidence(name, '.json')).read_text())
                 for name in ('before', 'hidden_unobserved', 'absence_discovered', 'returned')}
    for phase, snapshot in snapshots.items():
        if snapshot['color_collision_ids']:
            raise ValueError('Ambiguous checkpoint mask')
        image = np.asarray(Image.open(evidence(phase, '.seg.png')))
        count = visible_pixels(image, snapshot['color'])
        if phase in ('before', 'returned'):
            if count < config['evidence']['min_visible_pixels']:
                raise ValueError('A visible checkpoint has insufficient target pixels')
        elif count:
            raise ValueError('An absent checkpoint contains target pixels')
    if config.get('keep_container_open'):
        for phase in ('hidden_unobserved', 'absence_discovered'):
            if not any(e['from_id'] == event['object_id'] and e['to_id'] == event['container_id'] and e['relation_type'] == 'INSIDE' for e in snapshots[phase]['graph']['edges']):
                raise ValueError('The target was not stored in the container')
    elif config.get('compact_protocol') and not event.get('held_by_injector'):
        contained = any(e['from_id'] == event['object_id'] and e['to_id'] == event['container_id'] and e['relation_type'] == 'INSIDE'
                        for e in snapshots['hidden_unobserved']['graph']['edges'])
        if not contained or not inside_closed(snapshots['absence_discovered']['graph'], event['object_id'], event['container_id']):
            raise ValueError('The storage or closed-container inspection state was not reached')
    elif not inside_closed(snapshots['hidden_unobserved']['graph'], event['object_id'], event['container_id']):
        raise ValueError('The hidden state was not reached')
    if event.get('held_by_injector'):
        if not all(held_by(snapshots[phase]['graph'], event['object_id']) for phase in ('before', 'returned')):
            raise ValueError('The injector must visibly hold the target at both observation checkpoints')
    elif not on_surface(snapshots['returned']['graph'], event['object_id'], event['surface_id']):
        raise ValueError('The return state was not reached')
    reach_distances = {}
    for phase, object_id in (('before', event['object_id']), ('hidden_unobserved', event['container_id'])):
        nodes = {n['id']: n for n in snapshots[phase]['graph']['nodes']}
        actor_position = nodes[2]['obj_transform']['position']
        object_position = nodes[object_id]['obj_transform']['position']
        distance = sum((actor_position[i] - object_position[i]) ** 2 for i in (0, 2)) ** 0.5
        if distance > 2.5:
            raise ValueError(f'The injector did not reach the interaction site: {phase}')
        reach_distances[phase] = distance
    registration = None
    vacant_fraction = context_fraction = None
    neighbor_ids = []
    if event.get('held_by_injector'):
        inspected = np.asarray(Image.open(evidence('absence_discovered', '.seg.png')))
        actor_color = snapshots['absence_discovered']['instance_colors']['2']
        if visible_pixels(inspected, actor_color) < 40:
            raise ValueError('The injector must be visible during the absence inspection')
    else:
        original_mask = target_mask(np.asarray(Image.open(evidence('before', '.seg.png'))),
                                    snapshots['before']['color'])
        inspected_mask = np.asarray(Image.open(evidence('absence_discovered', '.seg.png')))
        surface_color = snapshots['absence_discovered']['instance_colors'][str(event['surface_id'])]
        exposed_surface = target_mask(inspected_mask, surface_color)
        registration = None
        if config.get('continuous_observer'):
            from .registration import align_region
            source_seg = np.asarray(Image.open(evidence('before', '.seg.png')))
            def actors(seg, snapshot):
                return np.logical_or.reduce([target_mask(seg, snapshot['instance_colors'][key])
                                             for key in ('1', '2')])
            original_mask, registration = align_region(
                np.asarray(Image.open(evidence('before', '.png')).convert('RGB')),
                np.asarray(Image.open(evidence('absence_discovered', '.png')).convert('RGB')),
                original_mask, actors(source_seg, snapshots['before']),
                actors(inspected_mask, snapshots['absence_discovered']))
            Image.fromarray(original_mask.astype(np.uint8) * 255).save(evidence('before', '.aligned_region.png'))
        vacant_fraction = float((original_mask & exposed_surface).sum() / max(1, original_mask.sum()))
        visible_context = exposed_surface.copy()
        neighbor_ids = []
        if registration is not None:
            initial_nodes = {n['id']: n for n in snapshots['before']['graph']['nodes']}
            final_nodes = {n['id']: n for n in snapshots['absence_discovered']['graph']['nodes']}
            origin = np.array(initial_nodes[event['object_id']]['obj_transform']['position'])
            supported = {e['from_id'] for e in snapshots['before']['graph']['edges']
                         if e['relation_type'] == 'ON' and e['to_id'] == event['surface_id']}
            for node_id in supported - {event['object_id']}:
                old_position = np.array(initial_nodes[node_id]['obj_transform']['position'])
                new_position = np.array(final_nodes[node_id]['obj_transform']['position'])
                if np.linalg.norm(old_position - origin) <= 0.3 and np.linalg.norm(new_position - old_position) <= 0.005:
                    neighbor_ids.append(node_id)
                    visible_context |= target_mask(inspected_mask,
                        snapshots['absence_discovered']['instance_colors'][str(node_id)])
        context_fraction = float((original_mask & visible_context).sum() / max(1, original_mask.sum()))
        # Contents left on the support can occupy part of a bowl's former silhouette.
        context_clear = registration is not None and vacant_fraction >= 0.25 and context_fraction >= 0.9
        if vacant_fraction < 0.5 and not context_clear:
            raise ValueError(f'The original target region is not visibly clear: {vacant_fraction:.3f}')
    samples = []
    first_seen = None
    for row in rows:
        if not lower_bound <= row['time'] < upper_bound:
            continue
        pixels, counts = row['target_pixels'], row['monitored_pixels']
        sample = {k: v for k, v in row.items() if k != 'monitored_collisions'}
        if start <= sample['time'] < end:
            if pixels:
                raise ValueError('The alleged offscreen event contains visible target pixels')
            if row.get('monitored_collisions', {}).get('2'):
                raise ValueError('The injector mask is ambiguous')
            for key in (str(event['surface_id']), '2'):
                if key not in counts or counts[key] != 0:
                    raise ValueError('The injector and original support must be outside the observer image')
            sample['offscreen_injection'] = True
        if sample['time'] >= stages['absence_discovered']['time'] and pixels >= config['evidence']['min_visible_pixels'] and first_seen is None:
            first_seen = sample['time']
        samples.append(sample)
    hidden = [r for r in samples if r.get('offscreen_injection')]
    if not hidden:
        raise ValueError('No frames in the offscreen interval')
    if first_seen is None:
        raise ValueError('The target was never seen again')
    if event['disappear']['discovery_confirmed_at'] <= end:
        raise ValueError('Absence discovery must follow the hidden event')
    visible_return_frames = sum(r['time'] >= stages['absence_discovered']['time'] and
                                r['monitored_pixels'].get('2', 0) >= 40 for r in samples)
    if visible_return_frames < 3:
        raise ValueError('The injector is not visible during the return sequence')
    report = {'observer_character': 0, 'injector_character': 1,
              'observer_manipulations': 0, 'camera_index': config['camera']['recording_camera'],
              'frames': len(samples), 'offscreen_injection_frames': len(hidden),
              'injector_reach_distances': reach_distances,
              'visible_return_injector_frames': visible_return_frames,
              'offscreen_target_max_pixels': max(r['target_pixels'] for r in hidden),
              'offscreen_injector_max_pixels': max(r['monitored_pixels']['2'] for r in hidden),
              'container_color_collision_ids': [int(key) for key, color in snapshots['before']['instance_colors'].items()
                  if key != str(event['container_id']) and np.allclose(color,
                      snapshots['before']['instance_colors'][str(event['container_id'])], atol=1e-6)],
              'offscreen_monitored_max_pixels': {key: max(r['monitored_pixels'].get(key, 0) for r in hidden)
                  for key in map(str, config.get('monitor_ids', []))},
              'first_reobserved_target_time': first_seen,
              'absence_confirmed_at': stages['absence_discovered']['time'],
              'vacant_original_target_fraction': vacant_fraction,
              'inspection_alignment': registration,
              'visible_original_context_fraction': context_fraction,
              'static_neighbor_ids': sorted(neighbor_ids)}
    return report


def audit_observer(root):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    events = json.loads((root / 'events.json').read_text())
    if not events:
        raise ValueError('No event pairs were recorded')
    manifest = [json.loads(line) for line in (root / 'frame_manifest.jsonl').open()]
    from .schedule import check_duration
    check_duration(len(manifest) / config['render']['fps'], config)
    rows = []
    for row in manifest:
        if row['frame'] != len(rows):
            raise ValueError('The frame manifest is not contiguous')
        if row.get('color_collision_ids'):
            raise ValueError('Ambiguous target color')
        seg = np.asarray(Image.open(root / row['mask']).convert('RGB'))
        rows.append({'frame': row['frame'], 'time': row['frame'] / config['render']['fps'],
                     'target_pixels': visible_pixels(seg, row['color']),
                     'monitored_pixels': {k: visible_pixels(seg, v) for k, v in row.get('monitored_colors', {}).items()},
                     'monitored_collisions': row.get('monitored_collisions', {})})
    reports = []
    lower_bound = 0
    for event in events:
        reports.append(audit_pair(root, event, rows, lower_bound))
        lower_bound = event.get('appear', {}).get('discovery_confirmed_at', float('inf'))
    report = dict(reports[0]) if len(events) == 1 else {'cycles': reports}
    report.update(frames=len(rows), observer_manipulations=0,
                  offscreen_injection_frames=sum(r['offscreen_injection_frames'] for r in reports),
                  offscreen_target_max_pixels=max(r['offscreen_target_max_pixels'] for r in reports),
                  offscreen_injector_max_pixels=max(r['offscreen_injector_max_pixels'] for r in reports),
                  **camera_audit(root))
    if 'event_count' in config:
        from .schedule import transition_schedule
        schedule = transition_schedule(events, config)
        (root / 'event_schedule.json').write_text(json.dumps(schedule, indent=2))
        report['schedule'] = schedule
    for row in rows:
        row.pop('monitored_collisions', None)
        row['offscreen_injection'] = any(e['disappear']['start'] <= row['time'] < e['disappear']['end'] for e in events)
    (root / 'visibility.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    if config.get('continuous_observer') and config['render'].get('save_pose_data'):
        from .motion import audit_motion
        report['motion'] = audit_motion(root)
    (root / 'observer_validation.json').write_text(json.dumps(report, indent=2))
    return report
