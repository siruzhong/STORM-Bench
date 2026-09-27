#!/usr/bin/env python3
"""Record an observer watching changes made by a separate character."""
import argparse
import json
import shutil
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.recording import Recorder
from storm_virtualhome.scene import action, inside_closed, on_surface, held_by
from storm_virtualhome.schedule import transition_schedule
from finalize_observer import finalize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--executable', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/observer.yaml')
    p.add_argument('--xorg-root', type=Path)
    p.add_argument('--gpu-index', type=int, default=0)
    p.add_argument('--display')
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    event_count = config.get('event_count', 2)
    if not isinstance(event_count, int) or event_count < 2 or event_count % 2:
        raise ValueError('event_count must be a positive even number of transitions')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    runtime_manifest = Path(args.executable).resolve().parent / 'camera_guard.json'
    if runtime_manifest.is_file():
        shutil.copy2(runtime_manifest, root / 'runtime_manifest.json')
    with UnityProcess(args.executable, root / 'logs', gpu_index=args.gpu_index,
                      display=args.display, xorg_root=args.xorg_root) as client:
        client.reset(config['scene'])
        client.follow_camera(**config['camera'])
        static_count = client.command('camera_count')['value']
        names = client.command('character_cameras')['message']
        if isinstance(names, str):
            names = json.loads(names)
        camera_index = static_count + names.index(config['camera']['name'])
        client.character(**config['observer'])
        client.character(**config['injector'])
        config['camera']['recording_camera'] = str(camera_index)
        (root / 'config.json').write_text(json.dumps(config, indent=2))
        graph = client.graph()
        (root / 'initial_graph.json').write_text(json.dumps(graph, indent=2))
        nodes = {n['id']: n for n in graph['nodes']}
        target, surface, container, away = [nodes[config[k]] for k in ('target_id', 'surface_id', 'container_id', 'away_id')]
        if config.get('compact_protocol'):
            setup_lines = [action('Walk', nodes[config['observer_anchor_id']], character=0),
                           action('Walk', nodes[264], character=0) + ' | ' + action('Walk', target, character=1),
                           action('Walk', nodes[config['observer_anchor_id']], character=0) + ' | ' + action('Open', container, character=1)]
            if config.get('held_target'):
                setup_lines.append(action('TurnTo', nodes[313], character=0) + ' | ' + action('Grab', target, character=1))
            for index, line in enumerate(setup_lines):
                print('SETUP', index, line, flush=True)
                folder = root / 'setup' / f'{index:04d}'
                folder.mkdir(parents=True)
                response = client.render([line], folder, f'setup_{index:04d}', camera=str(camera_index),
                                         record=True, width=320, height=240)
                status = json.loads(response.get('message', '{}'))
                if any(item.get('message') != 'Success' for item in status.values()):
                    raise RuntimeError(f'Scene preparation failed: {status}')
            (root / 'initial_graph.json').write_text(json.dumps(client.graph(), indent=2))
        recorder = Recorder(client, root, camera_index, config)
        recorder.set_target(target)
        stages = {}
        events = []
        cycle = 0
        phase = "setup"
        route_index = 0
        def run(who, verb, *objects):
            nonlocal route_index
            if who == 0 and verb not in ('Walk', 'WalkTowards', 'TurnTo'):
                raise ValueError('The observer may only walk or turn')
            pose_file = root / 'logs/observer_pose.csv'
            line = action(verb, *objects, character=who)
            if who == 1 and config.get('continuous_observer'):
                pose_file.unlink(missing_ok=True)
                route = config['patrol_routes'][phase]
                waypoint = nodes[route[route_index % len(route)]]
                route_index += 1
                navigation = 'WalkTowards' if config.get('compact_protocol') and phase != 'reveal' else 'Walk'
                if config.get('held_target') and phase == 'return' and route_index == 2:
                    navigation = 'Walk'
                line = action(navigation, waypoint, character=0) + ' | ' + line
            elif who == 0:
                pose_file.unlink(missing_ok=True)
            else:
                pose_file.write_text('hold')
                # Latch the Unity transform before the next script resets idle characters.
                client.image(camera_index, width=config['render']['width'], height=config['render']['height'])
            print(f'ACTION char{who} {verb}', flush=True)
            return recorder.run(line)
        def snap(name):
            key = f'cycle_{cycle:02d}_{name}' if event_count > 2 else name
            result = recorder.snapshot(key, target['id'])
            stages[key] = {k: v for k, v in result.items() if k != 'graph'}
            (root / 'stages.json').write_text(json.dumps(stages, indent=2))
            print('SNAPSHOT', key, stages[key], flush=True)
            return result
        try:
            if not config.get('compact_protocol'):
                run(0, 'Walk', nodes[config['observer_anchor_id']])
            if not config.get('compact_protocol'):
                run(1, 'Walk', target)
            if config.get('continuous_observer') and not config.get('compact_protocol'):
                run(0, 'Walk', nodes[config['observer_anchor_id']])
            run(0, 'TurnTo', nodes[config['look_at_id']])
            before = snap('before')
            injector_position = next(n for n in before['graph']['nodes'] if n['id'] == 2)['obj_transform']['position']
            target_position = target['obj_transform']['position']
            if sum((injector_position[i] - target_position[i]) ** 2 for i in (0, 2)) > 2.5 ** 2:
                raise RuntimeError('The injector did not reach the target')
            if before['visible_pixels'] < config['evidence']['min_visible_pixels'] or before['color_collision_ids']:
                raise RuntimeError('The initial target must be visible and have a unique instance color')
            for cycle in range(event_count // 2):
                if cycle:
                    before = snap('before')
                    if before['visible_pixels'] < config['evidence']['min_visible_pixels']:
                        raise RuntimeError('The next cycle must begin with a visible target')
                if not config.get('compact_protocol'):
                    run(0, 'TurnTo', away)
                run(0, 'Walk', nodes[config['away_anchor_id']])
                away_view = snap('looking_away')
                if away_view['visible_pixels']:
                    raise RuntimeError('The target remains visible after the observer turns away')
                phase, route_index = 'hidden', 0
                hide_start = recorder.time
                if not config.get('held_target'):
                    retrieved = run(1, 'Grab', target)
                if not config.get('compact_protocol'):
                    run(1, 'Walk', container)
                    run(1, 'Open', container)
                run(1, 'PutIn', target, container)
                if (not config.get('compact_protocol') or config.get('held_target')) and not config.get('keep_container_open'):
                    run(1, 'Close', container)
                hidden = snap('hidden_unobserved')
                contained = any(e['from_id'] == target['id'] and e['to_id'] == container['id'] and e['relation_type'] == 'INSIDE' for e in hidden['graph']['edges'])
                if not contained or (not config.get('compact_protocol') and not inside_closed(hidden['graph'], target['id'], container['id'])):
                    raise RuntimeError('The injector did not store the target')
                hide_end = recorder.time
                reveal_start = recorder.time
                # Return to the same viewing anchor before inspecting the original location.
                if config.get('compact_protocol') and not config.get('held_target'):
                    phase, route_index = 'reveal', 0
                    run(1, 'Close', container)
                else:
                    run(0, 'Walk', nodes[config['observer_anchor_id']])
                run(0, 'TurnTo', nodes[config['look_at_id']])
                discovered = snap('absence_discovered')
                if discovered['visible_pixels']:
                    raise RuntimeError('The target remains visible during the absence inspection')
                phase, route_index = 'return', 0
                show_start = recorder.time
                if not config.get('keep_container_open'):
                    run(1, 'Open', container)
                retrieved = run(1, 'Grab', target)
                if not config.get('compact_protocol'):
                    run(1, 'Close', container)
                    run(1, 'Walk', surface)
                put_back = retrieved if config.get('held_target') else run(1, 'PutBack', target, surface)
                if config.get('continuous_observer'):
                    run(0, 'TurnTo', nodes[config['look_at_id']])
                else:
                    run(1, 'Walk', away)
                returned = snap('returned')
                returned_state = held_by(returned['graph'], target['id']) if config.get('held_target') else on_surface(returned['graph'], target['id'], surface['id'])
                if not returned_state or returned['visible_pixels'] < config['evidence']['min_visible_pixels']:
                    raise RuntimeError('The returned target must be visible in the expected state')
                event = {'object_id': target['id'], 'object': target['class_name'],
                         'surface_id': surface['id'], 'surface': surface['class_name'],
                         'container_id': container['id'], 'container': container['class_name'],
                         'event_family': 'presence_change', 'mechanism': 'other_agent_container',
                         'observer_character': 0, 'injector_character': 1,
                         'disappear': {'action': 'disappear', 'start': hide_start, 'end': hide_end,
                                       'visibility': 'offscreen', 'discovery_start': reveal_start,
                                       'discovery_confirmed_at': discovered['time']},
                         'appear': {'action': 'appear', 'start': show_start, 'end': put_back['end'],
                                    'discovery_confirmed_at': returned['time']}}
                if config.get('held_target'):
                    event['held_by_injector'] = True
                if event_count > 2:
                    event['cycle_index'] = cycle
                    event['stage_keys'] = {name: f'cycle_{cycle:02d}_{name}' for name in
                        ('before', 'looking_away', 'hidden_unobserved', 'absence_discovered', 'returned')}
                events.append(event)
                (root / 'events.json').write_text(json.dumps(events, indent=2))
                schedule = transition_schedule(events, config, complete=False)
                (root / 'event_schedule.json').write_text(json.dumps(schedule, indent=2))
            minimum_duration = config.get('duration_range_seconds', [0])[0]
            tail_index = 0
            while recorder.time < minimum_duration:
                previous_time = recorder.time
                run(0, 'Walk', nodes[(264, 268)[tail_index % 2]])
                tail_index += 1
                if recorder.time - previous_time < 0.2:
                    raise RuntimeError('The closing walk made no progress')
        finally:
            recorder.close()
    finalize(root)


if __name__ == '__main__':
    main()
