#!/usr/bin/env python3
"""Run agent-driven presence-change episodes in VirtualHome."""
import argparse
import json
import sys
import subprocess
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.scene import candidates, action, inside_closed, on_surface
from storm_virtualhome.recording import Recorder
from storm_virtualhome.qa import build_questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/rollout.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--display', help='Existing GPU X display; otherwise start Xvfb')
    parser.add_argument('--xorg-root', type=Path, help='Extracted Xorg runtime for NVIDIA rendering')
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--debug', action='store_true', help='Record synchronized masks for a highlighted debug video')
    parser.add_argument('--event-pairs', type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.debug:
        config['debug'] = True
    if args.event_pairs is not None:
        config['event_pairs'] = args.event_pairs
    if config['event_pairs'] < 1:
        parser.error('--event-pairs must be at least 1')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    pairs, rejected = [], []
    with UnityProcess(args.executable, args.output / 'logs', gpu_index=args.gpu_index, display=args.display, xorg_root=args.xorg_root) as client:
        client.reset(config['scene'])
        client.follow_camera(**config['camera'])
        static_count = client.command('camera_count')['value']
        names = client.command('character_cameras')['message']
        if isinstance(names, str):
            names = json.loads(names)
        camera_index = static_count + names.index(config['camera']['name'])
        client.character(config['character'], config['room'])
        graph = client.graph()
        (args.output / 'initial_graph.json').write_text(json.dumps(graph, indent=2))
        containers, objects = candidates(graph, config['room'], config['seed'])
        if not containers:
            raise RuntimeError('No supported container in this room')
        target_classes = config.get('target_classes')
        if target_classes:
            objects = [pair for pair in objects if pair[0]['class_name'] in target_classes]
            objects.sort(key=lambda pair: target_classes.index(pair[0]['class_name']))
        container = containers[0]
        recorder = Recorder(client, args.output, camera_index, config)
        try:
            for target, surface in objects:
                if len(pairs) >= config['event_pairs']:
                    break
                if target['class_name'] in {p['object'] for p in pairs}:
                    continue
                index = len(pairs)
                print(f'TARGET {target["class_name"]} ({target["id"]})', flush=True)
                recorder.set_target(target)
                recorder.run(action('Walk', target))
                before = recorder.snapshot(f'{target["id"]}_before', target['id'])
                if before['color_collision_ids'] or before['visible_pixels'] < config['evidence']['min_visible_pixels']:
                    rejected.append({'id': target['id'], 'reason': 'ambiguous mask or insufficient visibility', 'pixels': before['visible_pixels'], 'color_collision_ids': before['color_collision_ids']})
                    (args.output / 'rejected.json').write_text(json.dumps(rejected, indent=2))
                    continue
                start = recorder.time
                recorder.run(action('Grab', target))
                recorder.run(action('Walk', container))
                if 'OPEN' not in next(n for n in client.graph()['nodes'] if n['id'] == container['id'])['states']:
                    recorder.run(action('Open', container))
                recorder.run(action('PutIn', target, container))
                recorder.run(action('Close', container))
                hidden = recorder.snapshot(f'{target["id"]}_hidden', target['id'])
                hidden_ok = inside_closed(hidden['graph'], target['id'], container['id']) and hidden['visible_pixels'] == 0
                show_start = recorder.time
                recorder.run(action('Open', container))
                recorder.run(action('Grab', target))
                recorder.run(action('Close', container))
                recorder.run(action('Walk', surface))
                recorder.run(action('PutBack', target, surface))
                recorder.run(action('Walk', target))
                after = recorder.snapshot(f'{target["id"]}_returned', target['id'])
                restored_ok = on_surface(after['graph'], target['id'], surface['id']) and not after['color_collision_ids'] and after['visible_pixels'] >= config['evidence']['min_visible_pixels']
                if hidden_ok and restored_ok:
                    pair = {'index': index, 'object_id': target['id'], 'object': target['class_name'],
                            'container_id': container['id'], 'container': container['class_name'],
                            'surface_id': surface['id'], 'surface': surface['class_name'],
                            'event_family': 'presence_change', 'mechanism': 'agent_container',
                            'visible_pixels': [before['visible_pixels'], hidden['visible_pixels'], after['visible_pixels']],
                            'disappear': {'action': 'disappear', 'start': start, 'end': hidden['time']},
                            'appear': {'action': 'appear', 'start': show_start, 'end': after['time']}}
                    pairs.append(pair)
                    print('ACCEPTED', json.dumps(pair), flush=True)
                else:
                    rejected.append({'id': target['id'], 'reason': 'event verification failed',
                                     'hidden_ok': hidden_ok, 'restored_ok': restored_ok,
                                     'pixels': [before['visible_pixels'], hidden['visible_pixels'], after['visible_pixels']]})
                (args.output / 'events.json').write_text(json.dumps(pairs, indent=2))
                (args.output / 'rejected.json').write_text(json.dumps(rejected, indent=2))
                if not (hidden_ok and restored_ok):
                    raise RuntimeError('A completed cycle failed visual verification; this episode cannot provide count QA')
        finally:
            recorder.close()
        if len(pairs) != config['event_pairs']:
            raise RuntimeError(f'Only {len(pairs)}/{config["event_pairs"]} pairs passed; inspect evidence and rejected.json')
        if config.get('debug'):
            from storm_virtualhome.debug_view import camera_audit
            camera_audit(args.output)
        episode_id = args.output.name
        questions = build_questions(episode_id, pairs, recorder.time, config['seed'])
        doc = {'episode_id': episode_id, 'video_path': 'video.mp4', 'duration_sec': recorder.time,
               'sample_fps': config['render']['fps'], 'qa_source': 'virtualhome_programmatic',
               'questions': questions}
        (args.output / 'qa.json').write_text(json.dumps(doc, indent=2))
        (args.output / 'questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in questions))
        print(json.dumps({'output': str(args.output), 'duration_sec': recorder.time, 'frames': recorder.frames,
                          'event_pairs': len(pairs), 'qa_types': dict(Counter(q['question_type'] for q in questions))}), flush=True)

    if config.get('debug'):
        subprocess.run([sys.executable, str(Path(__file__).with_name('render_debug.py')), str(args.output)], check=True)


if __name__ == '__main__':
    main()
