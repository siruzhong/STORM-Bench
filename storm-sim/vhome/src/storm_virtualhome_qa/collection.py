"""Collect capture annotations without interpreting graph states as visual proof."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture-root', type=Path,
                        help='Remap captures to CAPTURE_ROOT/episode_id')
    args = parser.parse_args(argv)
    source, reference, output = args.source, args.reference, args.output
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / 'manifest.json').read_text())
    result = dict(source=str(source.resolve()), reference=str(reference),
                  reference_sha256=digest(reference),
                  reference_rows=[json.loads(x) for x in reference.read_text().splitlines()],
                  manifest_sha256=digest(source / 'manifest.json'), episodes=[])
    for episode in manifest['episodes']:
        capture = Path(episode['capture'])
        if args.capture_root:
            capture = args.capture_root / episode['episode_id']
        elif not capture.is_absolute():
            capture = source / capture
        capture = capture.resolve()
        episode = {**episode, 'capture': str(capture)}
        events = json.loads((capture / 'events.json').read_text())
        stages = json.loads((capture / 'stages.json').read_text())
        frames = [json.loads(x) for x in (capture / 'frame_manifest.jsonl').read_text().splitlines()]
        frame_by_image = {x['rgb']: x['frame'] for x in frames}
        records = []
        for event in events:
            for side in ('before', 'after'):
                key = event['stage_keys'][side]
                path = capture / 'evidence' / f'{key}.json'
                evidence = json.loads(path.read_text())
                graph = evidence['graph']
                nodes = {n['id']: n for n in graph['nodes']}
                obj = nodes[event['object_id']]
                if event['verb'] in ('Open', 'Close'):
                    states = [s.lower() for s in obj['states'] if s in ('OPEN', 'CLOSED')]
                    state = states[0] if len(states) == 1 else None
                elif event['verb'] in ('SwitchOn', 'SwitchOff'):
                    states = [s.lower() for s in obj['states'] if s in ('ON', 'OFF')]
                    state = states[0] if len(states) == 1 else None
                else:
                    held = [e for e in graph['edges'] if e['to_id'] == obj['id'] and e['relation_type'].startswith('HOLDS')]
                    surfaces = [e['to_id'] for e in graph['edges'] if e['from_id'] == obj['id'] and e['relation_type'] == 'ON']
                    state = 'held by a person' if held else ('on:' + str(surfaces[0]) if len(surfaces) == 1 else None)
                records.append(dict(event_index=event['event_index'], object_id=obj['id'],
                    side=side, key=key, state=state, graph_states=obj['states'],
                    surface_names={str(i):nodes[i]['class_name'] for i in nodes},
                    source_label=event[side+'_state'], time=stages[key]['time'],
                    frame=frame_by_image['evidence/'+key+'.png'],
                    pixels=stages[key]['visible_pixels'], collisions=evidence.get('color_collision_ids', []),
                    source_json=str(path), source_sha256=digest(path)))
        result['episodes'].append(dict(metadata=episode, events=events, observations=records,
            qa=json.loads((capture/'qa.json').read_text()),
            events_sha256=digest(capture/'events.json')))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result))
    print(json.dumps(dict(episodes=len(result['episodes']), reference_sha256=result['reference_sha256'])))
