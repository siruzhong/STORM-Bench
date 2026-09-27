"""Check collection and mask auditing without a simulator installation."""
import json

from PIL import Image

from storm_virtualhome_qa import collection, retime, visibility


def test_collect_visibility_and_retime(tmp_path):
    source = tmp_path / 'rollout'
    capture = source / 'captures' / 'episode'
    evidence = capture / 'evidence'
    evidence.mkdir(parents=True)

    def write(path, value):
        path.write_text(json.dumps(value))

    metadata = dict(episode_id='episode', capture='captures/episode', scene=0,
                    room_id=1, room='kitchen', duration_seconds=6,
                    hashes={'video.mp4': '0' * 64})
    write(source / 'manifest.json', {'episodes': [metadata]})
    event = dict(event_index=0, object_id=10, object='cabinet', object_description='cabinet',
                 verb='Open', start=2, end=3, before_time=1, discovery_time=4,
                 visibility='onscreen', before_state='closed', after_state='open',
                 stage_keys={'before': 'before', 'after': 'after'})
    write(capture / 'events.json', [event])
    write(capture / 'stages.json', {'before': {'time': 1, 'visible_pixels': 256},
                                  'after': {'time': 4, 'visible_pixels': 256}})
    write(capture / 'qa.json', {'sample_fps': 20})
    for key, state in [('before', 'CLOSED'), ('after', 'OPEN')]:
        write(evidence / f'{key}.json', dict(
            graph={'nodes': [{'id': 10, 'class_name': 'cabinet', 'states': [state]}], 'edges': []},
            color=[1, 0, 0], instance_colors={'10': [1, 0, 0]}))
    Image.new('RGB', (16, 16), (255, 0, 0)).save(capture / 'mask.png')
    frames = []
    for frame in range(120):
        rgb = 'evidence/before.png' if frame == 20 else 'evidence/after.png' if frame == 80 else f'rgb/{frame}.png'
        frames.append(dict(frame=frame, rgb=rgb, mask='mask.png'))
    (capture / 'frame_manifest.jsonl').write_text(''.join(json.dumps(f) + '\n' for f in frames))
    reference = tmp_path / 'reference.jsonl'
    reference.write_text('{}\n')
    output = tmp_path / 'audit'
    collection.main(['--source', str(source), '--reference', str(reference),
                     '--output', str(output / 'evidence_bundle.json')])
    bundle = json.loads((output / 'evidence_bundle.json').read_text())
    assert bundle['episodes'][0]['metadata']['capture'] == str(capture.resolve())
    assert [o['state'] for o in bundle['episodes'][0]['observations']] == ['closed', 'open']
    visibility.main(['--evidence', str(output / 'evidence_bundle.json'),
                     '--output', str(output / 'visibility_audit.json'), '--workers', '1'])
    audit = json.loads((output / 'visibility_audit.json').read_text())['episodes'][0]
    assert audit['counts'] == [[256]] * 120
    retime.main([str(output)])
    revised = json.loads((output / 'retimed_evidence_bundle.json').read_text())['episodes'][0]
    assert revised['original_events'][0]['discovery_time'] == 4
    assert revised['events'][0]['discovery_time'] == 61 / 20
