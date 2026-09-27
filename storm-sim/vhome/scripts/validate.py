#!/usr/bin/env python3
"""Validate a completed episode against video metadata and event evidence."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.qa import TYPES
from storm_virtualhome.recording import visible_pixels
from storm_virtualhome.scene import inside_closed, on_surface


def validate(root):
    data = json.loads((root / 'qa.json').read_text())
    events = json.loads((root / 'events.json').read_text())
    reader = imageio.get_reader(str(root / data['video_path']))
    try:
        fps = reader.get_meta_data()['fps']
        frames = reader.count_frames()
    finally:
        reader.close()
    duration = frames / fps
    assert abs(duration - data['duration_sec']) < 1 / fps
    counts = Counter(q['question_type'] for q in data['questions'])
    assert set(counts) == set(TYPES), counts
    assert len({q['id'] for q in data['questions']}) == len(data['questions'])
    for q in data['questions']:
        assert len(q['options']) == len(set(q['options'])) == 4
        assert 0 <= q['answer_index'] < 4
        assert all(0 <= start <= end <= q['query_time'] <= duration for start, end in q['evidence_spans'])
    if data.get('qa_source') == 'virtualhome_mixed_programmatic':
        from storm_virtualhome.mixed import audit_mixed
        report = audit_mixed(root)
        assert report['frames'] == frames
        assert len(data['questions']) == 6 * len(events)
        for event in events:
            assert 0 <= event['before_time'] <= event['start'] < event['end'] <= event['discovery_time'] <= duration
        return dict(report, questions=len(data['questions']), qa_types=dict(counts))
    if data.get('qa_source') == 'virtualhome_observer_programmatic':
        from storm_virtualhome.observer_validation import audit_observer
        report = audit_observer(root)
        assert report['frames'] == frames
        for event in events:
            assert 0 <= event['disappear']['start'] < event['disappear']['end'] <= event['appear']['start'] < event['appear']['end'] <= duration
        assert len(data['questions']) == 8 * len(events)
        return dict(report, fps=fps, duration_sec=duration, event_pairs=len(events),
                    questions=len(data['questions']), qa_types=dict(counts))
    for event in events:
        before, hidden, returned = event['visible_pixels']
        assert before > 0 and hidden == 0 and returned > 0
        snapshots = {}
        for phase, expected in zip(('before', 'hidden', 'returned'), event['visible_pixels']):
            stem = root / 'evidence' / f'{event["object_id"]}_{phase}'
            snapshot = json.loads(stem.with_suffix('.json').read_text())
            mask = np.array(Image.open(stem.with_suffix('.seg.png')).convert('RGB'))
            assert visible_pixels(mask, snapshot['color']) == expected
            assert not snapshot['color_collision_ids']
            snapshots[phase] = snapshot
        assert inside_closed(snapshots['hidden']['graph'], event['object_id'], event['container_id'])
        assert on_surface(snapshots['returned']['graph'], event['object_id'], event['surface_id'])
        assert 0 <= event['disappear']['start'] < event['disappear']['end'] <= event['appear']['start'] < event['appear']['end'] <= duration
    return {'frames': frames, 'fps': fps, 'duration_sec': duration,
            'event_pairs': len(events), 'questions': len(data['questions']), 'qa_types': dict(counts)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    result = validate(parser.parse_args().episode)
    print(json.dumps(result, indent=2))
