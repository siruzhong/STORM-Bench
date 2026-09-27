#!/usr/bin/env python3
"""Audit an observer capture and generate its questions and debug overlay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.observer_validation import audit_observer
from storm_virtualhome.observer_qa import build_observer_questions


def finalize(root, render_debug=True):
    root = Path(root)
    report = audit_observer(root)
    config = json.loads((root / 'config.json').read_text())
    stages = json.loads((root / 'stages.json').read_text())
    events = json.loads((root / 'events.json').read_text())
    with imageio.get_reader(str(root / 'video.mp4')) as reader:
        frames = reader.count_frames()
        fps = reader.get_meta_data()['fps']
    if frames != report['frames'] or fps != config['render']['fps']:
        raise ValueError('Video and evidence frame counts do not match')
    questions = []
    history_times = []
    cycle_reports = report.get('cycles', [report])
    for index, (event, cycle_report) in enumerate(zip(events, cycle_reports)):
        event['appear']['first_reobserved_target_time'] = cycle_report['first_reobserved_target_time']
        keys = event.get('stage_keys', {name: name for name in
            ('before', 'hidden_unobserved', 'absence_discovered', 'returned')})
        cycle_stages = {name: stages[key] for name, key in keys.items()}
        history_times.extend(cycle_stages[name]['time'] for name in ('before', 'absence_discovered', 'returned'))
        questions.extend(build_observer_questions(root.name, event, cycle_stages, frames / fps,
                         config['seed'] + index, index, history_times))
    (root / 'events.json').write_text(json.dumps(events, indent=2))
    doc = {'episode_id': root.name, 'video_path': 'video.mp4', 'duration_sec': frames / fps,
           'sample_fps': fps, 'qa_source': 'virtualhome_observer_programmatic', 'questions': questions}
    (root / 'qa.json').write_text(json.dumps(doc, indent=2))
    (root / 'questions.jsonl').write_text(''.join(json.dumps(q) + '\n' for q in questions))
    print(json.dumps(report, indent=2), flush=True)
    if render_debug:
        subprocess.run([sys.executable, str(Path(__file__).with_name('render_debug.py')), str(root)], check=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--qa-only', action='store_true')
    args = parser.parse_args()
    finalize(args.episode, render_debug=not args.qa_only)
