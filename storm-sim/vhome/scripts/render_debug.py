#!/usr/bin/env python3
"""Create a highlighted video from synchronized rollout frames and masks."""
import argparse
import json
import re
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.debug_view import highlight, target_mask, camera_audit
from storm_virtualhome.qa import label


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    args = p.parse_args()
    root = args.episode
    config = json.loads((root / 'config.json').read_text())
    events = json.loads((root / 'events.json').read_text())
    targets = {event['object_id'] for event in events}
    stages_path = root / 'stages.json'
    stages = json.loads(stages_path.read_text()) if stages_path.exists() else None
    audit = camera_audit(root)
    fps = config['render']['fps']
    with imageio.get_reader(str(root / 'video.mp4')) as source:
        expected_frames = source.count_frames()
    output = root / 'debug.mp4'
    if output.exists():
        raise FileExistsError(output)
    highlighted = 0
    frames = 0
    with imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8, macro_block_size=1) as writer:
        for line in (root / 'frame_manifest.jsonl').open():
            row = json.loads(line)
            assert row['frame'] == frames
            rgb = np.array(Image.open(root / row['rgb']).convert('RGB'))
            active = row.get('target_id') in targets and not row.get('color_collision_ids')
            mask = np.zeros(rgb.shape[:2], dtype=bool)
            caption = 'Transit / candidate check'
            if active:
                segmentation = np.array(Image.open(root / row['mask']).convert('RGB'))
                mask = target_mask(segmentation, row['color'])
                caption = f'{label(row["target_name"])} #{row["target_id"]}'
                highlighted += bool(mask.any())
            action_text = row['action']
            if config.get('observer_only'):
                commands = []
                for command in action_text.split('|'):
                    match = re.match(r'<char([0-9]+)> \[(\w+)\](.*)', command.strip())
                    if match:
                        role = 'Observer (blue)' if match[1] == '0' else 'Kitchen actor (green)' if match[1] == '1' else 'TV actor'
                        objects = ', '.join(re.findall(r'<([^>]+)>', match[3]))
                        commands.append(f'{role}: {match[2]} {objects}')
                    else:
                        commands.append(command)
                action_text = ' | '.join(commands)
                timestamp = frames / fps
                if config.get('event_protocol') == 'mixed':
                    current = next((e for e in reversed(events) if timestamp >= e['start'] and e['object_id'] == row.get('target_id')), None)
                    if current is not None:
                        caption = f'Event {current["event_index"]+1:02d}/{len(events):02d} | ' + caption
                        if current['start'] <= timestamp < current['end']:
                            action_text = current['visibility'].upper() + ' | ' + action_text
                        elif timestamp >= current['discovery_time']:
                            action_text = 'CHANGE OBSERVED | ' + action_text
                else:
                    event = next((e for e in reversed(events) if timestamp >= e['disappear']['start']), events[0])
                    hidden = event['disappear']
                    ordinal = sum(timestamp >= e[k]['start'] for e in events for k in ('disappear', 'appear'))
                    if ordinal:
                        caption = f'Event {ordinal:02d}/{len(events) * 2:02d} | ' + caption
                    if hidden['start'] <= timestamp < hidden['end']:
                        action_text = 'OFFSCREEN | ' + action_text
                    elif hidden['discovery_start'] <= timestamp < hidden['discovery_confirmed_at']:
                        action_text = 'INSPECT PREVIOUS LOCATION | ' + action_text
                    elif timestamp >= event['appear']['discovery_confirmed_at'] - config['evidence']['hold_seconds']:
                        action_text = 'RETURN OBSERVED | ' + action_text
            writer.append_data(highlight(rgb, mask, caption, action_text, frames / fps))
            frames += 1
    if frames != expected_frames:
        raise RuntimeError(f'Debug/clean frame mismatch: {frames} != {expected_frames}')
    report = {'frames': frames, 'fps': fps, 'highlighted_frames': highlighted,
              **audit, 'targets': sorted(targets)}
    if not highlighted:
        raise RuntimeError('No visible target was highlighted')
    (root / 'debug_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
