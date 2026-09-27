"""Record action frames and retain state snapshots for event verification."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def visible_pixels(segmentation, color):
    rgb = np.asarray(color, dtype=float)
    if rgb.max() <= 1:
        rgb = rgb * 255
    rgb = np.rint(rgb).astype(np.int16)
    return int(np.all(np.abs(segmentation.astype(np.int16) - rgb) <= 1, axis=-1).sum())


def recorded_target_visible(observation, target_id, minimum_pixels):
    """Read target visibility from the exact final frame of a recorded action."""
    pixels = (observation or {}).get('last_visible_pixels', {})
    key = str(target_id)
    return None if key not in pixels else pixels[key] >= minimum_pixels


def target_visibility_threshold(config, node):
    """Return the evidence threshold for an object's rendered footprint."""
    overrides = config.get('evidence', {}).get('min_visible_pixels_by_class', {})
    return overrides.get(node['class_name'], config['evidence']['min_visible_pixels'])


def frame_interval(start, end, fps):
    """Convert a half-open time interval to frames without floating boundary leakage."""
    import math
    return math.ceil(start * fps - 1e-6), math.ceil(end * fps - 1e-6)


def observer_frame_issue(segmentation, color):
    """Return violations that no later frame can repair in a continuous clip."""
    rgb = np.rint(np.asarray(color) * 255).astype(np.int16)
    mask = np.all(np.abs(segmentation.astype(np.int16) - rgb) <= 1, axis=-1)
    rows = np.nonzero(mask)[0]
    if not len(rows):
        return 'observer_missing'
    if rows.min() == 0:
        return 'observer_touches_upper_border'
    return None


def validate_first_person_pose(pose):
    """Require a head-height camera that follows the observer's actual position."""
    camera=np.asarray(pose['position'], dtype=float)
    observer=np.asarray(pose['observer_position'], dtype=float)
    if camera.shape != (3,) or observer.shape != (3,) or not np.isfinite([camera,observer]).all():
        raise ValueError('Invalid first-person camera attachment evidence')
    offset=camera-observer
    if not 1.35 <= offset[1] <= 1.8 or np.linalg.norm(offset[[0,2]]) > .8:
        raise ValueError('First-person camera detached from the observer')
    return float(np.linalg.norm(offset[[0,2]]))


def native_frame_mapping(clock_path):
    """Align independently created character recorders through the Unity frame clock."""
    rows=[json.loads(line) for line in Path(clock_path).read_text().splitlines()]
    observer={row['unity_frame']:row['source_frame'] for row in rows if row['actor']==0}
    maps={};last={}
    for row in rows:
        actor=str(row['actor']);frame=row['source_frame']
        if frame<last.get(actor,-1):
            maps[actor]={}
        last[actor]=frame
        shared=observer.get(row['unity_frame'])
        if shared is not None:maps.setdefault(actor,{})[frame]=shared
    return maps


def native_action_spans(folder, first_source_frame, last_source_frame, video_start, fps, frame_maps=None):
    """Map cumulative native action annotations onto this recorded video segment."""
    result={}
    for path in Path(folder).glob('*/*/ftaa_*.txt'):
        actor=path.parent.name
        if not actor.isdigit():continue
        rows=[]
        for line in path.read_text().splitlines():
            fields=line.split()
            if len(fields)!=4:continue
            try:first,last=int(fields[2]),int(fields[3])
            except ValueError:continue
            native_first,native_last=first,last
            if frame_maps is not None:
                mapping=frame_maps.get(actor,{})
                first=mapping.get(first,mapping.get(first+1))
                last=mapping.get(last,mapping.get(last-1))
                if first is None or last is None:continue
            if last<first or last<first_source_frame or first>last_source_frame:continue
            rows.append(dict(verb=fields[1],source_start=first,source_end=last,
                actor_source_start=native_first,actor_source_end=native_last,
                start=round(video_start+(max(first,first_source_frame)-first_source_frame)/fps,6),
                end=round(video_start+(min(last,last_source_frame)-first_source_frame+1)/fps,6)))
        result[actor]=rows
    return result


class Recorder:
    def __init__(self, client, output, camera_index, config):
        self.client, self.output, self.camera_index, self.config = client, Path(output), camera_index, config
        self.output.mkdir(parents=True, exist_ok=True)
        self.fps = config['render']['fps']
        self.frames = 0
        self.actions = []
        self.target = None
        self.frame_info = {}
        self.debug = bool(config.get('debug', False))
        self.last_rgb_path = None
        self.last_mask_path = None
        self.manifest = (self.output / 'frame_manifest.jsonl').open('w') if self.debug else None
        encoding_threads = int(os.environ.get('STORM_FFMPEG_THREADS', '4'))
        if not 1 <= encoding_threads <= 32:
            raise ValueError('STORM_FFMPEG_THREADS must be between 1 and 32')
        self.writer = imageio.get_writer(str(self.output / 'video.mp4'), fps=self.fps, codec='libx264',
                                        quality=8, macro_block_size=1, ffmpeg_log_level='error',
                                        ffmpeg_params=['-threads',str(encoding_threads)])

    @property
    def time(self):
        return round(self.frames / self.fps, 6)

    def set_target(self, target):
        self.target = target

    def append(self, frame, count=1):
        if not np.any(np.asarray(frame)[..., :3]):
            raise RuntimeError('The RGB renderer returned an entirely black frame')
        for _ in range(count):
            self.writer.append_data(frame)
            if self.manifest:
                row = dict(self.frame_info, frame=self.frames)
                self.manifest.write(json.dumps(row) + '\n')
            self.frames += 1

    def run(self, line,find_solution=False):
        wall_start = time.monotonic()
        number = len(self.actions)
        folder = self.output / 'actions' / f'{number:04d}'
        folder.mkdir(parents=True)
        start = self.time
        colors = self.client.colors() if self.debug else {}
        if self.debug:
            self.frame_info = self.target_info(colors)
            self.frame_info['action'] = line
        response = self.client.render([line], folder, f'action_{number:04d}',
                                      camera=self.config['camera'].get('recording_camera', self.config['camera']['name']),
                                      modalities=['normal', 'seg_inst'] if self.debug else None,
                                      find_solution=find_solution,**self.config['render'])
        render_end = time.monotonic()
        paths = sorted(folder.rglob('*_normal.png'))
        if self.config.get('observer_only'):
            # VirtualHome writes the selected camera once for each character.
            selected = [path for path in paths if path.parent.name == '0']
            if paths and not selected:
                raise RuntimeError('The observer camera stream is missing')
            paths = selected
        status = json.loads(response.get('message', '{}'))
        if any(item.get('message') != 'Success' for item in status.values()):
            if paths:
                raise RuntimeError(f'Partial native action failed after {len(paths)} frames: {line}: {status}')
            raise RuntimeError(f'Action execution failed: {line}: {status}')
        last_mask_path = None
        last_rgb_path = None
        if not paths:
            # Walking to the current position can complete without animation frames.
            frame = self.client.image(self.camera_index, width=self.config['render']['width'],
                                      height=self.config['render']['height'])
            if self.debug:
                normal = folder / 'idle_normal.png'
                mask_path = folder / 'idle_seg_inst.png'
                Image.fromarray(frame).save(normal)
                Image.fromarray(self.client.image(self.camera_index, 'seg_inst',
                    self.config['render']['width'], self.config['render']['height'])).save(mask_path)
                last_mask_path = mask_path
                last_rgb_path = normal
                self.frame_info.update(rgb=str(normal.relative_to(self.output)), mask=str(mask_path.relative_to(self.output)))
                pose_path = self.output / 'logs/camera_collision.csv.pose.json'
                if pose_path.exists():
                    self.frame_info['camera_pose'] = json.loads(pose_path.read_text())
            self.append(frame)
        for path in paths:
            last_rgb_path = path
            if self.debug:
                mask_path = path.with_name(path.name.replace('_normal.png', '_seg_inst.png'))
                if not mask_path.exists():
                    raise RuntimeError(f'Missing synchronized mask: {mask_path}')
                last_mask_path = mask_path
                if self.config.get('native_navigation') and self.config.get('viewpoint') != 'first_person':
                    with Image.open(mask_path) as mask_image:
                        issue = observer_frame_issue(np.asarray(mask_image.convert('RGB')), colors['1'])
                    if issue:
                        (self.output / 'frame_failure.json').write_text(json.dumps(dict(
                            issue=issue, video_frame=self.frames, action=line,
                            rgb=str(path.relative_to(self.output)),
                            mask=str(mask_path.relative_to(self.output))), indent=2))
                        raise RuntimeError(f'Unrepairable recorded frame: {issue} at frame {self.frames}')
                self.frame_info.update(rgb=str(path.relative_to(self.output)), mask=str(mask_path.relative_to(self.output)))
                pose_path = path.with_name(path.name.replace('_normal.png', '_camera.json'))
                if pose_path.exists():
                    self.frame_info['camera_pose'] = json.loads(pose_path.read_text())
                    seg_pose_path = path.with_name(path.name.replace('_normal.png', '_seg_camera.json'))
                    if self.config.get('camera_continuity'):
                        seg_pose = json.loads(seg_pose_path.read_text())
                        pose = self.frame_info['camera_pose']
                        same_position = np.allclose(pose['position'], seg_pose['position'], atol=1e-5)
                        q, sq = np.asarray(pose['rotation']), np.asarray(seg_pose['rotation'])
                        same_rotation = abs(float(np.dot(q, sq))) > 1 - 1e-5
                        if not same_position or not same_rotation or abs(pose['fov'] - seg_pose['fov']) > 1e-4:
                            raise RuntimeError('RGB and segmentation camera poses differ')
                        self.frame_info['camera_modalities_aligned'] = True
            with Image.open(path) as frame:
                self.append(np.array(frame.convert('RGB')))
        record = {'action': line, 'start': start, 'end': self.time, 'frames': len(paths), 'response': response,
                  'find_solution':find_solution}
        if paths:
            first_source=re.search(r'Action_(\d+)_',paths[0].name)
            last_source=re.search(r'Action_(\d+)_',paths[-1].name)
            if first_source and last_source:
                clock=self.output/'logs/native_frame_clock.jsonl'
                record['native_action_spans']=native_action_spans(
                    folder,int(first_source[1]),int(last_source[1]),start,self.fps,
                    frame_maps=native_frame_mapping(clock) if clock.exists() else None)
        record.update(wall_seconds=time.monotonic() - wall_start, render_wall_seconds=render_end - wall_start,
                      ingest_wall_seconds=time.monotonic() - render_end)
        if self.debug:
            self.manifest.flush()
            record['target_id'] = self.target['id'] if self.target else None
            if last_mask_path is not None:
                with Image.open(last_mask_path) as image:
                    last_mask = np.asarray(image.convert('RGB'))
                tracked=set(self.config.get('monitor_ids',[]))
                if self.target is not None:tracked.add(self.target['id'])
                record['last_visible_pixels']={str(key):visible_pixels(last_mask,colors[str(key)])
                                               for key in sorted(tracked) if str(key) in colors}
                self.last_rgb_path = last_rgb_path
                self.last_mask_path = last_mask_path
            trace = self.output / 'logs/camera_collision.csv'
            if not trace.exists():
                raise RuntimeError('Debug capture requires the patched camera-guard simulator')
        self.actions.append(record)
        (self.output / 'actions.json').write_text(json.dumps(self.actions, indent=2))
        return record

    def target_info(self, colors):
        if self.target is None:
            return {'target_id': None}
        color = colors[str(self.target['id'])]
        collisions = [int(k) for k, v in colors.items()
                      if int(k) != self.target['id'] and np.allclose(v, color, atol=1e-6)]
        monitored=[key for key in self.config.get('monitor_ids',[]) if str(key) in colors]
        return {'target_id': self.target['id'], 'target_name': self.target['class_name'],
                'color': color, 'color_collision_ids': collisions,
                'monitored_colors': {str(key): colors[str(key)] for key in monitored},
                'monitored_collisions': {str(key): [int(k) for k, v in colors.items()
                    if int(k) != key and np.allclose(v, colors[str(key)], atol=1e-6)]
                    for key in monitored}}

    def snapshot(self, name, target_id, use_last_frame=False):
        folder = self.output / 'evidence'
        folder.mkdir(exist_ok=True)
        width, height = (self.config['render'][key] for key in ('width', 'height'))
        graph, colors = self.client.graph(), self.client.colors()
        if use_last_frame:
            if self.last_rgb_path is None or self.last_mask_path is None:
                raise RuntimeError('No recorded frame is available for evidence')
            with Image.open(self.last_rgb_path) as image:
                rgb=np.asarray(image.convert('RGB'))
            with Image.open(self.last_mask_path) as image:
                seg=np.asarray(image.convert('RGB'))
            pose_path=self.last_rgb_path.with_name(self.last_rgb_path.name.replace('_normal.png','_camera.json'))
        else:
            rgb = self.client.image(self.camera_index, width=width, height=height)
            seg = self.client.image(self.camera_index, mode='seg_inst', width=width, height=height)
            pose_path = self.output / 'logs/camera_collision.csv.pose.json'
        pose = json.loads(pose_path.read_text()) if pose_path.exists() else None
        color = colors[str(target_id)]
        pixels = visible_pixels(seg, color)
        collisions = [int(key) for key, value in colors.items()
                      if int(key) != target_id and np.allclose(value, color, atol=1e-6)]
        Image.fromarray(rgb).save(folder / f'{name}.png')
        Image.fromarray(seg).save(folder / f'{name}.seg.png')
        (folder / f'{name}.json').write_text(json.dumps({'graph': graph, 'target_id': target_id,
                                                      'color': color, 'visible_pixels': pixels,
                                                      'color_collision_ids': collisions, 'instance_colors': colors}, indent=2))
        if getattr(self, 'debug', False):
            self.frame_info = dict(self.target_info(colors), action=name,
                                   rgb=f'evidence/{name}.png', mask=f'evidence/{name}.seg.png')
            if pose is not None:
                self.frame_info['camera_pose'] = pose
        self.append(rgb, max(1, round(self.config['evidence']['hold_seconds'] * self.fps)))
        return {'time': self.time, 'visible_pixels': pixels, 'color_collision_ids': collisions, 'graph': graph, 'image': str(folder / f'{name}.png')}

    def close(self):
        self.writer.close()
        if self.manifest:
            self.manifest.close()
