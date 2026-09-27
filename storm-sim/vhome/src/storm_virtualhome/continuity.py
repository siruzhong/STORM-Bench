"""Check camera continuity on the exact sequence of exported video frames."""
import json
from pathlib import Path

import numpy as np


def camera_metrics(rows, fps):
    if len(rows) < 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Continuity checking needs at least two frames and a positive fps')
    if any('camera_pose' not in row for row in rows):
        raise ValueError('A video frame has no synchronized camera pose')
    positions = np.asarray([r['camera_pose']['position'] for r in rows], dtype=float)
    rotations = np.asarray([r['camera_pose']['rotation'] for r in rows], dtype=float)
    fov = np.asarray([r['camera_pose']['fov'] for r in rows], dtype=float)
    if positions.shape != (len(rows), 3) or rotations.shape != (len(rows), 4):
        raise ValueError('Invalid camera pose dimensions')
    norms = np.linalg.norm(rotations, axis=1)
    if not all(np.isfinite(x).all() for x in (positions, rotations, fov)) or np.any(np.abs(norms - 1) > 0.001):
        raise ValueError('Invalid camera pose values')
    rotations /= norms[:, None]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    dots = np.abs(np.sum(rotations[1:] * rotations[:-1], axis=1))
    turns = np.degrees(2 * np.arccos(np.clip(dots, 0, 1)))
    boundaries = [i for i in range(1, len(rows)) if rows[i].get('action') != rows[i - 1].get('action')]
    linear_acceleration = np.linalg.norm(np.diff(positions, n=2, axis=0), axis=1) * fps ** 2
    previous, following = rotations[:-1], rotations[1:]
    inverse_vector = -previous[:, :3]
    delta_vector = (following[:, 3:] * inverse_vector + previous[:, 3:] * following[:, :3]
                    + np.cross(following[:, :3], inverse_vector))
    delta_scalar = following[:, 3] * previous[:, 3] - np.sum(following[:, :3] * inverse_vector, axis=1)
    delta_vector *= np.where(delta_scalar < 0, -1.0, 1.0)[:, None]
    lengths = np.linalg.norm(delta_vector, axis=1)
    angles = np.degrees(2 * np.arctan2(lengths, np.abs(delta_scalar)))
    angular_velocity = delta_vector / np.maximum(lengths[:, None], 1e-12) * angles[:, None] * fps
    angular_acceleration = np.linalg.norm(np.diff(angular_velocity, axis=0), axis=1) * fps
    return {
        'frames': len(rows), 'fps': fps,
        'position_min': positions.min(axis=0).tolist(),
        'position_max': positions.max(axis=0).tolist(),
        'max_translation_step_m': float(steps.max()),
        'max_camera_speed_mps': float(steps.max() * fps),
        'p95_camera_speed_mps': float(np.percentile(steps * fps, 95)),
        'p95_linear_acceleration_mps2': float(np.percentile(linear_acceleration, 95)) if len(linear_acceleration) else 0.0,
        'max_rotation_step_degrees': float(turns.max()),
        'max_camera_turn_rate_dps': float(turns.max() * fps),
        'p95_camera_turn_rate_dps': float(np.percentile(turns * fps, 95)),
        'p95_angular_acceleration_dps2': float(np.percentile(angular_acceleration, 95)) if len(angular_acceleration) else 0.0,
        'max_fov_step_degrees': float(np.abs(np.diff(fov)).max()),
        'fov_degrees': float(fov[0]),
        'action_boundaries_checked': len(boundaries),
        'max_boundary_translation_m': max((float(steps[i - 1]) for i in boundaries), default=0),
        'max_boundary_rotation_degrees': max((float(turns[i - 1]) for i in boundaries), default=0),
    }


def check_limits(report, rules):
    if 'position_min' in rules and np.any(np.asarray(report['position_min']) < np.asarray(rules['position_min']) - 0.0001):
        raise ValueError('Camera position leaves the configured safe region')
    if 'position_max' in rules and np.any(np.asarray(report['position_max']) > np.asarray(rules['position_max']) + 0.0001):
        raise ValueError('Camera position leaves the configured safe region')
    if 'fov_degrees' in rules and abs(report['fov_degrees'] - rules['fov_degrees']) > 0.0001:
        raise ValueError('The actual camera field of view does not match the configured lens')
    if report['max_camera_speed_mps'] > rules['max_speed_mps'] + 0.002:
        raise ValueError('Camera translation exceeds the continuity limit')
    if report['max_camera_turn_rate_dps'] > rules['max_turn_rate_dps'] + 0.05:
        raise ValueError('Camera rotation exceeds the continuity limit')
    if report['p95_linear_acceleration_mps2'] > rules.get('max_p95_linear_acceleration_mps2', float('inf')):
        raise ValueError('Camera translation has an acceleration spike')
    if report['p95_angular_acceleration_dps2'] > rules.get('max_p95_angular_acceleration_dps2', float('inf')):
        raise ValueError('Camera rotation has an acceleration spike')
    if report['max_boundary_rotation_degrees'] > rules.get('max_boundary_rotation_degrees', float('inf')):
        raise ValueError('Camera direction jumps at an action boundary')
    if report['max_fov_step_degrees'] > 0.0001:
        raise ValueError('Camera focal length changed inside the episode')


def audit_continuity(root):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    rules = config.get('camera_continuity')
    if not rules:
        return {}
    rows = [json.loads(line) for line in (root / 'frame_manifest.jsonl').open()]
    report = camera_metrics(rows, config['render']['fps'])
    report['scope'] = 'camera_pose_continuity'
    report['limits'] = rules
    report['accepted'] = False
    output = root / 'camera_continuity.json'
    output.write_text(json.dumps(report, indent=2))
    check_limits(report, rules)
    report['accepted'] = True
    output.write_text(json.dumps(report, indent=2))
    return report
