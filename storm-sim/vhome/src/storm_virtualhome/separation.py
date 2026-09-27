"""Audit actor separation in recorded frames without filling missing observations."""
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
import re


def align_clock(rows):
    """Keep recorder epochs separate when adding an actor resets its frame counter."""
    observer = {row['unity_frame']: row['source_frame'] for row in rows if row['actor'] == 0}
    last, epochs = {}, defaultdict(int)
    aligned = defaultdict(dict)
    for row in rows:
        actor, source = row['actor'], row['source_frame']
        if source < last.get(actor, -1):
            epochs[actor] += 1
        last[actor] = source
        shared = observer.get(row['unity_frame'])
        if shared is not None:
            aligned[shared][actor] = (epochs[actor], source)
    return dict(aligned)


def read_hips(path):
    poses = {}
    with path.open() as stream:
        header = next(stream).split()
        offset = 1 + 3 * header.index('Hips')
        for line in stream:
            parts = line.split()
            values = tuple(map(float, parts[offset:offset + 3]))
            if len(values) != 3 or not all(map(math.isfinite, values)):
                raise ValueError(f'Invalid skeleton coordinate in {path}')
            poses[int(parts[0])] = values
    return poses


def audit_separation(root):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    threshold = config['motion_contract']['rules']['agent_clearance_m']
    clock = align_clock([json.loads(line) for line in
                         (root / 'logs/native_frame_clock.jsonl').read_text().splitlines()])
    frames = [json.loads(line) for line in (root / 'frame_manifest.jsonl').read_text().splitlines()]
    exact = all('actor_positions' in row.get('camera_pose', {}) for row in frames)
    groups = defaultdict(list)
    for row in frames:
        match = re.search(r'Action_(\d+)_0_normal\.png$', row['rgb'])
        if match and not exact:
            folder = (root / row['rgb']).parent.parent
            groups[folder].append((row['frame'], int(match[1])))
    # Each action writes cumulative poses. Keep the last file for each actor epoch.
    sources = {}
    actors_by_folder = {}
    for folder, samples in sorted(groups.items()):
        actors = {}
        for path in folder.glob('*/pd_*.txt'):
            actor = int(path.parent.name)
            observed = [source for _, source in samples if actor in clock.get(source, {})]
            if not observed:
                actors[actor] = None
                continue
            epoch = clock[max(observed)][actor][0]
            actors[actor] = epoch
            sources[actor, epoch] = path
        actors_by_folder[folder] = actors
    poses = {key: read_hips(path) for key, path in sources.items()}
    pairs = 0
    covered_frames = 0
    missing = []
    violations = []
    minima = {}
    for row in frames:
        match = re.search(r'Action_(\d+)_0_normal\.png$', row['rgb'])
        positions = {}
        if exact:
            positions = {int(actor): point for actor, point in row['camera_pose']['actor_positions'].items()}
            expected = {int(node) - 1 for node in row['monitored_colors']}
            for actor in expected - positions.keys():
                missing.append(dict(frame=row['frame'], actor=actor, reason='root_pose'))
            if 0 not in positions:
                missing.append(dict(frame=row['frame'], actor=0, reason='observer_root_pose'))
            basis = 'synchronized_root'
        elif match:
            source = int(match[1])
            folder = (root / row['rgb']).parent.parent
            expected = actors_by_folder[folder]
            for actor in expected:
                mapped = clock.get(source, {}).get(actor)
                if mapped is None or mapped[0] != expected[actor]:
                    missing.append(dict(frame=row['frame'], actor=actor, reason='clock_mapping'))
                    continue
                point = poses.get((actor, mapped[0]), {}).get(mapped[1])
                if point is None:
                    missing.append(dict(frame=row['frame'], actor=actor, reason='skeleton_pose'))
                    continue
                positions[actor] = point
            basis = 'animated_hips'
        else:
            evidence = json.loads((root / row['rgb']).with_suffix('.json').read_text())
            positions = {node['id'] - 1: node['obj_transform']['position']
                         for node in evidence['graph']['nodes'] if node['class_name'] == 'character'}
            expected = positions
            basis = 'snapshot_root'
        if len(positions) == len(expected) and len(positions) >= 2:
            covered_frames += 1
        for first, second in combinations(sorted(positions), 2):
            a, b = positions[first], positions[second]
            distance = math.hypot(a[0] - b[0], a[2] - b[2])
            if not math.isfinite(distance):
                raise ValueError('Nonfinite actor separation')
            pairs += 1
            minima[basis] = min(distance, minima.get(basis, float('inf')))
            if distance < threshold:
                violations.append(dict(frame=row['frame'], actors=[first, second],
                                       distance_m=distance, basis=basis))
    return dict(schema_version=1, capture=str(root.resolve()),
                status=('clearance_violation' if violations else
                        'incomplete_evidence' if missing or covered_frames != len(frames)
                        else 'separation_verified'),
                frames=len(frames), covered_frames=covered_frames, checked_pairs=pairs,
                minimum_distance_m=minima, threshold_m=threshold,
                missing_samples=missing, violations=violations,
                exact_root_evidence=exact,
                scope='Planar synchronized roots when recorded; otherwise animated hips and snapshot roots. '
                      'This is not a mesh-contact or swept-volume collision proof.')
