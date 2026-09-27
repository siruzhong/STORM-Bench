"""Build questions from the same fixed-rate frames consumed by the evaluator."""
from collections import defaultdict
from copy import deepcopy
import math

from .aligned_qa import build_pool, observed_state

SCHEMA = 'sampled_observation_qa_v2'


def align_observations(episode, visibility, *, evaluation_fps=1.0, minimum_pixels=40,
                       excluded_objects=()):
    """Replace stage timestamps with visible stable states on the model clock."""
    result = deepcopy(episode)
    source_fps = float(visibility['fps'])
    frame_count = visibility['frames']
    last_time = (frame_count - 1) / source_fps
    clock = [(round(i * source_fps / evaluation_fps), i / evaluation_fps)
             for i in range(math.floor(last_time * evaluation_fps) + 1)]
    excluded = set(excluded_objects)
    source = episode['observations']
    result['observations'] = []
    records = {}
    samples = defaultdict(list)
    for original in source:
        obj = original['object_id']
        if original['state'] is None or obj in excluded or original['collisions']:
            continue
        col = visibility['objects'].index(obj)
        threshold = max(minimum_pixels, original.get('visibility_threshold', 40))
        lo, hi = original['state_interval']
        visible = [(frame, time) for frame, time in clock if lo <= time < hi
                   and visibility['counts'][frame][col] >= threshold]
        for frame, time in visible:
            observation = deepcopy(original)
            observation.update(frame=frame, time=(frame + 1) / source_fps,
                               pixels=visibility['counts'][frame][col],
                               visibility_threshold=threshold,
                               model_time=time, side='sample',
                               key=f'sample_{obj}_{frame}')
            samples[obj, frame].append(observation)
        if not visible:
            continue
        frame, time = visible[-1] if original['side'] == 'before' else visible[0]
        record = deepcopy(original)
        record.update(frame=frame, time=(frame + 1) / source_fps,
                      pixels=visibility['counts'][frame][col], model_time=time,
                      visibility_threshold=threshold)
        records[original['event_index'], original['side']] = record

    conflicts = []
    for key, candidates in samples.items():
        if len({observed_state(o) for o in candidates}) != 1:
            conflicts.append(list(key))
            continue
        result['observations'].append(candidates[0])
    if conflicts:
        raise ValueError(f'Conflicting graph states on sampled frames: {conflicts[:5]}')
    result['observations'].extend(records.values())
    for event in result['events']:
        before = records.get((event['event_index'], 'before'))
        after = records.get((event['event_index'], 'after'))
        event['before_time'] = before['time'] if before else event['before_time']
        event['discovery_time'] = after['time'] if after else last_time + 10
    reveals = {e['discovery_time'] for e in result['events'] if e['discovery_time'] <= last_time + 1 / source_fps}
    regular = {(frame + 1) / source_fps for frame, time in clock if round(time * evaluation_fps) % 5 == 0}
    result['query_times'] = sorted(reveals | regular)
    result['sampling_contract'] = dict(evaluation_fps=evaluation_fps, source_fps=source_fps,
                                       minimum_pixels=minimum_pixels,
                                       excluded_objects=sorted(excluded),
                                       state_review='Instance visibility only; RGB state review is separate.')
    return result


def build_sampled_pool(episode, visibility, **kwargs):
    """Return candidates whose positive evidence is present on the model clock."""
    aligned = align_observations(episode, visibility, **kwargs)
    rows = build_pool(aligned, schema=SCHEMA)
    for q in rows:
        q['sampling_contract'] = aligned['sampling_contract']
        q['evidence_review'] = 'Graph state and visible model-clock frames; RGB review pending.'
        q['provenance']['sampling_contract'] = aligned['sampling_contract']
    return rows, aligned
