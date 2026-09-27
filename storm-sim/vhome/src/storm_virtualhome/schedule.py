"""Check transition counts and spacing in recorded video time."""
import math


def transition_schedule(pairs, config, complete=True):
    transitions = []
    for index, pair in enumerate(pairs):
        for kind in ('disappear', 'appear'):
            item = pair[kind]
            transitions.append({'event_index': len(transitions), 'cycle_index': index,
                                'type': kind, 'object_id': pair['object_id'],
                                'start': item['start'], 'end': item['end']})
    expected = config.get('event_count', 2)
    if len(transitions) > expected or (complete and len(transitions) != expected):
        raise ValueError(f'Expected {expected} transitions, recorded {len(transitions)}')
    gaps = [b['start'] - a['start'] for a, b in zip(transitions, transitions[1:])]
    for item in transitions:
        if not all(math.isfinite(item[k]) for k in ('start', 'end')) or not 0 <= item['start'] < item['end']:
            raise ValueError('Event intervals must be finite and nonempty')
    if any(b['start'] < a['end'] for a, b in zip(transitions, transitions[1:])):
        raise ValueError('Injected event intervals overlap or are out of order')
    target = config.get('event_interval_seconds')
    tolerance = config.get('event_interval_tolerance_seconds', 0)
    if target is not None:
        if target <= 0 or tolerance < 0 or tolerance >= target:
            raise ValueError('Invalid event spacing configuration')
        if any(abs(gap - target) > tolerance + 1e-6 for gap in gaps):
            raise ValueError(f'Event start gaps fall outside {target} +/- {tolerance} seconds: {gaps}')
    return {'event_count': len(transitions), 'expected_event_count': expected, 'complete': complete,
            'event_pairs': len(pairs),
            'counting_unit': 'one disappearance or reappearance',
            'timing_reference': 'recorded action sequence start, not exact physical transition',
            'target_interval_seconds': target, 'tolerance_seconds': tolerance,
            'start_gaps_seconds': gaps, 'events': transitions}


def check_duration(duration, config):
    bounds = config.get('duration_range_seconds')
    if bounds is None:
        return
    if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 < bounds[0] <= bounds[1]:
        raise ValueError('Invalid duration range')
    if not math.isfinite(duration) or not bounds[0] <= duration <= bounds[1]:
        raise ValueError(f'Video duration {duration:.2f}s is outside {bounds}')
