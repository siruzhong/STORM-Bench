"""Reframe recorded observations as four-choice questions without model feedback."""
from collections import Counter, defaultdict
from itertools import combinations, product
import hashlib
import json
import math
import random
import re

SCHEMA = 'ai2thor_aligned_observations_v1'
TERMS = {'lightswitch': 'light switch', 'tablelamp': 'table lamp',
         'facecream': 'face cream', 'barsoap': 'bar of soap',
         'poundcake': 'pound cake', 'coffeetable': 'coffee table',
         'mousemat': 'mouse pad', 'kitchencabinet': 'kitchen cabinet'}


def name(text):
    for old, new in TERMS.items():
        text = re.sub(r'\b' + old + r'\b', new, text)
    return 'toilet lid' if text == 'toilet' else text


def domain(event):
    if event['verb'] in ('Open', 'Close'):
        return ['closed', 'open']
    if event['verb'] in ('SwitchOn', 'SwitchOff'):
        return ['off', 'on']
    return ['resting on a surface', 'held by a person']


def observed_state(record):
    state = record['state']
    if state and state.startswith('on:'):
        return 'resting on a surface'
    return state


def apportion(reference_rows, total):
    """Use largest remainders with a stable alphabetical tie break."""
    counts = Counter(q['question_type'] for q in reference_rows)
    exact = {key: value * total / len(reference_rows) for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - result[k]), k))[:total-sum(result.values())]:
        result[key] += 1
    return result


def build_pool(episode, *, schema=SCHEMA):
    """Ground each answer in graph-checked, visible frames from the same prefix."""
    meta = episode['metadata']
    events = episode['events']
    by_index = {e['event_index']: e for e in events}
    by_object = {e['object_id']: e for e in events}
    names = {i: name(e['object_description']) for i, e in by_object.items()}
    domains = {i: domain(e) for i, e in by_object.items()}
    assert len(set(names.values())) == len(names)
    fps = episode['qa']['sample_fps']
    observations = [o for o in episode['observations'] if observed_state(o) in domains[o['object_id']]
                    and o['pixels'] >= o.get('visibility_threshold', 40) and not o['collisions']]
    records = {(o['event_index'], o['side']): o for o in observations}
    changes = []
    for event in events:
        before = records.get((event['event_index'], 'before'))
        after = records.get((event['event_index'], 'after'))
        if before and after and observed_state(before) != observed_state(after):
            changes.append(dict(event=event, before=before, after=after,
                                time=after['time'], object_id=event['object_id']))
    changes.sort(key=lambda c: c['time'])
    questions = []

    def add(kind, subtype, question, options, answer, query, evidence, semantic, anchors):
        assert len(options) == len(set(options)) == 4 and answer in options
        frame_count = int(query * fps + 1e-6)
        if not evidence or any(o['frame'] >= frame_count for o in evidence):
            return
        if any(o['time'] > query + 1e-6 for o in evidence):
            return
        evidence = sorted({o['key']: o for o in evidence}.values(), key=lambda o: o['frame'])
        identity = json.dumps([schema, meta['episode_id'], kind, subtype, query, anchors], sort_keys=True)
        row_id = meta['episode_id'] + '_a' + hashlib.sha256(identity.encode()).hexdigest()[:12]
        shuffled = list(options)
        random.Random(row_id).shuffle(shuffled)
        spans = [[o['frame']/fps, (o['frame']+1)/fps] for o in evidence]
        questions.append(dict(id=row_id, episode_id=meta['episode_id'],
            event_index=max(o['event_index'] for o in evidence),
            source_event_indices=sorted({o['event_index'] for o in evidence}),
            question_type=kind, question_subtype=subtype, question=question,
            options=shuffled, answer_index=shuffled.index(answer), num_choices=4,
            chance_accuracy=.25, query_time=query, evidence_spans=spans,
            diagnostics=dict(epistemic_status='known', uncertainty_sources=[]),
            change_intensity='high', qa_schema=schema, semantic_answer=semantic,
            semantic_anchor=anchors, generation_key=identity,
            source_video_path=meta['episode_path']+'/video.mp4', requires_query_prefix=True,
            video_evidence='Use the recorded observations within this video prefix.',
            evidence_review='graph_state_and_visible_stage_frames; perceptual review separate',
            polish_status='rule_reframed',
            provenance=dict(events_sha256=episode['events_sha256'],
                observations=[{k: o[k] for k in ('key', 'object_id', 'frame', 'time', 'state',
                                               'pixels', 'source_json', 'source_sha256',
                                               'original_stage_time', 'original_stage_frame',
                                               'state_interval', 'frame_manifest_sha256')} for o in evidence],
                answer_derivation=anchors)))

    def latest(cutoff):
        result = {}
        for o in sorted(observations, key=lambda x: x['frame']):
            if o['time'] <= cutoff + 1e-6:
                result[o['object_id']] = o
        return result

    times = sorted({t for t in episode.get('query_times', [e['discovery_time'] for e in events])
                    if not any(other['start'] <= t < other['end']
                               for other in events)})
    for query in times:
        present = latest(query)
        confirmed = [c for c in changes if c['time'] <= query]
        for left, right in combinations(sorted(present), 2):
            labels = [f'{names[left].capitalize()}: {a}; {names[right]}: {b}.'
                      for a, b in product(domains[left], domains[right])]
            states = [(a, b) for a, b in product(domains[left], domains[right])]
            current = (observed_state(present[left]), observed_state(present[right]))
            # A current-state item excludes hidden changes after either last sighting.
            pending = any(e['object_id'] in (left, right) and
                          present[e['object_id']]['time'] < e['end'] <= query and
                          e['discovery_time'] > query for e in events)
            if not pending:
                add('current_state', 'joint_last_observed_state',
                    f'What were the last visible states of the {names[left]} and the {names[right]} by the end of the clip?',
                    labels, labels[states.index(current)], query, [present[left], present[right]],
                    states.index(current), dict(objects=[left, right], states=list(current), cutoff=query))
            earlier = [t for t in times if t < query - 5]
            for cutoff in earlier:
                past = latest(cutoff)
                if left not in past or right not in past:
                    continue
                old = (observed_state(past[left]), observed_state(past[right]))
                # Require a subsequent observation so this is a memory question.
                if all(past[x]['key'] == present[x]['key'] for x in (left, right)):
                    continue
                add('factual_retrieval', 'joint_earlier_observation',
                    f'At {cutoff:.2f} seconds, what were the most recently seen states of the {names[left]} and the {names[right]}?',
                    labels, labels[states.index(old)], query, [past[left], past[right]],
                    states.index(old), dict(objects=[left, right], states=list(old), cutoff=cutoff))
            pair = [c for c in confirmed if c['object_id'] in (left, right)]
            missing_recent = (len(pair) >= 2 and any(
                e['object_id'] in (left, right) and pair[-2]['time'] <= e['discovery_time'] <= query
                and e['event_index'] not in {c['event']['event_index'] for c in pair}
                for e in events))
            if len(pair) >= 2 and pair[-1]['time'] != pair[-2]['time'] and not missing_recent:
                ordered = list(product((left, right), repeat=2))
                options = [f'{names[a].capitalize()}, then the {names[b]}.' for a, b in ordered]
                answer = (pair[-2]['object_id'], pair[-1]['object_id'])
                add('temporal_reasoning', 'last_two_confirmed_changes',
                    f'For the {names[left]} and the {names[right]}, which pair gives the objects involved in the last two confirmed changes, in order?',
                    options, options[ordered.index(answer)], query,
                    [o for c in pair for o in (c['before'], c['after'])], ordered.index(answer),
                    dict(objects=[left, right], ordered_events=[c['event']['event_index'] for c in pair[-2:]],
                         confirmation_times=[c['time'] for c in pair[-2:]]))
        if confirmed:
            recent = confirmed[-1]
            # The query must reveal this change rather than merely extend an old prefix.
            if recent['time'] == query and sum(c['time'] == query for c in confirmed) == 1:
                target = recent['object_id']
                for other in sorted(present):
                    if other == target:
                        continue
                    pair = sorted((target, other))
                    choices, identities = [], []
                    descriptions = []
                    for obj in pair:
                        for state in domains[obj]:
                            old = next(s for s in domains[obj] if s != state)
                            if state == 'held by a person':
                                change = 'was picked up'
                            elif state == 'resting on a surface':
                                change = 'was put down'
                            else:
                                change = f'changed from {old} to {state}'
                            choices.append(f'The {names[obj]} {change}.')
                            descriptions.append(f'{names[obj].capitalize()}, {state}.')
                            identities.append((obj, state))
                    identity = (target, observed_state(recent['after']))
                    index = identities.index(identity)
                    add('state_change', 'latest_confirmed_transition',
                        'Which description matches the most recently confirmed change in the clip?',
                        choices, choices[index], query,
                        [o for c in confirmed for o in (c['before'], c['after'])], index,
                        dict(objects=pair, event=recent['event']['event_index'],
                             before=observed_state(recent['before']), after=identity[1]))
                    if recent['event']['visibility'] == 'offscreen':
                        add('object_tracking', 'reidentified_after_offscreen_change',
                            'Which object was most recently seen again after changing out of view, and what state was it in when it reappeared?',
                            descriptions, descriptions[index], query,
                            [o for c in confirmed for o in (c['before'], c['after'])], index,
                            dict(objects=pair, event=recent['event']['event_index'], after=identity[1],
                                 hidden_interval=[recent['event']['start'], recent['event']['end']]))
        for obj in sorted(present):
            for start in [0.0] + [t for t in times if t < query - 6]:
                relevant = [c for c in confirmed if c['object_id'] == obj and start < c['time'] <= query]
                potential = [e for e in events if e['object_id'] == obj and start < e['discovery_time'] <= query]
                # Missing graph states cannot silently become zero-count evidence.
                if len(potential) != len(relevant):
                    continue
                count = len(relevant)
                if query - start < 6:
                    continue
                options = ['No confirmed changes.', 'One confirmed change.',
                           'Two confirmed changes.', 'Three or more confirmed changes.']
                evidence = [o for o in observations if o['object_id'] == obj and o['time'] <= query]
                if not evidence:
                    continue
                window = 'Over the clip' if start == 0 else f'After {start:.2f} seconds'
                add('history_aggregation', 'confirmed_change_count_in_window',
                    f'{window}, how many changes to the {names[obj]} were confirmed by seeing its new state?',
                    options, options[min(count, 3)], query, evidence, min(count, 3),
                    dict(object=obj, start=start, end=query, count=count,
                         events=[c['event']['event_index'] for c in relevant]))
    return list({q['id']: q for q in questions}.values())


def select_batch(pools, quotas, seed=20260921):
    """Allocate by annotation only, enforcing room, type, and semantic quotas."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    rows = [q for pool in pools for q in pool]
    groups = defaultdict(list)
    for i, q in enumerate(rows):
        groups['episode', q['episode_id']].append(i)
        groups['type', q['question_type']].append(i)
        groups['type_semantic', q['question_type'], q['semantic_answer']].append(i)
        groups['episode_type', q['episode_id'], q['question_type']].append(i)
        groups['request', q['episode_id'], q['question'], q['query_time']].append(i)
        groups['boundary', q['episode_id'], q['query_time']].append(i)
        anchor = dict(q['semantic_anchor'])
        if q['question_type'] == 'factual_retrieval':
            fact = (q['question'],)
        elif q['question_type'] == 'temporal_reasoning':
            fact = (tuple(anchor['objects']), tuple(anchor['ordered_events']))
        elif q['question_type'] == 'history_aggregation':
            fact = (anchor['object'], anchor['start'], tuple(anchor['events']))
        else:
            fact = tuple(o['key'] for o in q['provenance']['observations'])
        groups['fact', q['episode_id'], q['question_type'], fact].append(i)
    constraints = []
    for key, indices in groups.items():
        if key[0] == 'episode':
            limits = (10, 10)
        elif key[0] == 'type':
            limits = (quotas[key[1]], quotas[key[1]])
        elif key[0] == 'type_semantic':
            limits = (quotas[key[1]] // 4, math.ceil(quotas[key[1]] / 4))
        elif key[0] == 'episode_type':
            limits = (1, 2)
        elif key[0] in ('request', 'fact'):
            limits = (0, 1)
        else:
            limits = (0, 3)
        constraints.append((indices, *limits))
    rr, cc, values = [], [], []
    for r, (indices, _, _) in enumerate(constraints):
        rr.extend([r] * len(indices)); cc.extend(indices); values.extend([1.] * len(indices))
    matrix = coo_matrix((values, (rr, cc)), shape=(len(constraints), len(rows))).tocsc()
    objective = np.array([random.Random(str(seed)+q['id']).random() for q in rows])
    # Prefer larger visible targets without making visibility a hard semantic filter.
    objective += np.array([.5 if min(o['pixels'] for o in q['provenance']['observations']) < 64 else 0 for q in rows])
    result = milp(objective, integrality=np.ones(len(rows)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, [x[1] for x in constraints], [x[2] for x in constraints]),
        options=dict(time_limit=90, mip_rel_gap=.01))
    if result.x is None:
        raise ValueError(f'No valid annotation allocation: {result.message}')
    selected = [q for i, q in enumerate(rows) if result.x[i] > .5]
    assert len(selected) == sum(quotas.values())
    assert Counter(q['question_type'] for q in selected) == quotas
    assert set(Counter(q['episode_id'] for q in selected).values()) == {10}
    for kind, count in quotas.items():
        balance = Counter(q['semantic_answer'] for q in selected if q['question_type'] == kind)
        assert set(balance) == {0, 1, 2, 3}
        assert max(balance.values()) - min(balance.values()) <= 1
    return sorted(selected, key=lambda q: (q['episode_id'], q['query_time'], q['id']))
