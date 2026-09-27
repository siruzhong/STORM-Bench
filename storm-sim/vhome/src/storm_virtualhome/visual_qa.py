"""Generate visible-event and epistemic questions on the evaluation clock."""
from collections import Counter, defaultdict
from itertools import combinations, product
import hashlib
import json
import math
import random

from .aligned_qa import name

SCHEMA = 'virtualhome_visual_epistemic_v1'
UNDETERMINED = 'The available views do not establish an answer.'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_visual_pool(episode, visibility, *, minimum_pixels=128,
                      uncertainty_policy='reference'):
    """Use masks for visibility facts and hidden intervals for uncertainty facts."""
    if uncertainty_policy not in ('reference', 'all_questions'):
        raise ValueError(f'Unknown uncertainty option policy: {uncertainty_policy}')
    meta = episode['metadata']
    fps = float(visibility['fps'])
    duration = float(meta['duration_seconds'])
    objects = {e['object_id']: e for e in episode['events']}
    names = {obj: name(e['object_description']) for obj, e in objects.items()}
    assert len(set(names.values())) == len(names)
    times = list(range(math.floor((visibility['frames'] - 1) / fps) + 1))
    pixels = {obj: [visibility['counts'][round(t * fps)][visibility['objects'].index(obj)]
                   for t in times] for obj in objects}
    states = {obj: ['visible' if n >= minimum_pixels else 'absent' if n == 0 else 'ambiguous'
                    for n in values] for obj, values in pixels.items()}
    stable = {obj: {t: states[obj][t] for t in times[1:]
                    if states[obj][t] == states[obj][t-1] != 'ambiguous'} for obj in objects}
    transitions, returns = [], []
    for obj in objects:
        previous, witness, seen_before, last_visible = None, None, False, None
        for t, state in stable[obj].items():
            if previous is not None and state != previous:
                event = dict(object_id=obj, time=t, before=witness, after=t, previous_visible=last_visible,
                             transition='returned' if state == 'visible' else 'left')
                transitions.append(event)
                if state == 'visible' and seen_before:
                    returns.append(event)
            if state == 'visible':
                seen_before = True
                last_visible = t
            previous, witness = state, t
    transitions.sort(key=lambda e: (e['time'], e['object_id']))
    returns.sort(key=lambda e: (e['time'], e['object_id']))
    rows = []

    def observation(obj, t):
        return dict(object_id=obj, object_name=names[obj], time=float(t), frame=round(t*fps),
                    pixels=pixels[obj][t], visibility=states[obj][t])

    def add(kind, subtype, query, question, concrete, correct, evidence, anchor,
            status='known', sources=(), rationale='The indexed frames establish the selected alternative.'):
        query = round(query + 1/fps, 6)
        if not 0 < query < duration or not evidence:
            return
        assert len(set(concrete)) >= 3
        key = [SCHEMA, meta['episode_id'], kind, subtype, query, anchor]
        qid = meta['episode_id'] + '_vqa_' + _digest(key)[:12]
        rng = random.Random(qid)
        if status == 'known':
            assert correct in concrete
            distractors = sorted(set(concrete) - {correct})
            rng.shuffle(distractors)
            if uncertainty_policy == 'reference':
                if len(distractors) < 3:
                    raise ValueError(f'Known question lacks four concrete alternatives: {qid}')
                options = [correct, *distractors[:3]]
            else:
                options = [correct, *distractors[:2], UNDETERMINED]
        else:
            assert correct == UNDETERMINED and sources
            options = list(dict.fromkeys(concrete))[:3] + [UNDETERMINED]
        rng.shuffle(options)
        spans = [(min(o['time'] for o in evidence), round(max(o['time'] for o in evidence) + 1/fps, 6))]
        assert all(0 <= start < end <= query + 1e-6 for start, end in spans)
        rows.append(dict(id=qid, episode_id=meta['episode_id'], query_time=query,
            question_type=kind, question_subtype=subtype, question=question,
            options=options, answer_index=options.index(correct), evidence_spans=spans,
            video_evidence='Use sampled video observations no later than query_time.',
            diagnostics=dict(epistemic_status=status, uncertainty_sources=list(sources)),
            diagnostic_rationale=dict(volatility='Visibility events change the available object evidence.', uncertainty=rationale),
            change_intensity=max(1, sum(float(e['start']) <= query + 1e-6 for e in episode['events'])),
            qa_schema=SCHEMA, semantic_anchor=anchor,
            provenance=dict(events_sha256=episode['events_sha256'],
                frame_manifest_sha256=visibility['manifest_sha256'], observations=evidence,
                sampling_fps=1, duration_seconds=duration, minimum_target_pixels=minimum_pixels,
                answer_basis='sampled_visibility' if status == 'known' else 'unobserved_physical_facts'),
            review_status='requires_rgb_review' if status == 'known' else 'hidden_interval_verified'))

    def pair_options(left, right):
        values = list(product(('visible', 'out of view'), repeat=2))
        options = [f'{names[left].capitalize()}: {a}; {names[right]}: {b}.' for a, b in values]
        return values, options

    seen_at = {obj: min((t for t in stable[obj] if stable[obj][t] == 'visible'), default=math.inf) for obj in objects}
    queries = sorted({e['time'] for e in transitions} | {t for t in times if t >= 5 and t % 5 == 0})
    # Room-category questions match the reference's current and retrieval tasks.
    room_options = ['bathroom', 'bedroom', 'kitchen', 'living room']
    room = 'living room' if meta['room'] == 'livingroom' else meta['room']
    for kind, question in [('current_state', 'What type of room is visible around the observer?'),
                           ('factual_retrieval', 'What type of room was shown at the start of the video?')]:
        for query in queries:
            evidence = [dict(object_id=None, object_name='room', time=float(t), frame=round(t*fps),
                             pixels=0, visibility='room_context') for t in ([query] if kind == 'current_state' else [0, 1])]
            add(kind, 'visible_room_category', query, question, room_options, room, evidence,
                dict(room=room, observation_time=query if kind == 'current_state' else 0))
    for query in queries:
        for left, right in combinations(sorted(objects), 2):
            if max(seen_at[left], seen_at[right]) > query:
                continue
            values, options = pair_options(left, right)
            if query in stable[left] and query in stable[right]:
                current = tuple('visible' if stable[o][query] == 'visible' else 'out of view' for o in (left, right))
                evidence = [observation(o, t) for o in (left, right) for t in (query-1, query)]
                add('current_state', 'current_visibility_pair', query,
                    f'At the question point, which description matches the visibility of the {names[left]} and the {names[right]}?',
                    options, options[values.index(current)], evidence,
                    dict(objects=[left, right], states=current, observation_time=query))
            for earlier in queries:
                if earlier > query - 6 or max(seen_at[left], seen_at[right]) > earlier:
                    continue
                if earlier not in stable[left] or earlier not in stable[right]:
                    continue
                old = tuple('visible' if stable[o][earlier] == 'visible' else 'out of view' for o in (left, right))
                if query in stable[left] and query in stable[right] and all(stable[o][earlier] == stable[o][query] for o in (left, right)):
                    continue
                add('factual_retrieval', 'earlier_visibility_pair', query,
                    f'At {earlier:.0f} seconds, which description matched the visibility of the {names[left]} and the {names[right]}?',
                    options, options[values.index(old)], [observation(o, t) for o in (left, right) for t in (earlier-1, earlier)],
                    dict(objects=[left, right], states=old, observation_time=earlier))
            pair_returns = [e for e in returns if e['object_id'] in (left, right) and e['time'] <= query]
            if len(pair_returns) >= 2 and pair_returns[-1]['time'] != pair_returns[-2]['time']:
                ordered = list(product((left, right), repeat=2))
                labels = [f'{names[a].capitalize()}, then the {names[b]}.' for a, b in ordered]
                last = pair_returns[-2:]
                correct = tuple(e['object_id'] for e in last)
                evidence = [observation(e['object_id'], t) for e in last for t in (e['previous_visible'], e['before'], e['after']-1, e['after'])]
                add('temporal_reasoning', 'last_two_observed_returns', query,
                    f'For the {names[left]} and the {names[right]}, which order describes their last two returns to view?',
                    labels, labels[ordered.index(correct)], evidence,
                    dict(objects=[left, right], return_times=[e['time'] for e in last], order=list(correct)))
        available = [e for e in transitions if e['time'] <= query]
        if available and available[-1]['time'] == query and sum(e['time'] == query for e in available) == 1:
            event = available[-1]
            options = [f'The {names[obj]} {verb} view.' for obj in objects for verb in ('left', 'came into')]
            correct = f'The {names[event["object_id"]]} ' + ('left view.' if event['transition'] == 'left' else 'came into view.')
            evidence = [observation(event['object_id'], t) for t in (event['before'], query-1, query)]
            add('state_change', 'latest_observed_visibility_transition', query,
                'Among the ' + ', the '.join(names.values()) + ', which was the latest clear change in visibility?', options, correct, evidence,
                dict(object=event['object_id'], transition=event['transition'], time=query))
        recent_returns = [e for e in returns if e['time'] <= query]
        if recent_returns and recent_returns[-1]['time'] == query and sum(e['time'] == query for e in recent_returns) == 1:
            event = recent_returns[-1]
            alternatives = list(names.values())
            if uncertainty_policy == 'reference':
                alternatives.append('None of these objects returned to view.')
            add('object_tracking', 'object_return_after_absence', query,
                'Among the ' + ', the '.join(names.values()) + ', which has just returned to view after an earlier disappearance?', alternatives, names[event['object_id']],
                [observation(event['object_id'], t) for t in (event['previous_visible'], event['before'], query-1, query)],
                dict(object=event['object_id'], return_time=query))
        for obj in objects:
            if seen_at[obj] > query - 5:
                continue
            for start in (0, max(0, query - 20)):
                relevant = [e for e in returns if e['object_id'] == obj and start < e['time'] <= query]
                count = min(len(relevant), 3)
                labels = ['No returns.', 'One return.', 'Two returns.', 'Three or more returns.']
                start_text = 'Since the start of the video' if start == 0 else f'After {start:.0f} seconds'
                add('history_aggregation', 'observed_return_count', query,
                    f'{start_text}, how many clear returns of the {names[obj]} are shown after it was completely out of view?',
                    labels, labels[count], [observation(obj, t) for t in times if start <= t <= query],
                    dict(object=obj, start=start, return_times=[e['time'] for e in relevant], count=count))

    for event in episode['events']:
        if event['visibility'] != 'offscreen':
            continue
        obj = event['object_id']
        col = visibility['objects'].index(obj)
        lo, hi = math.floor(event['start'] * fps), math.ceil(event['end'] * fps)
        if not all(row[col] == 0 for row in visibility['counts'][lo:hi]):
            continue
        earlier = [t for t in times if states[obj][t] == 'visible' and t < event['start']]
        later = [t for t in times if states[obj][t] == 'visible' and t > event['end']]
        during = [t for t in times if event['start'] <= t and t + 1/fps < event['end']]
        if not earlier or not later:
            continue
        before, after = earlier[-1], later[0]
        evidence = [observation(obj, t) for t in (before, after)]
        anchor = dict(object=obj, event=event['event_index'], hidden_interval=[event['start'], event['end']],
                      before=before, after=after, native_hidden_frames=[lo, hi])
        add('object_tracking', 'physical_identity_after_gap', after,
            f'When the {names[obj]} returns, what can be established about its physical identity?',
            ['The original object returned without any hidden modification.', 'The original object returned after a hidden modification.', 'A replacement object returned instead of the original.'],
            UNDETERMINED, evidence, anchor, 'uncertain', ['ambiguous_evidence', 'missing_observation'],
            'The object is absent during the indexed interval; the images do not certify physical-instance continuity.')
        add('history_aggregation', 'hidden_manipulation_count', after,
            f'Between the earlier and later views of the {names[obj]}, how many separate manipulations took place?',
            ['No manipulation.', 'Exactly one manipulation.', 'Two or more manipulations.'], UNDETERMINED,
            evidence, anchor, 'uncertain', ['missing_observation', 'multiple_candidates'],
            'The recorded views do not show the full manipulation interval, so its action count is not established.')
        add('state_change', 'unobserved_change_cause', after,
            f'How was the {names[obj]} handled while it was out of view?',
            ['A person directly operated it.', 'It changed without being touched.', 'Another object came into contact with it.'],
            UNDETERMINED, evidence, anchor, 'uncertain', ['missing_observation'],
            'The target is absent throughout the operation; the images do not establish the physical cause.')
        add('temporal_reasoning', 'unobserved_action_order', after,
            f'Which sequence of interactions with the {names[obj]} occurred between its earlier and later appearances?',
            ['It was picked up and then set down.', 'It was set down and then picked up.', 'It was picked up, set down, and picked up again.'],
            UNDETERMINED, evidence, anchor, 'uncertain', ['missing_observation', 'multiple_candidates'],
            'No target pixels are visible during the operation, so the contact/release sequence is not observed.')
        if during and event['verb'] in ('Grab', 'PutBack', 'PutIn'):
            hidden_time = during[len(during)//2]
            descriptions = ['Resting at its last visible position.', 'Resting elsewhere in the room.', 'Being carried by a person.']
            add('current_state', 'unobserved_current_location', hidden_time,
                f'At the question point, which location description applies to the {names[obj]}?', descriptions,
                UNDETERMINED, [observation(obj, t) for t in (before, hidden_time)], anchor,
                'uncertain', ['missing_observation'], 'The object is out of view at the query point; its current location is not observed.')
            add('factual_retrieval', 'unobserved_earlier_location', after,
                f'At {hidden_time:.0f} seconds, which location description applied to the {names[obj]}?', descriptions,
                UNDETERMINED, [observation(obj, t) for t in (before, hidden_time, after)], anchor,
                'uncertain', ['missing_observation'], 'The queried earlier location lies inside an interval with no target observations.')
    return rows


def select_visual_batch(rows, reference, *, questions_per_episode=10, seed=20260921,
                        reference_duration=60.05, time_limit=120):
    """Match type/status proportions while keeping per-episode coverage explicit."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    from .aligned_qa import apportion

    episodes = sorted({q['episode_id'] for q in rows})
    total = len(episodes) * questions_per_episode
    type_quota = apportion(reference, total)
    status_counts = Counter(q['diagnostics']['epistemic_status'] for q in reference)
    uncertain_total = round(total * status_counts['uncertain'] / len(reference))
    ref_unc = Counter(q['question_type'] for q in reference if q['diagnostics']['epistemic_status'] == 'uncertain')
    exact = {k: v * uncertain_total / sum(ref_unc.values()) for k, v in ref_unc.items()}
    uncertain_quota = {k: math.floor(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - uncertain_quota[k]), k))[:uncertain_total-sum(uncertain_quota.values())]:
        uncertain_quota[k] += 1
    groups = defaultdict(list)
    for i, q in enumerate(rows):
        kind, status = q['question_type'], q['diagnostics']['epistemic_status']
        groups['episode', q['episode_id']].append(i)
        groups['type', kind].append(i)
        groups['type_status', kind, status].append(i)
        groups['episode_type', q['episode_id'], kind].append(i)
        groups['episode_status', q['episode_id'], status].append(i)
        anchor = q['semantic_anchor']
        if q['question_subtype'] == 'visible_room_category':
            fact = ('room_category', kind)
        elif status == 'uncertain':
            fact = ('hidden_event', anchor['event'])
        elif kind == 'factual_retrieval':
            fact = (kind, tuple(anchor['objects']), anchor['observation_time'])
        elif kind == 'temporal_reasoning':
            fact = (kind, tuple(anchor['objects']), tuple(anchor['return_times']))
        elif kind == 'history_aggregation':
            fact = (kind, anchor['object'], anchor['start'], tuple(anchor['return_times']))
        else:
            fact = (kind, json.dumps(anchor, sort_keys=True))
        groups['fact', q['episode_id'], fact].append(i)
    constraints = []
    for key, indices in groups.items():
        if key[0] == 'episode':
            bounds = (questions_per_episode, questions_per_episode)
        elif key[0] == 'type':
            bounds = (type_quota[key[1]],) * 2
        elif key[0] == 'type_status':
            count = uncertain_quota[key[1]] if key[2] == 'uncertain' else type_quota[key[1]] - uncertain_quota[key[1]]
            bounds = (count, count)
        elif key[0] == 'episode_type':
            bounds = (0, 3)
        elif key[0] == 'episode_status':
            bounds = (1, 2) if key[2] == 'uncertain' else (8, 9)
        else:
            bounds = (0, 1)
        constraints.append((indices, *bounds))
    # Soft marginals change which supported questions are selected, never their labels.
    marginals = defaultdict(list)
    targets = Counter()
    def cells(q, duration):
        status = q['diagnostics']['epistemic_status']
        time_bin = min(4, int(5 * q['query_time'] / duration))
        return [('intensity', q['change_intensity']), ('time', time_bin),
                ('status_time', status, time_bin),
                ('status_intensity', status, q['change_intensity'])]
    for i, q in enumerate(rows):
        for cell in cells(q, q['provenance']['duration_seconds']):
            marginals[cell].append(i)
    for q in reference:
        targets.update(cells(q, reference_duration))
    soft = []
    for cell in sorted(set(marginals) | set(targets)):
        soft.append((marginals[cell], total * targets[cell] / len(reference),
                     1.0 if cell[0] in ('time', 'intensity') else .5))
    rr, cc, vv = [], [], []
    for row, (indices, _, _) in enumerate(constraints):
        rr.extend([row]*len(indices)); cc.extend(indices); vv.extend([1.]*len(indices))
    lower = [c[1] for c in constraints]
    upper = [c[2] for c in constraints]
    slack_cost = []
    for j, (indices, target, weight) in enumerate(soft):
        row = len(constraints) + j
        rr.extend([row] * (len(indices) + 2))
        cc.extend(indices + [len(rows) + 2*j, len(rows) + 2*j + 1])
        vv.extend([1.] * len(indices) + [-1., 1.])
        lower.append(target); upper.append(target)
        slack_cost.extend([weight, weight])
    variables = len(rows) + 2*len(soft)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), variables)).tocsc()
    cost = np.array([.01 * random.Random(f'{seed}:{q["id"]}').random() for q in rows] + slack_cost)
    for i, q in enumerate(rows):
        visible = [o['pixels'] for o in q['provenance']['observations'] if o['visibility'] == 'visible']
        if visible:
            cost[i] += .1 / math.sqrt(min(visible))
    fit = milp(cost, integrality=np.r_[np.ones(len(rows)), np.zeros(2*len(soft))],
               bounds=Bounds(np.zeros(variables), np.r_[np.ones(len(rows)), np.full(2*len(soft), np.inf)]),
               constraints=LinearConstraint(matrix, lower, upper),
               options=dict(time_limit=time_limit, mip_rel_gap=.02))
    if fit.x is None:
        raise ValueError(f'No evidence-grounded allocation satisfies the requested distribution: {fit.message}')
    selected = [q for i, q in enumerate(rows) if fit.x[i] > .5]
    assert len(selected) == total
    actual = Counter()
    for q in selected:
        actual.update(cells(q, q['provenance']['duration_seconds']))
    return sorted(selected, key=lambda q: (q['episode_id'], q['query_time'], q['id'])), dict(
        types=type_quota, uncertain=uncertain_quota, solver_message=fit.message,
        marginal_counts=[dict(field=list(cell), target=total*targets[cell]/len(reference), actual=actual[cell])
                         for cell in sorted(set(marginals) | set(targets))])
