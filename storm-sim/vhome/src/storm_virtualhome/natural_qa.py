"""Build variable-choice questions from timestamped observer evidence."""
from collections import Counter, defaultdict
import random
import hashlib
from itertools import combinations

from .qa import TYPES, label

UNKNOWN = 'The current state or location cannot be determined from the available observations.'
SCHEMA = 'natural_structure_v2'


def family(event):
    if event['verb'] in ('Open', 'Close'):
        return 'openable'
    if event['verb'] in ('SwitchOn', 'SwitchOff'):
        return 'switchable'
    return 'portable'


def build_natural_questions(episode, events, seed=0):
    """Keep option domains independent of the correct answer and observation status."""
    rng = random.Random(seed)
    names = {}
    for event in events:
        names[event['object_id']] = event.get('object_description', label(event['object']))
    groups = defaultdict(dict)
    for event in events:
        groups[family(event)][event['object_id']] = names[event['object_id']]
    portable_states = sorted({e[key] for e in events if family(e) == 'portable'
                              for key in ('before_state', 'after_state')})
    if len(set(names.values())) != len(names):
        raise ValueError('Objects need unique, visible descriptions before QA generation')
    questions = []

    def add(event, kind, variant, question, answer, options, query, spans, unknown=False, semantic_anchor=None):
        choices = list(dict.fromkeys(options))
        if not 2 <= len(choices) <= 4 or answer not in choices:
            return
        if query <= 0 or any(not 0 <= a < b <= query for a, b in spans):
            raise ValueError('Evidence must precede the query boundary')
        rng.shuffle(choices)
        key = f"{SCHEMA}:{episode}:{event['event_index']}:{kind}:{variant}"
        questions.append(dict(
            id=f"{episode}_n{hashlib.sha256(key.encode()).hexdigest()[:12]}",
            generation_key=key, semantic_anchor=semantic_anchor,
            episode_id=episode, event_index=event['event_index'], question_type=kind,
            question_subtype=variant, question=question, options=choices,
            answer_index=choices.index(answer), query_time=query, evidence_spans=spans,
            diagnostics=dict(epistemic_status='uncertain' if unknown else 'known',
                             uncertainty_sources=['offscreen_change'] if unknown else []),
            change_intensity='high', qa_schema=SCHEMA, num_choices=len(choices),
            chance_accuracy=1 / len(choices),
            video_evidence=f'Observer evidence for {kind}: ' + '; '.join(f'{a:.2f}–{b:.2f} seconds' for a, b in spans),
            evidence_review='timestamp_and_capture_audit_only'))

    for event in events:
        name = names[event['object_id']]
        before, after = event['before_state'], event['after_state']
        start, end = event['before_time'], event['discovery_time']
        spans = [[max(0, start - .5), start], [max(0, end - .5), end]]
        group = family(event)
        domain = ['closed', 'open'] if group == 'openable' else (
            ['off', 'on'] if group == 'switchable' else portable_states)
        state_word = 'location' if group == 'portable' else 'state'
        add(event, 'factual_retrieval', 'previous_observation',
            f'Before the latest observed change to the {name}, what was its {state_word}?',
            before, domain, end, spans[:1])
        current_question = f'What is the current {state_word} of the {name} at the end of the clip?'
        add(event, 'current_state', 'observed', current_question, after,
            domain + [UNKNOWN], end, spans[-1:])
        if event['visibility'] == 'offscreen':
            blind = event.get('hidden_time', event['end'])
            add(event, 'current_state', 'unobserved', current_question, UNKNOWN,
                domain + [UNKNOWN], blind,
                [spans[0], [max(0, blind - .5), blind]], True)
        if before != after:
            directions = [f'From {before} to {after}.', f'From {after} to {before}.']
            add(event, 'state_change', 'change_direction',
                f'What was the latest observed change in the {state_word} of the {name}?', directions[0], directions, end, spans)
            candidates = list(groups[group].values())
            competing = any(other['object_id'] != event['object_id']
                            and other['before_state'] == before and other['after_state'] == after
                            and start < other['discovery_time'] <= end
                            for other in events)
            if competing:
                candidates = []
            add(event, 'object_tracking', 'changed_object',
                f'Which object was most recently seen to have changed from {before} to {after}?', name, candidates, end, spans)

        # Compare distinct objects so ordering does not duplicate previous-state recall.
        seen = {e['object_id'] for e in events if e['before_time'] < end}
        confirmed = [e for e in events if e['discovery_time'] <= end
                     and e['before_state'] != e['after_state']]
        counts = Counter(e['object_id'] for e in confirmed)
        for left, right in combinations(sorted(seen), 2):
            if family(next(e for e in events if e['object_id'] == left)) != family(next(e for e in events if e['object_id'] == right)):
                continue
            pair_events = [e for e in confirmed if e['object_id'] in (left, right)]
            if not pair_events:
                continue
            pair_spans = [[max(0, e['before_time'] - .5), e['discovery_time']] for e in pair_events]
            equal = 'Both have the same number of confirmed changes.'
            answer = equal if counts[left] == counts[right] else names[left if counts[left] > counts[right] else right]
            add(event, 'history_aggregation', f'comparative_count_{left}_{right}',
                f'Across the observations so far, which has more confirmed changes: the {names[left]} or the {names[right]}?',
                answer, [names[left], names[right], equal], end, pair_spans)
            first = {obj: min((e['discovery_time'] for e in pair_events if e['object_id'] == obj), default=None)
                     for obj in (left, right)}
            if None in first.values() or first[left] == first[right]:
                continue
            answer = names[left if first[left] < first[right] else right]
            add(event, 'temporal_reasoning', f'first_confirmation_{left}_{right}',
                f'For the {names[left]} and the {names[right]}, which was first seen to have changed?',
                answer, [names[left], names[right]], end, pair_spans)
            latest = {obj: max(e['discovery_time'] for e in pair_events if e['object_id'] == obj)
                      for obj in (left, right)}
            if latest[left] != latest[right]:
                answer = names[left if latest[left] > latest[right] else right]
                add(event, 'temporal_reasoning', f'latest_confirmation_{left}_{right}',
                    f'For the {names[left]} and the {names[right]}, which was most recently seen to have changed?',
                    answer, [names[left], names[right]], end, pair_spans,
                    semantic_anchor=(latest[left], latest[right]))

    return questions


def _request_key(question):
    """A first-confirmation fact is unchanged by showing a longer prefix."""
    time = question.get('semantic_anchor') if question['question_type'] == 'temporal_reasoning' else question['query_time']
    if isinstance(time, list):
        time = tuple(time)
    return question['question'], time, tuple(sorted(question['options']))


def _select_natural_batch(pools, seed=0):
    """Allocate ten questions per episode without consulting model predictions."""
    rng = random.Random(seed)
    support = defaultdict(set)
    for pool in pools:
        for q in pool:
            support[q['question_type'], tuple(sorted(q['options']))].add(q['options'][q['answer_index']])
    pools = [[q for q in pool if support[q['question_type'], tuple(sorted(q['options']))] == set(q['options'])]
             for pool in pools]
    semantic_counts, type_counts = Counter(), Counter()
    selected = []
    for slot, pool in enumerate(pools):
        used_ids, event_counts = set(), Counter()
        used_requests = set()
        kinds = [TYPES[(i + slot * 4) % len(TYPES)] for i in range(10)]
        rng.shuffle(kinds)
        for kind in kinds:
            candidates = [q for q in pool if q['question_type'] == kind and q['id'] not in used_ids
                          and _request_key(q) not in used_requests
                          and semantic_counts[kind, tuple(sorted(q['options'])), q['options'][q['answer_index']]]
                          <= min(semantic_counts[kind, tuple(sorted(q['options'])), answer] for answer in q['options'])]
            if not candidates:
                raise ValueError(f'Episode {slot} cannot supply enough {kind} questions')
            rng.shuffle(candidates)
            def rank(q):
                answer = q['options'][q['answer_index']]
                domain = tuple(sorted(q['options']))
                return (semantic_counts[kind, domain, answer], type_counts[kind, answer],
                        event_counts[q['event_index']])
            q = min(candidates, key=rank)
            used_ids.add(q['id'])
            used_requests.add(_request_key(q))
            event_counts[q['event_index']] += 1
            answer = q['options'][q['answer_index']]
            semantic_counts[kind, tuple(sorted(q['options'])), answer] += 1
            type_counts[kind, answer] += 1
            selected.append(q)
    return selected


def select_natural_batch(pools, seed=0):
    """Search only annotation-based allocations; never inspect model predictions."""
    last_error = None
    for attempt in range(256):
        ordered = list(pools)
        random.Random(seed + attempt).shuffle(ordered)
        try:
            return _select_natural_batch(ordered, seed + attempt)
        except ValueError as error:
            last_error = error
    raise ValueError(f'No balanced allocation exists within the search budget: {last_error}')


def audit_questions(rows):
    """Report empirical answer priors alongside each question's uniform baseline."""
    groups = defaultdict(Counter)
    types = defaultdict(list)
    for q in rows:
        assert len(q['options']) == len(set(q['options'])) == q['num_choices']
        assert 2 <= q['num_choices'] <= 4
        answer = q['options'][q['answer_index']]
        groups[q['question_type'], tuple(sorted(q['options']))][answer] += 1
        types[q['question_type']].append(q)
    return dict(schema=SCHEMA, questions=len(rows),
                arities=dict(Counter(q['num_choices'] for q in rows)),
                uniform_chance=sum(q['chance_accuracy'] for q in rows) / len(rows),
                option_set_majority_oracle=sum(max(v.values()) for v in groups.values()) / len(rows),
                interpretation='The option-set majority oracle is an in-sample artifact diagnostic, not a held-out model score.',
                by_type={k:dict(n=len(v), chance=sum(q['chance_accuracy'] for q in v)/len(v),
                               answers=dict(Counter(q['options'][q['answer_index']] for q in v)))
                         for k, v in types.items()})
