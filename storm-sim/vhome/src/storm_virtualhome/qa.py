"""Build six QA types from accepted event pairs and observed frame intervals."""
from __future__ import annotations

import random

LABELS = {'bellpepper': 'bell pepper', 'dishbowl': 'bowl', 'washingsponge': 'sponge',
          'dishwashingliquid': 'dishwashing liquid bottle', 'coffeepot': 'coffee pot',
          'waterglass': 'drinking glass', 'wineglass': 'wine glass', 'cutleryknife': 'knife',
          'cutleryfork': 'fork', 'kitchencounter': 'kitchen counter', 'kitchentable': 'kitchen table',
          'kitchencabinet': 'kitchen cabinet', 'bathroomcounter': 'bathroom counter',
          'bathroomcabinet': 'bathroom cabinet', 'washingmachine': 'washing machine',
          'tv': 'television', 'tvstand': 'TV stand', 'whippedcream': 'whipped cream can'}


def label(value):
    return LABELS.get(value, value)


TYPES = ('factual_retrieval', 'current_state', 'state_change', 'object_tracking',
         'temporal_reasoning', 'history_aggregation')


def build_questions(episode_id, pairs, duration, seed=42):
    if not pairs:
        raise ValueError('At least one visually verified event pair is required')
    rng = random.Random(seed)
    questions = []

    def add(kind, subtype, question, correct, alternatives, evidence, query, description):
        choices = list(dict.fromkeys([correct, *alternatives]))
        if len(choices) < 4:
            raise ValueError(f'Not enough distinct choices for {kind}')
        choices = [correct, *[x for x in choices if x != correct][:3]]
        rng.shuffle(choices)
        if any(start < 0 or end < start or end > query + 1e-6 for start, end in evidence):
            raise ValueError('Evidence must lie within the question prefix')
        questions.append(dict(
            id=f'{episode_id}_q{len(questions):03d}', episode_id=episode_id,
            query_time=round(query, 3), question_type=kind, question_subtype=subtype,
            video_evidence=description, question=question, options=choices,
            answer_index=choices.index(correct), evidence_spans=evidence,
            diagnostics={'epistemic_status': 'known', 'uncertainty_sources': []},
            diagnostic_rationale={'volatility': 'The object is moved by the character.',
                                  'uncertainty': 'The answer is supported by the recorded prefix.'},
            change_intensity='high'))

    for i, pair in enumerate(pairs):
        name, container, surface = (label(pair[key]) for key in ('object', 'container', 'surface'))
        hidden, restored = pair['disappear'], pair['appear']
        hide_span, show_span = [hidden['start'], hidden['end']], [restored['start'], restored['end']]
        end = restored['end']
        add('factual_retrieval', 'interaction_destination',
            f'Where did the person put the {name} when putting it away?', f'in the {container}',
            ['on the floor', 'on a chair', 'in their pocket'], [hide_span], hidden['end'],
            f'The person places the {name} in the {container} and closes it.')
        add('current_state', 'visible_location', f'At the end of this clip, where is the {name} that the person handled?',
            f'on the {surface}', [f'inside the {container}', 'in the person’s hand', 'on the floor'],
            [show_span], end, f'The {name} is visible on the {surface} after the person returns it.')
        add('state_change', 'disappearance_mechanism', f'How did the {name} become hidden?',
            f'It was placed in the {container}, which was then closed.',
            ['It was moved behind a large box.', 'It was carried out of the room.', 'It was covered by a towel.'],
            [hide_span], hidden['end'], f'The {name} becomes hidden when the {container} is closed.')
        add('object_tracking', 'return_location',
            f'After being taken out of the {container}, where was the {name} returned?',
            f'the {surface}', [f'the {container}', 'the floor', 'the kitchen table', 'a chair'], [hide_span, show_span], end,
            f'The person retrieves the same {name} and returns it to the {surface}.')
        add('history_aggregation', 'completed_return_count',
            'How many times has the person completed putting an object away and returning it so far?',
            str(i + 1), [str(x) for x in range(len(pairs) + 4) if x != i + 1],
            [[p['appear']['start'], p['appear']['end']] for p in pairs[:i + 1]], end,
            f'{i + 1} complete put-away-and-return cycles occur in this prefix.')
        if i:
            previous = pairs[i - 1]
            previous_name = label(previous['object'])
            add('temporal_reasoning', 'return_order',
                f'Which was returned to its original surface first: the {previous_name} or the {name}?',
                f'the {previous_name}', [f'the {name}', 'both at the same time', 'neither was returned'],
                [[previous['appear']['start'], previous['appear']['end']], show_span], end,
                f'The {previous_name} is returned before the {name}.')
        else:
            add('temporal_reasoning', 'interaction_order',
                f'What happened first to the {name} in this clip?',
                f'It was put away in the {container}.',
                [f'It was returned to the {surface}.', 'Both happened at the same time.', 'Neither happened.'],
                [hide_span, show_span], end,
                f'The {name} is put away before it is returned.')
    if not all(q['query_time'] <= duration + 1e-6 for q in questions):
        raise ValueError('Question timestamp exceeds video duration')
    return questions
