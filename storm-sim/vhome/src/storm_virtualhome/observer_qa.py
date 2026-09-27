"""Build questions from observations without exposing hidden action labels."""
import random

from .qa import label


def build_observer_questions(episode_id, event, stages, duration, seed=42, cycle_index=0, history_times=None):
    rng = random.Random(seed)
    name, surface = label(event['object']), label(event['surface'])
    held = event.get('held_by_injector', False)
    location = "in the other person's hand" if held else f'on the {surface}'
    initial = stages['before']['time']
    blind = stages['hidden_unobserved']['time']
    discovered = stages['absence_discovered']['time']
    returned = stages['returned']['time']
    if not 0 < initial < blind < discovered < returned <= duration:
        raise ValueError('Observation checkpoints must be ordered inside the video')
    result = []

    def add(kind, subtype, query, question, answer, alternatives, times, unknown=False):
        choices = [answer, *alternatives]
        if len(set(choices)) != 4:
            raise ValueError('Exactly four distinct choices are required')
        rng.shuffle(choices)
        spans = [[max(0, t - 0.5), t] for t in times]
        if any(end > query for _, end in spans):
            raise ValueError('An answer cannot use future observations')
        result.append({'id': f'{episode_id}_q{cycle_index * 8 + len(result):03d}', 'episode_id': episode_id,
                       'query_time': query, 'question_type': kind, 'question_subtype': subtype,
                       'question': question, 'options': choices, 'answer_index': choices.index(answer),
                       'evidence_spans': spans,
                       'diagnostics': {'epistemic_status': 'uncertain' if unknown else 'known',
                                       'uncertainty_sources': ['offscreen_change'] if unknown else []},
                       'video_evidence': 'Use only the observer camera up to the query time.',
                       'change_intensity': 'high'})

    add('factual_retrieval', 'last_observed_location', blind,
        f'Where was the {name} last seen before the camera most recently turned away?', location,
        ['on the floor', 'on the kitchen table', 'on a chair'], [initial])
    add('current_state', 'offscreen_location', blind,
        f'At the end of this clip, where is the {name} now?',
        'Its current location cannot be determined from this view.',
        [f'It is still {location}.', 'It is inside the refrigerator.', 'It is on the floor.'],
        [initial, blind], unknown=True)
    add('state_change', 'observed_absence', discovered,
        (f'When the camera returned to the other person, how did the {name} visibility compare with the earlier view?' if held else
         f'What changed at the place on the {surface} where the {name} was last seen?'),
        f'The {name} is no longer visible there.',
        [f'The {name} is still there.', f'Two {name}s are now there.', 'The other person has disappeared.' if held else f'The {surface} has disappeared.'],
        [initial, discovered])
    add('state_change', 'unobserved_mechanism', discovered,
        f'What does the video establish about how the {name} left its previous place?',
        'The removal happened outside the view; the method was not observed.',
        ['It was visibly put into the refrigerator.', 'It visibly fell onto the floor.',
         'The observer visibly picked it up.'], [initial, blind, discovered], unknown=True)
    add('current_state', 'visible_location', returned,
        f'Where is the {name} visible at the end of the clip?', location,
        ['on the floor', 'inside a closed cupboard', 'on a chair'], [returned])
    add('object_tracking', 'observed_return_location', returned,
        (f'Who is holding the {name} at the end of this clip after its latest absence?' if held else
         f'Where has the {name} been placed at the end of this clip after its latest absence?'),
        ('The other person.' if held else location), (['The blue-shirted observer.', 'Neither person.', 'Both people.'] if held else
         ['on the kitchen table', 'on a chair', 'on the floor']), [discovered, returned])
    add('temporal_reasoning', 'observation_order', returned,
        (f'In the latest sequence, was the {name} first found out of view, or seen back in the other person’s hand?' if held else
         f'In the latest absence-and-return sequence, which was observed first: the {name} missing from the {surface}, or visible on it again?'),
        ('It was found out of view first.' if held else f'It was found missing from the {surface} first.'),
        [('It was seen in the other person’s hand again first.' if held else f'It was seen on the {surface} again first.'), 'Both were observed at the same time.', 'Neither was observed.'],
        [discovered, returned])
    add('history_aggregation', 'observed_return_count', returned,
        (f'How many times has the {name} been seen back in the other person’s hand after a check found it out of view?' if held else
         f'How many times has the {name} been seen back on the {surface} after that place was seen without it?'),
        str(cycle_index + 1), [str(n) for n in range(cycle_index + 5) if n != cycle_index + 1][:3],
        history_times or [initial, discovered, returned])
    return result
