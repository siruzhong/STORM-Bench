"""Apply the established candidate question templates."""
from copy import deepcopy

def revise(question, proof):
    q = deepcopy(question)
    anchor = proof['semantic_anchor']
    names = {obs['object_id']: obs['object_name'] for obs in proof['provenance']['observations']}
    kind = q['question_subtype']
    now = f"{q['query_time']:g}"
    obj = names.get(anchor.get('object'))
    flags = []
    if kind == 'visible_room_category':
        q['question'] = ('What type of room is shown at the start of the video?'
                         if q['question_type'] == 'factual_retrieval'
                         else 'What type of room is shown at this point in the video?')
    elif kind in ('current_visibility_pair', 'earlier_visibility_pair'):
        left, right = [names[obj_id] for obj_id in anchor['objects']]
        when = (f'At {now} seconds' if kind == 'current_visibility_pair'
                else f"Looking back at {anchor['observation_time']:g} seconds")
        q['question'] = f'{when}, which option correctly describes whether the {left} and the {right} are in view?'
    elif kind == 'last_two_observed_returns':
        prefix, suffix = 'For the ', ', which order describes their last two returns to view?'
        assert q['question'].startswith(prefix) and q['question'].endswith(suffix)
        left, right = q['question'][len(prefix):-len(suffix)].split(' and the ')
        for obj_id, label in zip(anchor['objects'], (left, right)):
            assert obj_id not in names or names[obj_id] == label
        q['question'] = (
            f'Consider only the {left} and the {right}. In the two most recent returns to view, '
            'which object appeared first and which appeared second? '
            'Count each reappearance after a complete disappearance; the same object can appear twice.'
        )
    elif kind == 'latest_observed_visibility_transition':
        suffix = ', which was the latest clear change in visibility?'
        assert q['question'].endswith(suffix)
        candidates = q['question'][:-len(suffix)]
        q['question'] = candidates + ', which of the listed changes happened most recently?'
    elif kind == 'object_return_after_absence':
        suffix = ', which has just returned to view after an earlier disappearance?'
        assert q['question'].endswith(suffix)
        candidates = q['question'][:-len(suffix)]
        q['question'] = candidates + ', which has just reappeared after being completely out of view?'
    elif kind == 'observed_return_count':
        start = anchor['start']
        when = ('From the start of the video up to this point' if start == 0
                else f'After {start:g} seconds and up to this point')
        q['question'] = f'{when}, how many times has the {obj} reappeared after being completely out of view?'
        replacement = {'No returns.': 'Zero times.', 'One return.': 'Once.',
                       'Two returns.': 'Twice.', 'Three or more returns.': 'Three or more times.'}
        q['options'] = [replacement.get(option, option) for option in q['options']]
    elif kind == 'physical_identity_after_gap':
        q['question'] = (
            f"Compare the views of the {obj} at {anchor['before']:g} and {anchor['after']:g} seconds. "
            'Which statement about the object in the later view is supported by the video?'
        )
        replacement = {
            'The original object returned without any hidden modification.':
                'It is the original object, unchanged while out of view.',
            'The original object returned after a hidden modification.':
                'It is the original object, changed while out of view.',
            'A replacement object returned instead of the original.':
                'It is a replacement for the original object.',
        }
        q['options'] = [replacement.get(option, option) for option in q['options']]
        flags.append('Physical identity and unobserved modification still require semantic review.')
    elif kind == 'hidden_manipulation_count':
        q['question'] = (f"Between the views at {anchor['before']:g} and {anchor['after']:g} seconds, "
                         f'how many separate manipulations of the {obj} took place?')
    elif kind == 'unobserved_action_order':
        q['question'] = (f"Between the views at {anchor['before']:g} and {anchor['after']:g} seconds, "
                         f'in what order was the {obj} handled?')
        flags.append('Check that the handling alternatives are physically plausible for the object.')
    elif kind == 'unobserved_change_cause':
        q['question'] = (f"How was the {obj} affected between the views at "
                         f"{anchor['before']:g} and {anchor['after']:g} seconds?")
        flags.append('Cause alternatives may overlap; revising them requires a new evidence review.')
    elif kind == 'unobserved_current_location':
        q['question'] = f'Where is the {obj} at {now} seconds?'
    elif kind == 'unobserved_earlier_location':
        target_times = [obs['time'] for obs in proof['provenance']['observations']
                        if obs['object_id'] == anchor['object'] and obs['visibility'] == 'absent']
        assert len(set(target_times)) == 1
        q['question'] = f'Where was the {obj} at {target_times[0]:g} seconds?'
    else:
        raise ValueError(f'Unsupported subtype: {kind}')
    if 'closest to' in q['question'] or 'farthest from' in q['question']:
        flags.append('Verify that the relative object description can be resolved from the video.')
    return q, flags
