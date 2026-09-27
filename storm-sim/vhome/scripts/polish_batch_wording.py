#!/usr/bin/env python3
"""Apply reviewed English wording to batch questions without changing evaluation labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

PORTABLES = {'bowl', 'dishwashing liquid bottle'}
STATE_OBJECTS = {'microwave', 'television',
                 'upper cabinet above the sink on the refrigerator side',
                 'upper cabinet above the sink on the stove side'}
OBJECT_QUESTIONS = {
    'Which object changed from closed to open in the latest comparison?':
        'Between the two most recent observations, which object went from closed to open?',
    'Which object changed from open to closed in the latest comparison?':
        'Between the two most recent observations, which object went from open to closed?',
    'Which object changed from on to off in the latest comparison?':
        'Between the two most recent observations, which object went from on to off?',
    'Which object changed from in the woman’s hand to on the kitchen table in the latest comparison?':
        'Which object moved from the woman’s hand to the kitchen table between the two most recent observations?',
    'Which object changed from on the kitchen counter to in the woman’s hand in the latest comparison?':
        'Which object moved from the kitchen counter to the woman’s hand between the two most recent observations?',
}


def rewrite(question):
    if question in OBJECT_QUESTIONS:
        return OBJECT_QUESTIONS[question]
    patterns = [
        (r'At the preceding inspection, what was the state or location of the (.+)\?', 'previous'),
        (r'At the end of this clip, what is the state or location of the (.+)\?', 'current'),
        (r'At the end of this clip, what state is the (.+) in\?', 'current'),
        (r'At the end of this clip, where is the (.+) now\?', 'current'),
        (r'What changed between the two most recent inspections of the (.+)\?', 'change'),
        (r'Which state or location of the (.+) was seen first in the latest comparison\?', 'temporal'),
        (r'How many changes to the (.+) have been confirmed by successive inspections so far\?', 'history'),
    ]
    for pattern, kind in patterns:
        match = re.fullmatch(pattern, question)
        if not match:
            continue
        name = match[1]
        if name not in PORTABLES | STATE_OBJECTS:
            raise ValueError(f'Object wording needs review: {name}')
        portable = name in PORTABLES
        if kind == 'previous':
            return f'Where was the {name} at the previous observation?' if portable else f'What state was the {name} in at the previous observation?'
        if kind == 'current':
            return f'Where is the {name} at the end of the clip?' if portable else f'What state is the {name} in at the end of the clip?'
        if kind == 'change':
            return f'What changed about the {name} between the two most recent observations?'
        if kind == 'temporal':
            return f'In the two most recent observations of the {name}, which {"location" if portable else "state"} was seen first?'
        return f'How many changes to the {name} have been confirmed across observations so far?'
    raise ValueError(f'Question needs manual wording review: {question}')


def polish(source, output):
    rows = [json.loads(line) for line in (source / 'qa_private.jsonl').read_text().splitlines()]
    inputs = [json.loads(line) for line in (source / 'model_inputs.jsonl').read_text().splitlines()]
    labels = [json.loads(line) for line in (source / 'evaluation_labels.jsonl').read_text().splitlines()]
    originals = {q['id']: q for q in rows}
    if len(originals) != len(rows) or set(originals) != {q['id'] for q in inputs} or set(originals) != {q['id'] for q in labels}:
        raise ValueError('Question IDs must be unique and match across all three files')
    changes = {key: rewrite(q['question']) for key, q in originals.items()}
    ledger = []
    for q in rows:
        ledger.append({'id': q['id'], 'question_type': q['question_type'], 'original': q['question'], 'revised': changes[q['id']]})
    revised = [dict(q, question=changes[q['id']]) for q in rows]
    revised_inputs = []
    for q in inputs:
        original = originals[q['id']]
        if q['question'] != original['question'] or q['options'] != original['options'] or q['query_time'] != original['query_time']:
            raise ValueError('Model inputs disagree with the private question record')
        revised_inputs.append(dict(q, question=changes[q['id']]))
    for before, after in zip(rows, revised):
        assert {k: v for k, v in before.items() if k != 'question'} == {k: v for k, v in after.items() if k != 'question'}
    output.mkdir(parents=True, exist_ok=False)
    for filename, data in [('qa_private.jsonl', revised), ('model_inputs.jsonl', revised_inputs), ('evaluation_labels.jsonl', labels), ('wording_changes.jsonl', ledger)]:
        (output / filename).write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in data))
    report = {'questions': len(rows), 'changed': sum(x['original'] != x['revised'] for x in ledger),
              'unique_original_wordings': len({q['question'] for q in rows}),
              'qa_types': dict(Counter(q['question_type'] for q in revised)),
              'only_question_field_changed': True, 'model_video_paths_unchanged': True,
              'method': 'Reviewed English template editing; no external API or video reannotation',
              'video_path_base': 'The original batch root; the wording directory contains metadata only'}
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(polish(args.input, args.output), indent=2))
