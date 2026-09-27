"""Revise an aligned QA release without changing labels or evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.qa_export import (
    distribution, model_prompt, sha256, validate_annotation_contract,
    write_json, write_jsonl,
)

VERSION = 'virtualhome_reference_options_language_revision_20260922'
EXPECTED_PARENT = 'fad5c74a69d95d95a4a6ac402273c04a0222afbc9d17575e0c9e8f1739fd430e'


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source, output = args.source, args.output
    assert sha256(source / 'questions.jsonl') == EXPECTED_PARENT, 'Unexpected source release'
    if output.exists():
        raise FileExistsError(f'Refusing to replace an existing release: {output}')
    original = read_rows(source / 'questions.jsonl')
    ledger = read_rows(source / 'private/qa_provenance.jsonl')
    old_gt = read_rows(source / 'evaluation_gt.jsonl')
    gt_by_id = {row['id']: row for row in old_gt}
    revised, changes, queue = [], [], []
    for q, proof in zip(original, ledger):
        assert q['id'] == proof['question_id']
        new, flags = revise(q, proof)
        assert {k: v for k, v in q.items() if k not in ('question', 'options')} == {
            k: v for k, v in new.items() if k not in ('question', 'options')}
        assert len(new['options']) == len(set(new['options'])) == 4
        revised.append(new)
        changes.append(dict(question_id=q['id'], question_subtype=q['question_subtype'],
                            before_question=q['question'], after_question=new['question'],
                            before_options=q['options'], after_options=new['options'],
                            answer_index=q['answer_index'], labels_and_evidence_unchanged=True))
        if flags:
            queue.append(dict(question_id=q['id'], issues=flags))
    assert len(revised) == 290 and len({q['episode_id'] for q in revised}) == 29
    output.mkdir(parents=True)
    for filename in ('video_manifest.json', 'evaluation_labels.json', 'evaluation_splits.json', 'evaluation_protocol.json'):
        shutil.copyfile(source / filename, output / filename)
    by_id = {q['id']: q for q in revised}
    for path in sorted((source / 'meta_data/qa_results').glob('*/*.json')):
        doc = json.loads(path.read_text())
        doc['questions'] = [by_id[q['id']] for q in doc['questions']]
        doc['qa_source'] = VERSION
        write_json(output / path.relative_to(source), doc)
    canonical, inputs = [], []
    for q in revised:
        row = deepcopy(gt_by_id[q['id']])
        prompt = model_prompt(q)
        row['question'] = f"At {q['query_time']:.2f} seconds, use only observations up to that point.\n" + prompt
        row['options'] = q['options']
        canonical.append(row)
        inputs.append(dict(question_id=q['id'], episode_id=q['episode_id'], video_path=row['video_path'],
                           query_time=q['query_time'], prompt=prompt))
    write_jsonl(output / 'questions.jsonl', revised)
    write_jsonl(output / 'evaluation_gt.jsonl', canonical)
    write_jsonl(output / 'model_inputs.jsonl', inputs)
    write_jsonl(output / 'private/qa_provenance.jsonl', ledger)
    write_jsonl(output / 'private/language_changes.jsonl', changes)
    write_jsonl(output / 'private/review_queue.jsonl', queue)
    manifest = json.loads((source / 'qa_revision_manifest.json').read_text())
    manifest.update(schema=VERSION, parent_questions_sha256=EXPECTED_PARENT,
                    parent_evaluation_gt_sha256=sha256(source / 'evaluation_gt.jsonl'),
                    questions_sha256=sha256(output / 'questions.jsonl'),
                    evaluation_gt_sha256=sha256(output / 'evaluation_gt.jsonl'),
                    revision_scope='Language and disambiguation; no selection, label, timing, or evidence changes.',
                    per_question_predictions_used=False, performance_claim='Not evaluated')
    write_json(output / 'qa_revision_manifest.json', manifest)
    durations = {p['question_id']: p['provenance']['duration_seconds'] for p in ledger}
    episode_durations = {q['episode_id']: durations[q['id']] for q in revised}
    stats = json.loads((source / 'distribution_alignment.json').read_text())
    new_stats = distribution(revised, durations=episode_durations)
    assert new_stats['marginals'] == stats['current']['marginals'], 'Categorical distribution changed'
    stats['current'] = new_stats
    write_json(output / 'distribution_alignment.json', stats)
    summary = json.loads((source / 'summary.json').read_text())
    summary.update(version=VERSION, empirical_accuracy=None,
                   candidate_status='Language revision; structural biases and RGB semantic review remain unresolved.')
    write_json(output / 'summary.json', summary)
    validation = validate_annotation_contract(output)
    validation.update(question_text_changed=sum(a['question'] != b['question'] for a, b in zip(original, revised)),
                      option_text_changed_questions=sum(a['options'] != b['options'] for a, b in zip(original, revised)),
                      preserved_fields='All public fields except question and options; option order retained.',
                      categorical_marginals='unchanged', review_queue_questions=len(queue),
                      questions_sha256=manifest['questions_sha256'])
    write_json(output / 'validation.json', validation)
    write_json(output / 'release_status.json', dict(annotation_export='complete', structural_validation='passed',
                video_materialization='pending', full_rgb_semantic_review='pending', evaluation='not_run'))
    lines = ['# QA language revision', '',
             'All 290 questions were revised. The 29 videos, answers, option order, evidence, query times, and type/status distributions are unchanged.',
             'The uncertainty-option rule and visibility-answer imbalance remain unchanged. This revision makes no accuracy or leakage-removal claim.',
             'The 53 uncertain and 237 known labels are preserved. The option-aware random rule still has an expected accuracy of 38.71%.',
             'No new evaluation was started. Prior prediction files do not apply to this text revision.', '',
             'See private/language_changes.jsonl for every edit and private/review_queue.jsonl for questions requiring evidence or option-design review.', '',
             '| Subtype | Questions |', '|---|---:|']
    for kind, count in sorted(Counter(q['question_subtype'] for q in revised).items()):
        lines.append(f'| {kind} | {count} |')
    lines += ['', '## Examples', '']
    for kind in ('last_two_observed_returns', 'observed_return_count', 'earlier_visibility_pair',
                 'physical_identity_after_gap', 'hidden_manipulation_count', 'unobserved_action_order'):
        edit = next(e for e in changes if e['question_subtype'] == kind)
        lines += [f"### {kind}", '', f"Before: {edit['before_question']}", '', f"After: {edit['after_question']}", '']
    (output / 'QA_REVISION_REPORT.md').write_text('\n'.join(lines) + '\n')
    sample = [q for q in revised if q['episode_id'] == revised[0]['episode_id']]
    lines = ['# Revised episode sample', '', f"Episode: {sample[0]['episode_id']}", '']
    for i, q in enumerate(sample, 1):
        lines += [f"## {i}. {q['query_time']:g} s", '', q['question'], '']
        lines += [f'{chr(65+j)}. {option}' for j, option in enumerate(q['options'])]
        lines += ['', f"Answer: {chr(65+q['answer_index'])}; status: {q['diagnostics']['epistemic_status']}", '']
    (output / 'EPISODE_SAMPLE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
