"""Export evidence-grounded questions using the AI2THOR dataset contract."""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from .visual_qa import SCHEMA, UNDETERMINED

PUBLIC_FIELDS = ('id', 'episode_id', 'query_time', 'question_type', 'question_subtype',
                 'video_evidence', 'question', 'options', 'answer_index', 'evidence_spans',
                 'diagnostics', 'diagnostic_rationale', 'change_intensity')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(q, ensure_ascii=False, allow_nan=False) + '\n' for q in rows))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def balanced_positions(rows, *, seed=20260921):
    """Balance displayed answers globally and within type, status, and episode."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    count = len(rows)
    groups = defaultdict(list)
    for i, q in enumerate(rows):
        kind, status = q['question_type'], q['diagnostics']['epistemic_status']
        for key in [('all',), ('type', kind), ('status', status),
                    ('type_status', kind, status), ('episode', q['episode_id'])]:
            groups[key].append(i)
    rr, cc, vv, lower, upper = [], [], [], [], []
    def constraint(indices, lo, hi):
        rr.extend([len(lower)] * len(indices)); cc.extend(indices); vv.extend([1.] * len(indices))
        lower.append(lo); upper.append(hi)
    for i in range(count):
        constraint(list(range(4*i, 4*i+4)), 1, 1)
    for key, indices in groups.items():
        for position in range(4):
            if key == ('all',):
                lo = hi = count//4 + (position < count % 4)
            else:
                lo, hi = len(indices)//4, math.ceil(len(indices)/4)
            constraint([4*i+position for i in indices], lo, hi)
    matrix = coo_matrix((vv, (rr, cc)), shape=(len(lower), 4*count)).tocsc()
    cost = [random.Random(f'{seed}:{q["id"]}:{j}').random() for q in rows for j in range(4)]
    result = milp(cost, integrality=np.ones(4*count), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lower, upper),
                  options=dict(time_limit=30, mip_rel_gap=.05))
    if result.x is None:
        raise ValueError('Cannot balance option positions under the declared constraints')
    revised = deepcopy(rows)
    for i, q in enumerate(revised):
        target = int(np.argmax(result.x[4*i:4*i+4]))
        correct = q['options'][q['answer_index']]
        others = [s for s in q['options'] if s != correct]
        random.Random(f'{seed}:{q["id"]}:distractors').shuffle(others)
        others.insert(target, correct)
        q['options'], q['answer_index'] = others, target
    return revised


def model_prompt(q):
    choices = '\n'.join(f'{chr(65+i)}. {text}' for i, text in enumerate(q['options']))
    return f"Question: {q['question']}\nOptions:\n{choices}\nAnswer with one option letter."


def distribution(rows, *, durations=None):
    """Report categorical marginals, joint counts, and text/time summaries."""
    durations = durations or {}
    counters = defaultdict(Counter)
    lengths = defaultdict(list)
    uncertainty_texts = {q['options'][q['answer_index']] for q in rows
                         if q['diagnostics']['epistemic_status']=='uncertain'} | {UNDETERMINED}
    for q in rows:
        status = q['diagnostics']['epistemic_status']
        kind = q['question_type']
        has_uncertainty = any(option in uncertainty_texts for option in q['options'])
        fields = dict(question_type=kind, question_subtype=q['question_subtype'],
            epistemic_status=status, type_status=f'{kind}|{status}',
            option_count=len(q['options']), answer_index=q['answer_index'],
            type_answer=f'{kind}|{q["answer_index"]}', status_answer=f'{status}|{q["answer_index"]}',
            uncertainty_sources='|'.join(sorted(q['diagnostics']['uncertainty_sources'])) or '(none)',
            type_uncertainty_sources=kind+'|' + ('|'.join(sorted(q['diagnostics']['uncertainty_sources'])) or '(none)'),
            uncertainty_option=has_uncertainty, status_uncertainty_option=f'{status}|{has_uncertainty}',
            evidence_span_count=len(q['evidence_spans']), change_intensity=q['change_intensity'],
            status_intensity=f'{status}|{q["change_intensity"]}',
            normalized_query_quintile=min(4, int(5*q['query_time']/durations.get(q['episode_id'], 60.05))))
        for key, value in fields.items():
            counters[key][str(value)] += 1
        for key in q:
            counters['field_presence'][key] += 1
        for key in ('question', 'video_evidence'):
            lengths[key+'_words'].append(len(q[key].split()))
        lengths['option_words'].extend(len(s.split()) for s in q['options'])
        lengths['query_time_seconds'].append(q['query_time'])
        lengths['evidence_duration_seconds'].append(sum(b-a for a,b in q['evidence_spans']))
    return dict(count=len(rows), marginals={key: dict(sorted(value.items())) for key,value in counters.items()},
                numeric={key: dict(min=min(value), max=max(value), mean=sum(value)/len(value))
                         for key,value in lengths.items()})


def compare_distributions(reference, current, *, durations):
    left, right = distribution(reference), distribution(current, durations=durations)
    comparison = {}
    for key in left['marginals']:
        a, b = left['marginals'][key], right['marginals'][key]
        cells = {value: dict(reference_count=a.get(value,0), current_count=b.get(value,0),
                     reference_probability=a.get(value,0)/len(reference), current_probability=b.get(value,0)/len(current))
                 for value in sorted(set(a) | set(b))}
        comparison[key] = dict(total_variation=.5*sum(abs(c['reference_probability']-c['current_probability']) for c in cells.values()),
                               cells=cells)
    return dict(reference=left, current=right, comparison=comparison,
                probability_unit='questions; option_words uses individual options',
                reference_duration_seconds=60.05)


def export_dataset(rows, episodes, reference, output, *, reference_sha256, selection,
                   uncertainty_policy='reference'):
    """Write portable annotations and a separate private provenance ledger."""
    output = Path(output)
    if uncertainty_policy not in ('reference', 'all_questions'):
        raise ValueError(f'Unknown uncertainty option policy: {uncertainty_policy}')
    schema = SCHEMA + '_reference_options_v2' if uncertainty_policy=='reference' else SCHEMA
    for q in rows:
        expected = uncertainty_policy=='all_questions' or q['diagnostics']['epistemic_status']=='uncertain'
        if (UNDETERMINED in q['options']) != expected:
            raise ValueError(f'Question does not follow the declared option policy: {q["id"]}')
    by_episode = defaultdict(list)
    for q in balanced_positions(rows):
        by_episode[q['episode_id']].append(q)
    public, inputs, canonical, ledger, videos = [], [], [], [], []
    labels = {}
    durations = {e['metadata']['episode_id']: e['metadata']['duration_seconds'] for e in episodes}
    for episode in sorted(episodes, key=lambda e: (f"Scene{e['metadata']['scene']}_Room{e['metadata']['room_id']}", e['metadata']['episode_id'])):
        meta = episode['metadata']; eid = meta['episode_id']
        scene = f"Scene{meta['scene']}_Room{meta['room_id']}"
        video = f'meta_data/gene_videos/{scene}/{eid}.mp4'
        questions = []
        for original in sorted(by_episode[eid], key=lambda q:(q['query_time'],q['id'])):
            q = {key: original[key] for key in PUBLIC_FIELDS}
            questions.append(q); public.append(q)
            prompt = model_prompt(q)
            inputs.append(dict(question_id=q['id'], episode_id=eid, video_path=video,
                               query_time=q['query_time'], prompt=prompt))
            status = q['diagnostics']['epistemic_status']
            labels[q['id']] = dict(answer_index=q['answer_index'], question_type=q['question_type'],
                epistemic_status=status, split='primary_visual_known' if status=='known' else 'epistemic_uncertain')
            # Canonical evaluator rows bypass its legacy automatic option permutation.
            canonical.append(dict(id=q['id'], episode_id=eid, video_path=video,
                query_time=q['query_time'], question=f"At {q['query_time']:.2f} seconds, use only observations up to that point.\n"+prompt,
                options=q['options'], answer=q['answer_index'], source_answer_index=q['answer_index'],
                option_permutation=list(range(4)), question_type=q['question_type'], a_type=q['question_type'],
                question_subtype=q['question_subtype'], change_intensity=q['change_intensity'],
                diagnostics=q['diagnostics']))
            ledger.append(dict(question_id=q['id'], qa_schema=schema,
                semantic_anchor=original['semantic_anchor'], provenance=original['provenance'],
                review_status=original['review_status']))
        assert len(questions) == 10
        write_json(output/f'meta_data/qa_results/{scene}/{eid}.json', dict(
            episode_id=eid, video_path=video, duration_sec=meta['duration_seconds'], sample_fps=1.,
            qa_source=schema, sample_timestamps=list(range(math.floor(meta['duration_seconds']-.0001)+1)), questions=questions))
        videos.append(dict(episode_id=eid, video_path=video, sha256=meta['hashes']['video.mp4'],
                           source_filename=eid+'.mp4', room_type=meta['room']))
    write_jsonl(output/'questions.jsonl', public)
    write_jsonl(output/'model_inputs.jsonl', inputs)
    write_jsonl(output/'evaluation_gt.jsonl', canonical)
    write_jsonl(output/'private/qa_provenance.jsonl', ledger)
    write_json(output/'video_manifest.json', videos)
    write_json(output/'evaluation_labels.json', labels)
    write_json(output/'evaluation_splits.json', {split:[qid for qid,label in labels.items() if label['split']==split]
        for split in ('primary_visual_known','epistemic_uncertain')})
    write_json(output/'evaluation_protocol.json', dict(video_frame_rule='timestamp <= query_time',
        full_video_offline_allowed=False, required_conditions=['text_only','matched_video','shuffled_or_mismatched_video'],
        headline_split='primary_visual_known', minimum_matched_video_gain=.1, maximum_shuffled_video_gain=.03,
        video_path_base='dataset root', sampling_fps=1, evaluator_gt='evaluation_gt.jsonl',
        private_fields=['answer_index','video_evidence','evidence_spans','diagnostics','diagnostic_rationale','change_intensity'],
        offline_note='Full-video offline may be reported separately as a future-context diagnostic.'))
    stats = compare_distributions(reference,public,durations=durations)
    write_json(output/'distribution_alignment.json', stats)
    write_json(output/'qa_revision_manifest.json', dict(schema=schema, reference_sha256=reference_sha256,
        questions_sha256=sha256(output/'questions.jsonl'), evaluation_gt_sha256=sha256(output/'evaluation_gt.jsonl'),
        selection=selection, model_predictions_used=False, video_content_changed=False,
        uncertainty_option_policy=uncertainty_policy,
        change_intensity_definition='max(1, count of injected events with start <= query_time + 1e-6), matching the AI2THOR generator.',
        source_tag_policy='Tags describe evidence limitations; unsupported source combinations are not assigned to meet a quota.',
        stability_rule='Two consecutive 1 FPS samples; visible >=128 target pixels, absent =0; intermediate counts are unclassified.',
        known_answer_basis='Sampled visibility and room context. No visually unreadable appliance power state is treated as known.'))
    write_json(output/'summary.json', dict(version=schema, episode_count=len(videos), question_count=len(public),
        question_type_counts=dict(Counter(q['question_type'] for q in public)),
        epistemic_status_counts=dict(Counter(q['diagnostics']['epistemic_status'] for q in public)),
        answer_index_counts=dict(Counter(q['answer_index'] for q in public)),
        room_type_counts=dict(Counter(v['room_type'] for v in videos)),
        uniform_random_baseline=.25, empirical_accuracy=None,
        candidate_status='structurally_aligned_candidate; complete RGB semantic review and evaluation pending'))
    write_json(output/'release_status.json', dict(annotation_export='complete', video_materialization='pending',
        structural_validation='pending', full_rgb_semantic_review='pending', evaluation='not_run'))
    return public, stats


def validate_annotation_contract(root):
    """Validate causal timing, public/private separation, and synchronized labels."""
    root = Path(root)
    read_rows = lambda path:[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    rows = read_rows(root/'questions.jsonl')
    docs = [json.loads(p.read_text()) for p in sorted((root/'meta_data/qa_results').glob('*/*.json'))]
    assert rows == [q for d in docs for q in d['questions']]
    assert len({q['id'] for q in rows}) == len(rows)
    inputs = read_rows(root/'model_inputs.jsonl')
    canonical = read_rows(root/'evaluation_gt.jsonl')
    ledger = read_rows(root/'private/qa_provenance.jsonl')
    labels = json.loads((root/'evaluation_labels.json').read_text())
    policy = json.loads((root/'qa_revision_manifest.json').read_text())['uncertainty_option_policy']
    assert len(rows)==len(inputs)==len(canonical)==len(ledger)==len(labels)
    for q, inp, gt, evidence in zip(rows, inputs, canonical, ledger):
        assert set(q)==set(PUBLIC_FIELDS)
        assert q['id']==inp['question_id']==gt['id']==evidence['question_id']
        assert len(q['options'])==len(set(q['options']))==4
        correct=q['options'][q['answer_index']]
        assert (correct==UNDETERMINED)==(q['diagnostics']['epistemic_status']=='uncertain')
        assert (UNDETERMINED in q['options']) == (policy=='all_questions' or q['diagnostics']['epistemic_status']=='uncertain')
        assert labels[q['id']]['answer_index']==gt['answer']==q['answer_index']
        assert 'answer_index' not in gt and gt['options']==q['options']
        assert set(inp)=={'question_id','episode_id','video_path','query_time','prompt'}
        assert inp['prompt']==model_prompt(q)
        assert all(0<=a<b<=q['query_time'] for a,b in q['evidence_spans'])
        assert all(o['time']<=q['query_time'] and o['time']==int(o['time']) for o in evidence['provenance']['observations'])
    positions=Counter(q['answer_index'] for q in rows)
    assert max(positions.values())-min(positions.values())<=1
    return dict(question_count=len(rows), episode_count=len(docs), structural_validation='passed',
                video_verification='separate', causal_clock_checks='passed', public_private_separation='passed')
