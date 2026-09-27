"""Read QA datasets, write evaluation bundles, and check field consistency."""
from __future__ import annotations

import errno
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
    os.replace(temporary, path)


def materialize(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError as exc:
        if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES):
            raise
        shutil.copy2(source, target)


def load_documents(root):
    root = Path(root).resolve()
    result = []
    for path in sorted((root / 'meta_data/qa_results').glob('*/*.json')):
        document = read_json(path)
        # Resolve moved datasets against their local video directory first.
        local = root / 'meta_data/gene_videos' / path.parent.name / f"{document['episode_id']}.mp4"
        given = Path(document['video_path'])
        video = local if local.is_file() else (given if given.is_absolute() else root / given)
        if not video.is_file() or not video.stat().st_size:
            raise ValueError(f'missing video for {path}')
        result.append((path.relative_to(root), document, video.resolve()))
    if not result:
        raise ValueError(f'no QA documents under {root}/meta_data/qa_results')
    validate_questions(result)
    return result


def validate_questions(documents):
    seen, episodes = set(), set()
    for _, document, _ in documents:
        eid = document['episode_id']
        if eid in episodes:
            raise ValueError(f'duplicate episode: {eid}')
        episodes.add(eid)
        duration = float(document['duration_sec'])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'invalid duration: {eid}')
        if not document['questions']:
            raise ValueError(f'empty questions: {eid}')
        for q in document['questions']:
            qid = q['id']
            if qid in seen or q['episode_id'] != eid:
                raise ValueError(f'duplicate ID or mismatched episode: {qid}')
            seen.add(qid)
            query = float(q['query_time'])
            if not math.isfinite(query) or not 0 <= query < duration:
                raise ValueError(f'query outside video: {qid}')
            if not isinstance(q['question'], str) or not q['question'].strip():
                raise ValueError(f'empty question: {qid}')
            options = q['options']
            if len(options) != 4 or any(not isinstance(x, str) or not x.strip() for x in options) or len(set(options)) != 4:
                raise ValueError(f'invalid options: {qid}')
            if type(q['answer_index']) is not int or not 0 <= q['answer_index'] < 4:
                raise ValueError(f'invalid answer: {qid}')
            if q['diagnostics']['epistemic_status'] not in ('known', 'uncertain'):
                raise ValueError(f'invalid epistemic status: {qid}')
            if not q['evidence_spans']:
                raise ValueError(f'empty evidence: {qid}')
            for start, end in q['evidence_spans']:
                if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start <= end <= query:
                    raise ValueError(f'evidence extends outside visible prefix: {qid}')
    return seen


def model_prompt(q):
    from tools.storm.benchgen.stream_eqa.text_debiased_qa import build_model_prompt
    return build_model_prompt(q)


def write_bundle(output, documents, *, parent, audit):
    output = Path(output)
    questions, inputs, labels = [], [], {}
    for relative, document, video in documents:
        target = output / 'meta_data/gene_videos' / relative.parent.name / video.name
        materialize(video, target)
        document = dict(document, video_path=target.relative_to(output).as_posix())
        write_json(output / relative, document)
        for q in document['questions']:
            questions.append(q)
            status = q['diagnostics']['epistemic_status']
            inputs.append(dict(question_id=q['id'], episode_id=q['episode_id'], video_path=document['video_path'],
                               query_time=q['query_time'], prompt=model_prompt(q)))
            labels[q['id']] = dict(answer_index=q['answer_index'], question_type=q['question_type'],
                                   epistemic_status=status,
                                   split='primary_visual_known' if status == 'known' else 'epistemic_uncertain')
    write_jsonl(output / 'questions.jsonl', questions)
    write_jsonl(output / 'model_inputs.jsonl', inputs)
    write_json(output / 'evaluation_labels.json', labels)
    write_json(output / 'evaluation_splits.json', {
        name: [qid for qid, label in labels.items() if label['split'] == name]
        for name in ('primary_visual_known', 'epistemic_uncertain')})
    write_json(output / 'evaluation_protocol.json', {
        'video_frame_rule': 'timestamp <= query_time', 'full_video_offline_allowed': False,
        'video_path_base': 'dataset root',
        'private_fields': ['answer_index', 'video_evidence', 'evidence_spans', 'diagnostics', 'diagnostic_rationale'],
        'required_conditions': ['text_only', 'matched_video', 'shuffled_or_mismatched_video'],
        'headline_split': 'primary_visual_known'})
    write_json(output / 'summary.json', {
        'version': 'vlm_polish_v1', 'parent_dataset': str(parent),
        'candidate_status': 'polished_candidate_requires_semantic_review_and_reevaluation',
        'episode_count': len(documents), 'question_count': len(questions),
        'question_type_counts': dict(Counter(q['question_type'] for q in questions)),
        'epistemic_status_counts': dict(Counter(q['diagnostics']['epistemic_status'] for q in questions)),
        'answer_index_counts': dict(Counter(q['answer_index'] for q in questions)),
        'polish_counts': dict(Counter(r['status'] for r in audit))})
    write_jsonl(output / 'polish_audit.jsonl', audit)


def validate_bundle(root):
    root = Path(root).resolve()
    documents = load_documents(root)
    questions = [q for _, d, _ in documents for q in d['questions']]
    flat = [json.loads(line) for line in (root / 'questions.jsonl').read_text().splitlines() if line.strip()]
    if flat != questions:
        raise ValueError('questions.jsonl differs from per-episode documents')
    if (root / 'model_inputs.jsonl').exists():
        records = [json.loads(line) for line in (root / 'model_inputs.jsonl').read_text().splitlines() if line.strip()]
        labels = read_json(root / 'evaluation_labels.json')
        by_id = {q['id']: q for q in questions}
        if len(records) != len(questions) or {r['question_id'] for r in records} != set(by_id) or set(labels) != set(by_id):
            raise ValueError('model inputs or labels do not cover every question exactly once')
        for record in records:
            q = by_id[record['question_id']]
            if set(record) != {'question_id', 'episode_id', 'video_path', 'query_time', 'prompt'}:
                raise ValueError('model-input field allowlist violated')
            if record['prompt'] != model_prompt(q) or record['query_time'] != q['query_time'] or record['episode_id'] != q['episode_id']:
                raise ValueError('stale model input')
            p = Path(record['video_path'])
            if not (p if p.is_absolute() else root / p).is_file():
                raise ValueError('model input references missing video')
            if labels[q['id']]['answer_index'] != q['answer_index']:
                raise ValueError('stale evaluation label')
    return {'episode_count': len(documents), 'question_count': len(questions), 'structural_validation': 'passed'}
