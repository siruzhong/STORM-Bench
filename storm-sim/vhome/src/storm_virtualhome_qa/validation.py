"""Check final annotations, synchronized labels, and causal evidence timestamps."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .revision import UNCERTAIN, context


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root):
    root = Path(root)
    rows = read_rows(root / 'questions.jsonl')
    gt = read_rows(root / 'evaluation_gt.jsonl')
    inputs = read_rows(root / 'model_inputs.jsonl')
    ledger = read_rows(root / 'private/qa_provenance.jsonl')
    labels = json.loads((root / 'evaluation_labels.json').read_text())
    splits = json.loads((root / 'evaluation_splits.json').read_text())
    videos = json.loads((root / 'video_manifest.json').read_text())
    video_by_episode = {v['episode_id']: v for v in videos}
    ids = [q['id'] for q in rows]
    require(bool(rows) and len(set(ids)) == len(ids), 'Empty dataset or duplicate question IDs')
    require(len(rows) == len(gt) == len(inputs) == len(ledger) == len(labels), 'Row counts differ')
    require(set(labels) == set(ids), 'Label IDs differ')
    require(len(videos) == len(video_by_episode), 'Duplicate video manifest episodes')
    for q, target, inp, proof in zip(rows, gt, inputs, ledger):
        qid = q['id']
        require(qid == target['id'] == inp['id'] == proof['question_id'], f'ID mismatch: {qid}')
        require(len(q['options']) == len(set(q['options'])) == 4, f'Invalid options: {qid}')
        answer = q['answer_index']
        require(type(answer) is int and 0 <= answer < 4, f'Invalid answer index: {qid}')
        require(UNCERTAIN in q['options'], f'Missing uncertainty option: {qid}')
        status = q['diagnostics']['epistemic_status']
        require(status in ('known', 'uncertain'), f'Invalid epistemic status: {qid}')
        require((q['options'][answer] == UNCERTAIN) == (status == 'uncertain'), f'Uncertainty mismatch: {qid}')
        require(answer == target['answer'] == target['source_answer_index'] == labels[qid]['answer_index'],
                f'Answer mismatch: {qid}')
        require(target['option_permutation'] == list(range(4)), f'Unexpected option permutation: {qid}')
        require(q['options'] == target['options'], f'GT options differ: {qid}')
        for field in ('episode_id', 'query_time', 'question_type', 'question_subtype', 'diagnostics', 'change_intensity'):
            require(q[field] == target[field], f'GT field mismatch: {qid}/{field}')
        require(labels[qid]['epistemic_status'] == status and labels[qid]['question_type'] == q['question_type'],
                f'Label metadata differs: {qid}')
        expected_stem = f"{context(q)}\nQuestion: {q['question']}"
        require(q['question_stem'] == target['question_stem'] == expected_stem, f'Prompt stem differs: {qid}')
        choices = '\n'.join(f'{chr(65+i)}. {option}' for i, option in enumerate(q['options']))
        require(target['question'] == f'{expected_stem}\nOptions:\n{choices}\nAnswer with one option letter.',
                f'Prompt options differ: {qid}')
        allowed = ('id', 'episode_id', 'video_path', 'query_time', 'question', 'question_stem', 'options')
        require(inp == {k: target[k] for k in allowed}, f'Model input mismatch or private fields: {qid}')
        time = q['query_time']
        require(math.isfinite(time) and 0 <= time < proof['provenance']['duration_seconds'], f'Invalid time: {qid}')
        require(q['evidence_spans'] and all(0 <= a < b <= time for a, b in q['evidence_spans']),
                f'Noncausal evidence span: {qid}')
        observations = proof['provenance']['observations']
        require(observations and all(0 <= o['time'] <= time and o['time'] == int(o['time']) for o in observations),
                f'Noncausal or unsampled observation: {qid}')
        video = Path(target['video_path'])
        require(not video.is_absolute() and '..' not in video.parts, f'Nonportable video path: {qid}')
        require(target['video_path'] == video_by_episode[q['episode_id']]['video_path'], f'Video manifest mismatch: {qid}')
    for split, status in [('primary_visual_known', 'known'), ('epistemic_uncertain', 'uncertain')]:
        expected = [q['id'] for q in rows if q['diagnostics']['epistemic_status'] == status]
        require(splits[split] == expected, f'Split mismatch: {split}')
    episode_counts = Counter(q['episode_id'] for q in rows)
    require(set(episode_counts.values()) == {10}, 'Expected ten questions per episode')
    require(set(episode_counts) == set(video_by_episode), 'Video manifest episode coverage differs')
    mirrors = {}
    for path in (root / 'meta_data/qa_results').glob('*/*.json'):
        body = json.loads(path.read_text())
        require(body['episode_id'] not in mirrors, 'Duplicate episode mirror')
        mirrors[body['episode_id']] = body['questions']
    require(set(mirrors) == set(episode_counts), 'Episode mirror coverage differs')
    for episode, questions in mirrors.items():
        require(questions == [q for q in rows if q['episode_id'] == episode], f'Episode mirror differs: {episode}')
    positions = [sum(q['answer_index'] == i for q in rows) for i in range(4)]
    require(max(positions) - min(positions) <= 1, 'Answer positions are not balanced')
    return dict(passed=True, questions=len(rows), episodes=len(episode_counts),
                known=sum(q['diagnostics']['epistemic_status'] == 'known' for q in rows),
                uncertain=sum(q['diagnostics']['epistemic_status'] == 'uncertain' for q in rows),
                answer_positions=positions, video_verification='not_performed',
                rgb_semantic_review='not_performed_by_this_validator')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.dataset), indent=2))
