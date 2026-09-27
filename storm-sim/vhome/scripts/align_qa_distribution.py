"""Build a four-choice QA revision using a frozen reference and capture evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from storm_virtualhome.aligned_qa import apportion, build_pool, select_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if (args.output.parent / 'data/qa_results/questions.jsonl').exists():
        raise RuntimeError('This evaluation dataset is frozen; use a separate revision directory.')
    bundle = json.loads(args.bundle.read_text())
    reference = bundle['reference_rows']
    assert set(len(q['options']) for q in reference) == {4}
    quotas = apportion(reference, 10 * len(bundle['episodes']))
    pools = [build_pool(ep) for ep in bundle['episodes']]
    availability = {ep['metadata']['episode_id']: dict(Counter(q['question_type'] for q in pool))
                    for ep, pool in zip(bundle['episodes'], pools)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'candidate_availability.json').write_text(json.dumps(availability, indent=2))
    print(json.dumps(dict(candidates=sum(map(len, pools)), quotas=quotas)), flush=True)
    rows = select_batch(pools, quotas)
    assert all(len(q['options']) == 4 and 0 <= q['answer_index'] < 4 for q in rows)
    assert len({q['id'] for q in rows}) == len(rows)
    target = args.output / 'polished_questions.jsonl'
    target.write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in rows))
    report = dict(reference=bundle['reference'], reference_sha256=bundle['reference_sha256'],
        reference_questions=len(reference), reference_arities=dict(Counter(len(q['options']) for q in reference)),
        reference_types=dict(Counter(q['question_type'] for q in reference)),
        reference_status=dict(Counter(q['diagnostics']['epistemic_status'] for q in reference)),
        questions=len(rows), episodes=len(pools), arities=dict(Counter(len(q['options']) for q in rows)),
        types=dict(Counter(q['question_type'] for q in rows)), uniform_random_accuracy=.25,
        allocation='Largest remainder from reference proportions; 10 questions per room.',
        semantic_answers={k: dict(Counter(q['semantic_answer'] for q in rows if q['question_type']==k)) for k in quotas},
        source_manifest_sha256=bundle['manifest_sha256'],
        polished_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        model_predictions_used=False, videos_changed=False,
        uncertainty='Known-answer-only revision; choice and type alignment does not imply matching reference status labels.',
        review='Graph states and visible stage frames checked; full perceptual answer review pending.',
        small_target_questions=sum(min(o['pixels'] for o in q['provenance']['observations'])<64 for q in rows),
        offscreen_tracking_questions=sum(q['question_type']=='object_tracking' for q in rows))
    (args.output / 'alignment_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
