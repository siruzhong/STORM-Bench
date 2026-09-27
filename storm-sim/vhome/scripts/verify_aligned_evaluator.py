"""Check the exported QA against the installed evaluator and source masks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('visibility',type=Path)
    parser.add_argument('evaluator',type=Path)
    args=parser.parse_args()
    sys.path.insert(0,str(args.evaluator))
    from scripts.eval.data_io import normalize_mcq_sample
    from scripts.eval.storm_streaming import IncrementalVideoSource, diagnostic_cell
    read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    questions=read(args.dataset/'questions.jsonl')
    canonical=read(args.dataset/'evaluation_gt.jsonl')
    ledger=read(args.dataset/'private/qa_provenance.jsonl')
    masks={x['episode_id']:x for x in json.loads(args.visibility.read_text())['episodes']}
    clocks={}; hidden=0; witnesses=0
    for q,row,proof in zip(questions,canonical,ledger):
        assert normalize_mcq_sample(row)==row
        assert row['answer']==q['answer_index'] and row['options']==q['options']
        assert diagnostic_cell(row)[1]==q['diagnostics']['epistemic_status']
        eid=q['episode_id']; mask=masks[eid]
        if eid not in clocks:
            clock=IncrementalVideoSource(args.dataset/row['video_path'],1.)
            clocks[eid]=dict(zip(clock._timestamps,clock._indices))
        actual=clocks[eid]
        for o in proof['provenance']['observations']:
            assert actual[o['time']]==o['frame'] and o['time']<=q['query_time']
            assert any(a<=o['time']<b for a,b in q['evidence_spans'])
            if o['object_id'] is not None:
                pixels=mask['counts'][o['frame']][mask['objects'].index(o['object_id'])]
                assert pixels==o['pixels']
                if o['visibility']=='visible':assert pixels>=128
                if o['visibility']=='absent':assert pixels==0
            witnesses+=1
        if q['diagnostics']['epistemic_status']=='uncertain':
            anchor=proof['semantic_anchor']; lo,hi=anchor['native_hidden_frames']
            col=mask['objects'].index(anchor['object'])
            assert all(frame[col]==0 for frame in mask['counts'][lo:hi])
            hidden+=1
    report=dict(questions=len(questions),episodes=len(clocks),verified_frame_witnesses=witnesses,
        verified_hidden_questions=hidden,actual_evaluator_normalization='passed',
        actual_evaluator_clock='passed',diagnostic_cells='passed',native_mask_evidence='passed',
        option_order='unchanged',full_rgb_semantic_review='not_performed_by_this_check',
        evaluation_gt_sha256=hashlib.sha256((args.dataset/'evaluation_gt.jsonl').read_bytes()).hexdigest())
    (args.dataset/'evaluator_contract_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
