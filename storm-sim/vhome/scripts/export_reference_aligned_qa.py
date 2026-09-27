"""Build and export QA distributions against a frozen AI2THOR reference."""
import argparse
import json
from pathlib import Path

from storm_virtualhome.qa_export import export_dataset, sha256, validate_annotation_contract, write_json, write_jsonl
from storm_virtualhome.visual_qa import build_visual_pool, select_visual_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--visibility', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--reuse-selection', action='store_true')
    parser.add_argument('--uncertainty-policy', choices=['reference', 'all_questions'], default='reference')
    args = parser.parse_args()
    status_path = args.output/'release_status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('evaluation') in ('running', 'complete'):
        raise ValueError('Cannot replace an evaluated dataset. Choose a new output directory.')
    bundle = json.loads(args.evidence.read_text())
    visibility = {v['episode_id']:v for v in json.loads(args.visibility.read_text())['episodes']}
    reference = [json.loads(line) for line in args.reference.read_text().splitlines() if line.strip()]
    pool = [q for ep in bundle['episodes'] for q in build_visual_pool(
        ep,visibility[ep['metadata']['episode_id']],uncertainty_policy=args.uncertainty_policy)]
    if args.reuse_selection:
        ids = {json.loads(s)['id'] for s in (args.work/'visual_selected.jsonl').read_text().splitlines()}
        selected = [q for q in pool if q['id'] in ids]
        if len(selected)!=len(ids):
            raise ValueError('The saved selection is incompatible with the regenerated candidate pool')
        quotas = json.loads((args.work/'visual_quotas.json').read_text())
    else:
        selected, quotas = select_visual_batch(pool,reference)
    write_jsonl(args.work/'visual_selected.jsonl', selected)
    write_json(args.work/'visual_quotas.json', quotas)
    public, stats = export_dataset(selected,bundle['episodes'],reference,args.output,
        reference_sha256=sha256(args.reference),selection=quotas,uncertainty_policy=args.uncertainty_policy)
    result = validate_annotation_contract(args.output)
    write_json(args.output/'validation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
