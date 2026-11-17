"""Clarify the frozen QA contract without changing evidence or answer positions."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib,json,math,re,shutil,sys
from pathlib import Path

OLD_UNCERTAIN='The available views do not establish an answer.'
UNCERTAIN='Cannot be determined from the video up to this point.'
SCHEMA='virtualhome_qa290_eval_ready_20260922'

def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def revise(q,anchor):
    text=q['question'];subtype=q['question_subtype'];t=math.floor(q['query_time'])
    if subtype=='current_visibility_pair':
        text=re.sub(r'^At [\d.]+ seconds, which option correctly describes whether (.+) are in view\?$',
                    lambda m:f"At {anchor['observation_time']:g} seconds, are {m[1]} visible or out of view?",text)
    elif subtype=='earlier_visibility_pair':
        text=re.sub(r'^Looking back at ([\d.]+) seconds, which option correctly describes whether (.+) are in view\?$',
                    lambda m:f'At {m[1]} seconds earlier in the video, were {m[2]} visible or out of view?',text)
    elif subtype=='visible_room_category':
        text=f"What type of room is shown at {anchor['observation_time']:g} seconds?"
    elif subtype=='object_return_after_absence':
        text=text.replace('which has just reappeared after being completely out of view?',
                          f'which returned to view most recently by {t} seconds, after an earlier disappearance?')
    elif subtype=='observed_return_count':
        text=re.sub(r'^After ([\d.]+) seconds and up to this point, how many times has (.+) reappeared after being completely out of view\?$',
                    lambda m:f'After {m[1]} seconds and through {t} seconds, how many times did {m[2]} return to view after disappearing?',text)
    elif subtype=='observed_exit_count':
        text=re.sub(r'^Between ([\d.]+) and ([\d.]+) seconds, how many times did (.+) disappear completely from view\?$',
                    lambda m:f'After {m[1]} seconds and through {m[2]} seconds, how many times did {m[3]} leave view completely?',text)
    elif subtype=='observed_physical_state':
        text=re.sub(r'^At ([\d.]+) seconds, which description fits (.+)\?$',
                    lambda m:f'What was the physical state of {m[2]} at {m[1]} seconds?',text)
    elif subtype=='tracked_object_state':
        text=re.sub(r'^At ([\d.]+) seconds, which description fits (.+) seen earlier in the video\?$',
                    lambda m:f'For {m[2]} identified earlier in the video, what was its physical state at {m[1]} seconds?',text)
    elif subtype=='observed_state_sequence':
        text=re.sub(r'^At ([\d.]+) and ([\d.]+) seconds, which sequence describes the state of (.+), in that order\?$',
                    lambda m:f'What were the physical states of {m[3]} at {m[1]} seconds and {m[2]} seconds, respectively?',text)
    elif subtype=='latest_observed_visibility_transition':
        text=text.replace('which of the listed changes happened most recently?',
                          f'which listed change in visibility happened most recently by {t} seconds?')
    elif subtype=='last_two_observed_returns':
        text=text.replace('In the two most recent returns to view,',f'In the two most recent returns to view by {t} seconds,')
        text=text.replace('Count each reappearance after a complete disappearance;',
                          'A return follows an earlier disappearance;')
    elif subtype=='unobserved_action_count':
        text=text.replace('how many times was','how many times in total was')
    elif subtype=='unobserved_change_cause':
        text=text.replace('what caused the change in','what action caused the change in')
    else:
        raise ValueError(subtype)
    q['question']=text
    q['options']=[UNCERTAIN if x==OLD_UNCERTAIN else x.replace('in both views.','at both times.') for x in q['options']]
    return q

def context(q):
    return (f"Use only video observations at or before {q['query_time']:g} seconds. "
            'All times are measured from the start of the clip. '
            'Do not assume that an unseen physical state stayed unchanged or infer an unseen action from a typical routine. '
            'For visibility questions, out of view is a valid observed condition; it does not by itself make the answer unknown. '
            'For returns, exits, and their order, confirm a visibility change only when the new condition appears in two consecutive one-second samples. '
            'A physical state or action is known only when the available observations uniquely establish it.')

def copy_annotations(source, output):
    """Copy annotation files only; leave videos and review images external."""
    output.mkdir(parents=True)
    for filename in (
        'questions.jsonl', 'evaluation_gt.jsonl', 'model_inputs.jsonl',
        'evaluation_labels.json', 'evaluation_splits.json', 'video_manifest.json',
        'distribution_alignment.json', 'qa_revision_manifest.json',
        'private/qa_provenance.jsonl', 'private/physical_endpoint_review.json',
    ):
        path = source / filename
        if path.exists():
            target = output / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copytree(source / 'meta_data/qa_results', output / 'meta_data/qa_results')


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    if a.output.exists():raise ValueError(f'Output already exists: {a.output}')
    old=read_rows(a.source/'questions.jsonl');oldgt={r['id']:r for r in read_rows(a.source/'evaluation_gt.jsonl')}
    proof={r['question_id']:r for r in read_rows(a.source/'private/qa_provenance.jsonl')}
    copy_annotations(a.source, a.output)
    rows=[];gts=[];changes=[]
    for original in old:
        q=revise(deepcopy(original),proof[original['id']]['semantic_anchor'])
        unchanged=('id','episode_id','query_time','question_type','question_subtype','answer_index','evidence_spans','diagnostics','change_intensity')
        assert all(q[k]==original[k] for k in unchanged)
        assert len(set(q['options']))==4 and UNCERTAIN in q['options']
        assert (q['options'][q['answer_index']]==UNCERTAIN)==(q['diagnostics']['epistemic_status']=='uncertain')
        stem=f"{context(q)}\nQuestion: {q['question']}"
        q['question_stem']=stem
        q['qa_schema']=SCHEMA
        gt=deepcopy(oldgt[q['id']]);gt['question_stem']=stem;gt['options']=q['options'];gt['qa_schema']=SCHEMA
        choices='\n'.join(f'{chr(65+i)}. {option}' for i,option in enumerate(q['options']))
        gt['question']=f'{stem}\nOptions:\n{choices}\nAnswer with one option letter.'
        rows.append(q);gts.append(gt)
        changes.append(dict(id=q['id'],before=original['question'],after=q['question'],options_before=original['options'],options_after=q['options']))
    write_rows(a.output/'questions.jsonl',rows);write_rows(a.output/'evaluation_gt.jsonl',gts)
    write_rows(a.output/'model_inputs.jsonl',[{k:r[k] for k in ('id','episode_id','video_path','query_time','question','question_stem','options')} for r in gts])
    # Replace per-episode mirrors in the same directory contract.
    byepisode=defaultdict(list)
    for q in rows:byepisode[q['episode_id']].append(q)
    for path in (a.output/'meta_data/qa_results').rglob('*.json'):
        body=json.loads(path.read_text())
        if isinstance(body,list):
            episode=body[0]['episode_id'];body=byepisode[episode]
        else:
            episode=body.get('episode_id',path.stem)
            key='questions' if 'questions' in body else 'qa'
            if key not in body:raise ValueError(f'Unknown episode schema: {path}')
            body[key]=byepisode[episode]
            body['source_qa_schema']=body.get('qa_source')
            body['qa_source']=SCHEMA
        write_json(path,body)
    write_rows(a.output/'private/qa_wording_changes.jsonl',changes)
    counts=dict(questions=len(rows),episodes=len(byepisode),known=sum(q['diagnostics']['epistemic_status']=='known' for q in rows),
                uncertain=sum(q['diagnostics']['epistemic_status']=='uncertain' for q in rows),
                answer_positions=dict(Counter(q['answer_index'] for q in rows)),types=dict(Counter(q['question_type'] for q in rows)))
    assert counts['questions'] > 0
    assert len({q['id'] for q in rows}) == len(rows)
    assert all(len(q)==10 for q in byepisode.values())
    revision=dict(schema=SCHEMA,source=str(a.source),source_gt_sha256=sha(a.source/'evaluation_gt.jsonl'),
                  qa_sha256=sha(a.output/'questions.jsonl'),gt_sha256=sha(a.output/'evaluation_gt.jsonl'),
                  counts=counts,answer_indices_changed=0,query_times_changed=0,options_reordered=0,
                  model_predictions_used_for_label_changes=False,full_rgb_semantic_review='Inherited partial review; not claimed complete')
    write_json(a.output/'qa_revision_manifest.json',revision)
    for name in ('evaluator_contract_validation.json','video_verification.json'):
        f=a.output/name
        if f.exists():f.unlink()
    write_json(a.output/'release_status.json',dict(schema=SCHEMA,qa_revision='complete',video_materialization='pending',evaluation='not_run',full_rgb_semantic_review='partial'))
    write_json(a.output/'validation.json',dict(passed=True,checks=['Unique question IDs','Ten questions per episode','Answer indices and evidence preserved','No duplicate options','Uncertainty option on all rows'],counts=counts))
    write_json(a.output/'summary.json',counts)
    from .qa_export import distribution
    alignment=json.loads((a.source/'distribution_alignment.json').read_text())
    durations={q['episode_id']:proof[q['id']]['provenance']['duration_seconds'] for q in rows}
    alignment['current']=distribution(rows,durations=durations)
    for key,left in alignment['reference']['marginals'].items():
        right=alignment['current']['marginals'].get(key,{})
        cells={v:dict(reference_count=left.get(v,0),current_count=right.get(v,0),
                      reference_probability=left.get(v,0)/alignment['reference']['count'],
                      current_probability=right.get(v,0)/len(rows)) for v in sorted(set(left)|set(right))}
        alignment['comparison'][key]=dict(total_variation=sum(abs(v['reference_probability']-v['current_probability']) for v in cells.values())/2,cells=cells)
    write_json(a.output/'distribution_alignment.json',alignment)
    write_json(a.output/'evaluation_protocol.json',dict(schema=SCHEMA,gt_file='evaluation_gt.jsonl',video_root='.',fps=1,online='Causal prefix through query_time',offline='Full video; question time restriction still applies',uncertainty_option=UNCERTAIN,diagnostic_prompt='v3_separate_options',model_input_fields=['question','video_path','query_time']))
    write_json(a.output/'EPISODE_SAMPLE.json',byepisode[rows[0]['episode_id']])
    lines=[f'# VirtualHome: {len(rows)} revised questions','']
    for episode,questions in byepisode.items():
        lines+=['## '+episode,'']
        for i,q in enumerate(questions,1):
            lines += [f"### {i}. {q['question']} ({q['query_time']:g}s)",'']+[f"- {chr(65+j)}. {s}" for j,s in enumerate(q['options'])]+['',f"Answer: {chr(65+q['answer_index'])}",'']
    (a.output/'ALL_QA.md').write_text('\n'.join(lines))
    (a.output/'QA_REBUILD_REPORT.md').write_text('# QA revision for evaluation\n\n'+json.dumps(revision,indent=2)+'\n\nThis revision clarifies temporal scope, physical state versus visibility, and the sampled transition definition. Labels, options order, distributions, and evidence remain unchanged. Status/source prompts use question_stem and separate answer vocabularies. It does not claim a complete RGB semantic review or eliminate all question-type priors.\n')
    from .validation import validate
    write_json(a.output/'validation.json', validate(a.output))
    print(json.dumps(revision,indent=2))
if __name__=='__main__':main()
