"""Rebuild 290 questions from observed facts and reviewed physical-state frames."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from .aligned_qa import name
from .visual_qa import build_visual_pool, UNDETERMINED
from .qa_export import export_dataset, validate_annotation_contract, write_json, write_jsonl, sha256

from . import language

SCHEMA = 'virtualhome_evidence_rebuilt_20260922'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def state_domain(event):
    if event['verb'] in ('Open', 'Close'):
        return ['Fully closed.', 'Fully open.', 'Partly open.'], {'closed': 'Fully closed.', 'open': 'Fully open.'}
    if event['verb'] in ('SwitchOn', 'SwitchOff'):
        return ['On.', 'Off.', 'Being switched between on and off.'], {'on': 'On.', 'off': 'Off.'}
    return ['Held by a person.', 'Resting on a surface.', 'Lying on the floor.'], {
        'held by a person': 'Held by a person.', 'resting on a surface': 'Resting on a surface.'}


def make_row(ep, vis, kind, subtype, query, question, concrete, correct, observations, anchor,
             status='known', sources=(), reviewed=False):
    query = round(query, 6)
    assert max(o['time'] for o in observations) <= query < ep['metadata']['duration_seconds']
    q = dict(episode_id=ep['metadata']['episode_id'], query_time=query, question_type=kind,
             question_subtype=subtype, question=question, options=[*concrete, UNDETERMINED],
             answer_index=([*concrete, UNDETERMINED]).index(correct),
             evidence_spans=[[min(o['time'] for o in observations), min(query, max(o['time'] for o in observations)+.05)]],
             video_evidence='Use sampled video observations no later than query_time.',
             diagnostics=dict(epistemic_status=status, uncertainty_sources=list(sources)),
             diagnostic_rationale=dict(volatility='Object observations and physical states can change over time.',
                 uncertainty='The referenced RGB frames establish the answer.' if reviewed else
                 'The queried physical state is not visible; the video does not determine the answer.'),
             change_intensity=max(1, sum(e['start'] <= query+1e-6 for e in ep['events'])),
             semantic_anchor=anchor, qa_schema=SCHEMA,
             provenance=dict(events_sha256=ep['events_sha256'], frame_manifest_sha256=vis['manifest_sha256'],
                 observations=observations, sampling_fps=1, duration_seconds=ep['metadata']['duration_seconds'],
                 minimum_target_pixels=128, answer_basis='reviewed_rgb_physical_state' if reviewed else 'unobserved_physical_state'),
             review_status='endpoint_rgb_reviewed' if reviewed else 'hidden_interval_verified')
    q['id'] = q['episode_id'] + '_rqa_' + digest([kind, subtype, query, anchor, question])
    return q


def observation(ep, vis, obj, t, label):
    frame = round(t*vis['fps'])
    pixels = vis['counts'][frame][vis['objects'].index(obj)]
    return dict(object_id=obj, object_name=label, time=float(t), frame=frame, pixels=pixels,
                visibility='visible' if pixels >= 128 else 'absent' if pixels == 0 else 'ambiguous')


def physical_pool(episodes, visibility, review):
    pool = []
    for item in review:
        if not item['approved']:
            continue
        ep = episodes[item['episode_id']]; vis = visibility[item['episode_id']]
        obj = item['object_id']; label = name(item['name'])
        event = next(e for e in ep['events'] if e['object_id'] == obj)
        domain = state_domain(event)
        if event['object'] == 'television':
            # Describe visible screen content, not an unobservable electrical state.
            concrete = ['The screen is dark.', 'The screen displays a picture.', 'The screen displays static.']
            mapping = {'off': concrete[0], 'on': concrete[1]}
        else:
            concrete, mapping = domain
        examples = sorted(item['examples'], key=lambda x: x['time'])
        early, late = examples
        obs = [observation(ep, vis, obj, x['time'], label) for x in examples]
        assert all(o['pixels'] >= 128 for o in obs)
        proof = dict(object=obj, reviewed_frames=[x['frame'] for x in examples],
                     review_sheet_index=item['sheet_index'], states=[mapping[x['state']] for x in examples])
        for x, o in zip(examples, obs):
            for kind in ('current_state', 'factual_retrieval'):
                query = x['time']+.05 if kind == 'current_state' else min(x['time']+8.05, ep['metadata']['duration_seconds']-.1)
                if kind == 'factual_retrieval' and query-x['time'] < 5:
                    continue
                question = f"At {x['time']:g} seconds, which description fits the {label}?"
                pool.append(make_row(ep, vis, kind, 'observed_physical_state', query, question,
                    concrete, mapping[x['state']], [o], {**proof, 'target_time': x['time']}, reviewed=True))
        states = [mapping[x['state']].rstrip('.').lower() for x in examples]
        forward = f'{states[0].capitalize()}, then {states[1]}.'
        reverse = f'{states[1].capitalize()}, then {states[0]}.'
        unchanged = f'{states[0].capitalize()} in both views.'
        for kind in ('state_change', 'temporal_reasoning'):
            question = (f"At {early['time']:g} and {late['time']:g} seconds, "
                        f'which sequence describes the state of the {label}, in that order?')
            pool.append(make_row(ep, vis, kind, 'observed_state_sequence', late['time']+.05,
                question, [forward, reverse, unchanged], forward, obs, proof, reviewed=True))
        question = f"At {late['time']:g} seconds, which description fits the {label} seen earlier in the video?"
        pool.append(make_row(ep, vis, 'object_tracking', 'tracked_object_state', late['time']+.05,
            question, concrete, mapping[late['state']], obs, {**proof,'target_time':late['time']}, reviewed=True))
    return pool


def exit_count_pool(episodes, visibility):
    pool=[]
    for eid,ep in episodes.items():
        vis=visibility[eid];fps=vis['fps']
        last=math.floor((vis['frames']-1)/fps)
        objects={event['object_id']:name(event['object_description']) for event in ep['events']}
        for obj,label in objects.items():
            obs=[observation(ep,vis,obj,t,label) for t in range(last+1)]
            previous=None;exits=[]
            for t in range(1,last+1):
                state=obs[t]['visibility']
                if state=='ambiguous' or obs[t-1]['visibility']!=state:continue
                if previous=='visible' and state=='absent':exits.append(t)
                previous=state
            for end in sorted(set(range(10,last+1,5))|set(exits)):
                for start in (0,max(0,end-20)):
                    times=[t for t in exits if start<t<=end]
                    if not times or end-start<8 or end+.05>=ep['metadata']['duration_seconds']:continue
                    count=min(3,len(times));values=['Zero times.','Once.','Twice.','Three or more times.']
                    correct=values[count]
                    other=[x for x in values if x!=correct]
                    random.Random(f'{eid}:{obj}:{start}:{end}').shuffle(other)
                    q=make_row(ep,vis,'history_aggregation','observed_exit_count',end+.05,
                        f'Between {start:g} and {end:g} seconds, how many times did the {label} disappear completely from view?',
                        [correct,*other[:2]],correct,obs[start:end+1],
                        dict(object=obj,start=start,return_times=times,count=count,event_kind='disappearance'))
                    q['provenance']['answer_basis']='sampled_visibility'
                    q['diagnostic_rationale']['uncertainty']='The sampled frames establish the observed disappearance count.'
                    q['review_status']='requires_rgb_review'
                    pool.append(q)
    return pool


def rewrite_uncertain(q, ep, vis):
    anchor = q['semantic_anchor']; obj = anchor['object']
    event = next(e for e in ep['events'] if e['event_index'] == anchor['event'])
    domain = state_domain(event)
    if domain is None:
        return None
    concrete, mapping = domain
    label = name(event['object_description'])
    lo, hi = anchor['hidden_interval']
    times = [t for t in range(math.ceil(lo), math.floor(hi)+1)
             if t < hi and vis['counts'][round(t*vis['fps'])][vis['objects'].index(obj)] == 0]
    if not times:
        return None
    target = times[len(times)//2]
    kind = q['question_type']
    # All uncertainty questions refer to a genuinely unobserved physical fact.
    if kind in ('current_state', 'factual_retrieval', 'object_tracking'):
        query = target+.05 if kind == 'current_state' else anchor['after']+.05
        question = f'At {target:g} seconds, which description fits the {label}?'
        if kind == 'object_tracking':
            question = f'At {target:g} seconds, which description fits the {label} seen earlier in the video?'
        proof = [observation(ep, vis, obj, anchor['before'], label), observation(ep, vis, obj, target, label)]
        if kind != 'current_state':
            proof.append(observation(ep, vis, obj, anchor['after'], label))
        return make_row(ep, vis, kind, 'tracked_object_state' if kind == 'object_tracking' else 'observed_physical_state',
                        query, question, concrete, UNDETERMINED, proof, {**anchor, 'target_time':target},
                        status='uncertain', sources=['missing_observation'])
    if kind == 'temporal_reasoning':
        first, second = [value.rstrip('.').lower() for value in concrete[:2]]
        choices = [f'{first.capitalize()}, then {second}.', f'{second.capitalize()}, then {first}.',
                   f'{first.capitalize()} in both views.']
        question = (f"At {anchor['before']:g} and {target:g} seconds, "
                    f'which sequence describes the state of the {label}, in that order?')
        proof = [observation(ep, vis, obj, anchor['before'], label), observation(ep, vis, obj, target, label)]
        return make_row(ep, vis, kind, 'observed_state_sequence', anchor['after']+.05, question,
                        choices, UNDETERMINED, proof, {**anchor,'target_time':target},
                        status='uncertain', sources=['missing_observation'])
    result = deepcopy(q)
    if kind == 'history_aggregation':
        action = 'opened or closed' if event['verb'] in ('Open', 'Close') else 'picked up or put down'
        result['question'] = f"Between {anchor['before']:g} and {anchor['after']:g} seconds, how many times was the {label} {action}?"
        result['options'] = ['Zero times.', 'Once.', 'Two or more times.', UNDETERMINED]
        result['answer_index'] = 3
        result['question_subtype'] = 'unobserved_action_count'
    else:
        result['question'] = f"Between {anchor['before']:g} and {anchor['after']:g} seconds, what caused the change in the {label}?"
        result['question_subtype'] = 'unobserved_change_cause'
    return result


def select(rows, quotas, *, relax=(), time_limit=90):
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    expected = sum(quotas['types'].values())
    episodes = {q['episode_id'] for q in rows}
    if expected != 10 * len(episodes):
        raise ValueError('Type quotas must sum to ten questions per episode')
    for kind, total in quotas['types'].items():
        unknown = quotas['uncertain'].get(kind, 0)
        if not 0 <= unknown <= total:
            raise ValueError(f'Invalid uncertainty quota for {kind}')
        for status, needed in [('known', total-unknown), ('uncertain', unknown)]:
            available = sum(q['question_type']==kind and q['diagnostics']['epistemic_status']==status for q in rows)
            if available < needed:
                raise ValueError(f'Insufficient candidates for {kind}/{status}: {available} < {needed}')
    groups = defaultdict(list)
    for i,q in enumerate(rows):
        eid=q['episode_id'];kind=q['question_type'];status=q['diagnostics']['epistemic_status'];a=q['semantic_anchor']
        for key in [('episode',eid),('type_status',kind,status),('episode_type',eid,kind),('episode_status',eid,status)]:
            groups[key].append(i)
        physical=q['provenance']['answer_basis']=='reviewed_rgb_physical_state'
        if physical:
            groups[('physical',)].append(i)
            groups[('physical_kind',kind)].append(i)
            groups[('physical_object',eid,a['object'])].append(i)
            fact=('physical',a['object'],a.get('target_time',tuple(a['reviewed_frames'])))
        elif status=='uncertain':
            fact=('hidden',a['event'])
        elif q['question_subtype']=='visible_room_category':
            groups[('room',)].append(i);fact=('room',)
        elif q['question_subtype'] in ('current_visibility_pair','earlier_visibility_pair'):
            groups[('pair',tuple(a['states']))].append(i)
            groups[('episode_pair',eid)].append(i)
            fact=('visibility',tuple(a['objects']),a['observation_time'])
        elif q['question_subtype'] in ('observed_return_count','observed_exit_count'):
            groups[('count',a['count'])].append(i)
            groups[('count_kind',q['question_subtype'])].append(i)
            groups[('episode_count',eid)].append(i)
            groups[('count_events',eid,q['question_subtype'],a['object'],tuple(a['return_times']))].append(i)
            fact=('returns',q['question_subtype'],a['object'],a['start'],tuple(a['return_times']))
        elif q['question_subtype']=='last_two_observed_returns':
            fact=('order',tuple(a['return_times']),tuple(a['order']))
        else:
            fact=('transition',a['object'],a.get('time',a.get('return_time')))
        groups[('fact_'+fact[0],eid,fact)].append(i)
        groups[('episode_time',eid,round(q['query_time'],2))].append(i)
    rr=[];cc=[];vv=[];lower=[];upper=[]
    def add(indices,lo,hi):
        j=len(lower);rr.extend([j]*len(indices));cc.extend(indices);vv.extend([1.]*len(indices));lower.append(lo);upper.append(hi)
    for key,ids in groups.items():
        field=key[0]
        if field=='episode':lo=hi=10
        elif field=='type_status':
            lo=hi=quotas['uncertain'][key[1]] if key[2]=='uncertain' else quotas['types'][key[1]]-quotas['uncertain'][key[1]]
        elif field=='episode_type':lo,hi=0,3
        elif field=='episode_status':lo,hi=(1,2) if key[2]=='uncertain' else (8,9)
        elif field=='physical':lo,hi=24,40
        elif field=='physical_kind':
            lo,hi=(4,10) if key[1]=='factual_retrieval' else (8,14) if key[1]=='current_state' else (5,10)
        elif field=='physical_object':lo,hi=0,2
        elif field=='room':lo,hi=0,8
        elif field=='pair':lo,hi=(8,16) if key[1]==('out of view','out of view') else (8,24)
        elif field=='episode_pair':lo,hi=0,3
        elif field=='count':lo,hi=(4,6) if key[1]==0 else (0,16)
        elif field=='count_kind':lo,hi=0,24
        elif field=='episode_count':lo,hi=0,3
        elif field=='count_events':lo,hi=0,2 if key[-1] else 1
        elif field=='episode_time':lo,hi=0,1
        else:lo,hi=0,1
        if field in relax:lo,hi=0,float('inf')
        add(ids,lo,hi)
    costs=[]
    for q in rows:
        visible=[o['pixels'] for o in q['provenance']['observations'] if o['visibility']=='visible']
        quality=1/math.sqrt(min(visible)) if visible else .1
        value=.05*random.Random(q['id']).random()+quality
        if q['provenance']['answer_basis']=='reviewed_rgb_physical_state':value-=1
        if q['question_subtype']=='visible_room_category':value+=1
        if q['question_subtype']=='observed_return_count' and q['semantic_anchor']['count']==0:value+=1
        costs.append(value)
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(lower),len(rows))).tocsc()
    fit=milp(costs,integrality=np.ones(len(rows)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,lower,upper),options=dict(time_limit=time_limit,mip_rel_gap=.05))
    if fit.x is None:raise RuntimeError(f'Cannot allocate evidence-grounded questions: {fit.message}')
    chosen=[q for i,q in enumerate(rows) if fit.x[i]>.5]
    assert len(chosen)==expected
    for kind, total in quotas['types'].items():
        assert sum(q['question_type']==kind for q in chosen)==total
        assert sum(q['question_type']==kind and q['diagnostics']['epistemic_status']=='uncertain' for q in chosen)==quotas['uncertain'][kind]
    return sorted(chosen,key=lambda q:(q['episode_id'],q['query_time'],q['id'])), {
        'solver_message':fit.message,'types':quotas['types'],'uncertain':quotas['uncertain'],
        'repeated_fact_groups': [dict(key=str(k), questions=[rows[i]['id'] for i in ids if fit.x[i]>.5])
            for k,ids in groups.items() if k[0].startswith('fact_') and sum(fit.x[i]>.5 for i in ids)>1]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('evidence','visibility','reference','review','quotas','output','work'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--time-limit', type=float, default=90)
    args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError(args.output)
    if args.work.exists():raise FileExistsError(args.work)
    bundle=json.loads(args.evidence.read_text());episodes={ep['metadata']['episode_id']:ep for ep in bundle['episodes']}
    visibility={v['episode_id']:v for v in json.loads(args.visibility.read_text())['episodes']}
    reference=[json.loads(line) for line in args.reference.read_text().splitlines() if line.strip()]
    review=json.loads(args.review.read_text());quotas=json.loads(args.quotas.read_text())
    pool=[]
    for eid,ep in episodes.items():
        for original in build_visual_pool(ep,visibility[eid],minimum_pixels=128,uncertainty_policy='all_questions'):
            if original['diagnostics']['epistemic_status']=='uncertain':
                q=rewrite_uncertain(original,ep,visibility[eid])
                if q is None:continue
            else:
                if original['question_subtype']=='last_two_observed_returns' and len(set(original['semantic_anchor']['order']))<2:
                    continue
                if original['question_subtype']=='observed_return_count':
                    # Use a positive duration and retain full sampled evidence.
                    if original['query_time']-original['semantic_anchor']['start']<8:continue
                q,_=language.revise(original,dict(semantic_anchor=original['semantic_anchor'],provenance=original['provenance']))
            q['id']=eid+'_rqa_'+digest([q['question_type'],q['question_subtype'],q['query_time'],q['semantic_anchor'],q['question']])
            q['qa_schema']=SCHEMA
            pool.append(q)
    pool.extend(physical_pool(episodes,visibility,review))
    pool.extend(exit_count_pool(episodes,visibility))
    pool=list({q['id']:q for q in pool}.values())
    args.work.mkdir(parents=True,exist_ok=True)
    write_json(args.work/'availability.json',dict(candidates=len(pool),
        type_status=dict(Counter(q['question_type']+'|'+q['diagnostics']['epistemic_status'] for q in pool))))
    write_jsonl(args.work/'candidates.jsonl',pool)
    selected,allocation=select(pool,quotas,time_limit=args.time_limit)
    write_jsonl(args.work/'selected.jsonl',selected)
    public,stats=export_dataset(selected,bundle['episodes'],reference,args.output,reference_sha256=sha256(args.reference),
        selection=allocation,uncertainty_policy='all_questions')
    manifest=json.loads((args.output/'qa_revision_manifest.json').read_text())
    manifest.update(schema=SCHEMA,reference_alignment_exception='All questions include uncertainty as an option; its presence no longer identifies the label.',
        known_answer_basis='Reviewed physical-state endpoints and sampled visibility facts; no unseen action count is treated as known.',
        minimum_visibility_pixels=128,physical_endpoint_review_sha256=sha256(args.review),model_predictions_used=False)
    write_json(args.output/'qa_revision_manifest.json',manifest)
    summary=json.loads((args.output/'summary.json').read_text());summary['version']=SCHEMA
    summary['question_subtype_counts']=dict(Counter(q['question_subtype'] for q in public))
    write_json(args.output/'summary.json',summary)
    for path in (args.output/'meta_data/qa_results').glob('*/*.json'):
        doc=json.loads(path.read_text());doc['qa_source']=SCHEMA;write_json(path,doc)
    ledger=[json.loads(line) for line in (args.output/'private/qa_provenance.jsonl').read_text().splitlines()]
    for p in ledger:p['qa_schema']=SCHEMA
    write_jsonl(args.output/'private/qa_provenance.jsonl',ledger)
    result=validate_annotation_contract(args.output)
    result.update(physical_state_questions=sum(q['provenance']['answer_basis']=='reviewed_rgb_physical_state' for q in selected),
        room_questions=sum(q['question_subtype']=='visible_room_category' for q in selected),
        return_count_answers=dict(Counter(q['semantic_anchor']['count'] for q in selected if q['question_subtype']=='observed_return_count')),
        visibility_pair_answers=dict(Counter('|'.join(q['semantic_anchor']['states']) for q in selected if 'states' in q['semantic_anchor'] and q['question_subtype'] in ('current_visibility_pair','earlier_visibility_pair'))),
        uniform_random_baseline=.25,always_uncertain_baseline=sum(q['diagnostics']['epistemic_status']=='uncertain' for q in selected)/len(selected),
        full_rgb_review='Physical endpoint candidates reviewed; sampled visibility candidates still require selected-frame inspection.')
    write_json(args.output/'validation.json',result)
    write_json(args.output/'private/physical_endpoint_review.json',review)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
