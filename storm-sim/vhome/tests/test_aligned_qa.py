"""Check causal evidence, missing annotations, and distribution allocation."""
from copy import deepcopy

from storm_virtualhome.aligned_qa import apportion, build_pool


def episode():
    events, observations = [], []
    for i, (obj, before, after, start, end, seen) in enumerate([
        (1, 'closed', 'open', 2, 4, 7),
        (2, 'closed', 'open', 10, 12, 17),
        (1, 'open', 'closed', 20, 22, 27),
        (2, 'open', 'closed', 30, 32, 37),
    ]):
        events.append(dict(event_index=i, object_id=obj, object='cabinet',
            object_description=f'cabinet {obj}', verb='Open' if after=='open' else 'Close',
            before_state=before, after_state=after, start=start, end=end,
            before_time=start-1, discovery_time=seen, visibility='offscreen'))
        for side, state, time in [('before', before, start-1), ('after', after, seen)]:
            frame=int(time*20)-1
            observations.append(dict(event_index=i, object_id=obj, side=side, state=state,
                key=f'{i}_{side}', time=time, frame=frame, pixels=100, collisions=[],
                source_json='graph.json', source_sha256='fixture', visibility_threshold=40,
                original_stage_time=time, original_stage_frame=frame,
                state_interval=[0, 40], frame_manifest_sha256='fixture'))
    return dict(metadata=dict(episode_id='fixture', episode_path='episodes/fixture'),
                events=events, observations=observations, qa=dict(sample_fps=20), events_sha256='fixture')


def test_reference_allocation_preserves_total_and_rounding():
    counts={'current_state':54, 'factual_retrieval':54, 'state_change':51,
            'history_aggregation':51, 'object_tracking':50, 'temporal_reasoning':40}
    result=apportion([dict(question_type=k) for k,n in counts.items() for _ in range(n)], 290)
    assert result==dict(current_state=52, factual_retrieval=52, state_change=49,
                        history_aggregation=49, object_tracking=49, temporal_reasoning=39)


def test_future_changes_cannot_rewrite_earlier_prefix_answers():
    source=episode()
    first={q['id']:q for q in build_pool(source) if q['query_time']<=17}
    changed=deepcopy(source)
    for o in changed['observations']:
        if o['time']>17:
            o['state']='open' if o['state']=='closed' else 'closed'
    second={q['id']:q for q in build_pool(changed) if q['query_time']<=17}
    assert first==second
    for q in build_pool(source):
        assert all(0<=a<b<=q['query_time'] for a,b in q['evidence_spans'])


def test_missing_state_is_not_counted_as_no_change():
    source=episode()
    source['observations'][0]['state']=None
    pool=build_pool(source)
    counts=[q for q in pool if q['question_type']=='history_aggregation'
            and q['semantic_anchor']['object']==1 and q['semantic_anchor']['start']==0]
    assert not counts
    assert all('at an unverified location' not in option for q in pool for option in q['options'])


def test_delayed_reveal_changes_temporal_order():
    source=episode()
    source['events'][0]['discovery_time']=18
    source['observations'][1].update(time=18, frame=359)
    pool=build_pool(source)
    question=next(q for q in pool if q['question_type']=='temporal_reasoning' and q['query_time']==18)
    assert question['semantic_anchor']['ordered_events']==[1,0]
