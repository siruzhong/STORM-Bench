"""Verify natural option domains, evidence bounds, and answer-blind selection."""
from collections import Counter
import pytest
from storm_virtualhome.natural_qa import UNKNOWN, build_natural_questions, select_natural_batch, _request_key


def events():
    result=[]
    for i, (obj, before, after, hidden) in enumerate([
        (1, 'closed', 'open', False), (2, 'closed', 'open', False),
        (3, 'closed', 'open', False), (1, 'open', 'closed', True),
        (2, 'open', 'closed', True), (3, 'open', 'closed', True)]):
        result.append(dict(object_id=obj, object='cabinet', object_description=f'cabinet {obj}',
                           event_index=i, verb='Open' if after=='open' else 'Close',
                           before_state=before, after_state=after, before_time=i*6+1,
                           end=i*6+3, hidden_time=i*6+3, discovery_time=i*6+5,
                           visibility='offscreen' if hidden else 'fully_observed'))
    return result


def test_domains_and_hidden_answers_share_identical_options():
    pool=build_natural_questions('episode',events(),42)
    current=[q for q in pool if q['question_type']=='current_state']
    assert all(set(q['options'])=={'open','closed',UNKNOWN} for q in current)
    assert {q['options'][q['answer_index']] for q in current}=={'open','closed',UNKNOWN}
    counts=[q for q in pool if q['question_type']=='history_aggregation']
    assert all('Both have the same number of confirmed changes.' in q['options'] for q in counts)
    for q in pool:
        assert q['chance_accuracy']==1/len(q['options'])
        assert all(0<=a<b<=q['query_time'] for a,b in q['evidence_spans'])
        assert not {'broken','missing','Its color changed.'}&set(q['options'])


def test_count_uses_discovery_order_not_program_order():
    source=events()
    source[0]['discovery_time']=30
    pool=build_natural_questions('episode',source)
    q=next(q for q in pool if q['event_index']==3 and q['question_subtype']=='comparative_count_1_2')
    assert q['options'][q['answer_index']]=='Both have the same number of confirmed changes.'


def test_ambiguous_names_fail_and_selection_is_balanced():
    source=events()
    source[1]['object_description']='cabinet 1'
    source[4]['object_description']='cabinet 1'
    with pytest.raises(ValueError,match='unique'):
        build_natural_questions('episode',source)
    pools=[]
    for i in range(3):
        source=events()
        if i % 2:
            for event in source:
                event['object_id']=4-event['object_id']
                event['object_description']=f"cabinet {event['object_id']}"
        pools.append(build_natural_questions(f'episode_{i}',source,i))
    rows=select_natural_batch(pools,7)
    assert rows==select_natural_batch(pools,7)
    assert len(rows)==30
    assert set(Counter(q['question_type'] for q in rows).values())=={5}
    assert len({q['id'] for q in rows})==30
    assert len({(q['episode_id'], _request_key(q)) for q in rows})==30


def test_tracking_rejects_overlapping_confirmations_with_older_before_frame():
    source=events()
    source[4]['before_time']=7
    source[4]['discovery_time']=22
    pool=build_natural_questions('episode',source)
    assert not any(q['event_index']==3 and q['question_type']=='object_tracking' for q in pool)


def test_temporal_ties_are_not_forced_into_an_order():
    source=events()
    source[0]['discovery_time']=source[1]['discovery_time']
    pool=build_natural_questions('episode',source)
    assert not any(q['question_subtype']=='first_confirmation_1_2' for q in pool)
    assert all('seconds' not in q['question'] for q in pool)
    assert all(q['question_type'] not in q['id'] for q in pool)
