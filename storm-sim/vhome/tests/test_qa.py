"""Check prefix boundaries and coverage without starting Unity."""
import pytest

from storm_virtualhome.qa import TYPES, build_questions
from storm_virtualhome.scene import inside_closed, on_surface, room_of


def pairs():
    return [{'object': name, 'container': 'fridge', 'surface': 'kitchencounter',
             'disappear': {'start': i * 20 + 1, 'end': i * 20 + 9},
             'appear': {'start': i * 20 + 10, 'end': i * 20 + 19}}
            for i, name in enumerate(['apple', 'plate', 'mug'])]


def test_six_types_and_no_future_evidence():
    questions = build_questions('episode', pairs(), 60)
    assert {q['question_type'] for q in questions} == set(TYPES)
    for q in questions:
        assert len(set(q['options'])) == 4
        assert 0 <= q['answer_index'] < 4
        assert all(0 <= start <= end <= q['query_time'] <= 60 for start, end in q['evidence_spans'])
    counts = [q for q in questions if q['question_type'] == 'history_aggregation']
    assert [q['options'][q['answer_index']] for q in counts] == ['1', '2', '3']


def test_reject_short_episode_and_insufficient_events():
    with pytest.raises(ValueError):
        build_questions('episode', pairs(), 1)
    with pytest.raises(ValueError):
        build_questions('episode', [], 20)


def test_seed_is_reproducible():
    assert build_questions('episode', pairs(), 60, 11) == build_questions('episode', pairs(), 60, 11)


def test_containment_requires_closed_container():
    graph = {'nodes': [{'id': 1, 'class_name': 'kitchen', 'category': 'Rooms'},
                       {'id': 2, 'class_name': 'fridge', 'states': ['CLOSED']}, {'id': 3}],
             'edges': [{'from_id': 3, 'to_id': 2, 'relation_type': 'INSIDE'},
                       {'from_id': 2, 'to_id': 1, 'relation_type': 'INSIDE'}]}
    assert inside_closed(graph, 3, 2)
    assert room_of(graph, 3) == 'kitchen'
    assert not on_surface(graph, 3, 2)
    graph['nodes'][1]['states'] = ['OPEN']
    assert not inside_closed(graph, 3, 2)


def test_single_pair_still_covers_all_six_types():
    questions = build_questions('episode', pairs()[:1], 20)
    assert len(questions) == 6
    assert {q['question_type'] for q in questions} == set(TYPES)
