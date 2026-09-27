"""Check draft coverage, recorded answer provenance, and incomplete captures."""
import json
import pytest
from storm_virtualhome.raw_qa import write_raw_questions


def capture(root):
    events = []
    for index in range(10):
        before, after = ('closed', 'open') if index % 2 == 0 else ('open', 'closed')
        events.append(dict(event_index=index, object_id=10, object='cabinet',
                           before_state=before, after_state=after,
                           verb='Open' if after == 'open' else 'Close',
                           before_time=index * 6 + 1, end=index * 6 + 3,
                           hidden_time=index * 6 + 3, discovery_time=index * 6 + 5,
                           visibility='offscreen' if index % 2 else 'fully_observed'))
    for name, value in [('events.json', events),
                        ('config.json', {'episode_id': 'room_1', 'seed': 42}),
                        ('observer_validation.json', {'duration_sec': 65, 'fps': 20})]:
        (root / name).write_text(json.dumps(value))
    return events


def test_raw_qa_covers_every_event_with_natural_choices_and_no_polishing(tmp_path):
    events = capture(tmp_path)
    status = write_raw_questions(tmp_path)
    saved = (tmp_path / 'qa.json').read_text()
    rows = json.loads(saved)['questions']
    assert status['status'] == 'raw_ready' and status['polishing'] == 'pending'
    assert len(rows) == 10 and {q['event_index'] for q in rows} == set(range(10))
    assert all(q['polish_status'] == 'pending' for q in rows)
    for q in rows:
        event = events[q['event_index']]
        answer = q['options'][q['answer_index']]
        assert 2 <= len(q['options']) <= 4
        assert all(0 <= start < end <= q['query_time'] <= 65 for start, end in q['evidence_spans'])
        if q['question_type'] == 'factual_retrieval':
            assert answer == event['before_state']
        elif q['question_type'] == 'state_change':
            assert answer == f"From {event['before_state']} to {event['after_state']}."
        elif q['question_subtype'] == 'observed':
            assert answer == event['after_state']
        else:
            assert q['diagnostics']['epistemic_status'] == 'uncertain'
    assert 'object_tracking' in status['unavailable_candidate_types']
    assert status['candidate_questions'] > 10
    write_raw_questions(tmp_path)
    assert (tmp_path / 'qa.json').read_text() == saved


def test_partial_capture_cannot_be_exported_as_complete_draft(tmp_path):
    events = capture(tmp_path)
    (tmp_path / 'events.json').write_text(json.dumps(events[:9]))
    with pytest.raises(ValueError, match='ten distinct'):
        write_raw_questions(tmp_path)
    assert not (tmp_path / 'qa.json').exists()
