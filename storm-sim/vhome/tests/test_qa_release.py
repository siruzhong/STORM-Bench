"""Exercise generation, wording revision, and accidental label disclosure."""

import pytest

from storm_virtualhome_qa import language, revision
from storm_virtualhome_qa.generation import select
from storm_virtualhome_qa.qa_export import export_dataset
from storm_virtualhome_qa.validation import validate
from storm_virtualhome_qa.visual_qa import build_visual_pool
from test_qa_visual import fixture


def make_release(tmp_path):
    episode, visibility = fixture()
    episode['metadata'].update(scene=0, room_id=1, hashes={'video.mp4': '0' * 64})
    pool = build_visual_pool(episode, visibility, uncertainty_policy='all_questions')
    selected = []
    for row in pool:
        if row['question_subtype'] != 'current_visibility_pair':
            continue
        q, _ = language.revise(row, row)
        selected.append(q)
        if len(selected) == 10:
            break
    assert len(selected) == 10
    raw, final = tmp_path / 'raw', tmp_path / 'final'
    export_dataset(selected, [episode], selected, raw, reference_sha256='fixture',
                   selection={}, uncertainty_policy='all_questions')
    revision.main(['--source', str(raw), '--output', str(final)])
    return raw, final


def test_revision_preserves_labels_evidence_and_positions(tmp_path):
    raw, final = make_release(tmp_path)
    before = revision.read_rows(raw / 'questions.jsonl')
    after = revision.read_rows(final / 'questions.jsonl')
    for a, b in zip(before, after):
        for key in ('id', 'query_time', 'answer_index', 'evidence_spans', 'diagnostics'):
            assert a[key] == b[key]
        assert a['options'][a['answer_index']] == b['options'][b['answer_index']]
    assert validate(final)['questions'] == 10
    with pytest.raises(ValueError, match='already exists'):
        revision.main(['--source', str(raw), '--output', str(final)])


def test_validator_rejects_answer_fields_in_model_inputs(tmp_path):
    _, final = make_release(tmp_path)
    path = final / 'model_inputs.jsonl'
    rows = revision.read_rows(path)
    rows[0]['answer'] = 0
    revision.write_rows(path, rows)
    with pytest.raises(ValueError, match='private fields'):
        validate(final)


def test_selector_rejects_missing_positive_quota():
    episode, visibility = fixture()
    pool = build_visual_pool(episode, visibility)
    rows = [q for q in pool if q['question_type'] == 'current_state']
    with pytest.raises(ValueError, match='Insufficient candidates'):
        select(rows, {'types': {'factual_retrieval': 10}, 'uncertain': {'factual_retrieval': 0}})
