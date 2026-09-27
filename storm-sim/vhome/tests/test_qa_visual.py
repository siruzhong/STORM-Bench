"""Check sampled visibility boundaries and the exported evaluator contract."""
from copy import deepcopy

from storm_virtualhome_qa.visual_qa import build_visual_pool, UNDETERMINED
from storm_virtualhome_qa.qa_export import balanced_positions, compare_distributions


def fixture():
    events=[dict(object_id=i, object_description=n, event_index=i, visibility='offscreen',
                 start=4., end=7., verb='Grab')
            for i,n in enumerate(('mug','cabinet','switch'))]
    episode=dict(metadata=dict(episode_id='fixture',duration_seconds=16.,room='kitchen'),
                 events=events,events_sha256='fixture')
    counts=[]
    for frame in range(320):
        t=frame/20
        counts.append([200 if t<3 or t>=8 else 0,
                       200 if 2<=t<5 or t>=11 else 0,
                       200 if t<6 or t>=13 else 0])
    visibility=dict(fps=20,frames=320,objects=[0,1,2],counts=counts,manifest_sha256='fixture')
    return episode,visibility


def test_known_answers_use_only_sampled_prefix_evidence():
    episode,visibility=fixture()
    before=build_visual_pool(episode,visibility)
    changed=deepcopy(visibility)
    for row in changed['counts'][221:]:
        row[:]=[0,0,0]
    after=build_visual_pool(episode,changed)
    select=lambda rows:{q['id']:q for q in rows if q['query_time']<=11.05 and q['diagnostics']['epistemic_status']=='known'}
    assert select(before)==select(after)
    for q in before:
        assert all(a<=b<=q['query_time'] for a,b in q['evidence_spans'])
        assert all(o['time']==int(o['time']) and o['frame']==20*o['time']
                   and o['time']<=q['query_time'] for o in q['provenance']['observations'])


def test_hidden_event_requires_zero_pixels_at_native_frame_rate():
    episode,visibility=fixture()
    original=build_visual_pool(episode,visibility)
    hidden=[q for q in original if q['diagnostics']['epistemic_status']=='uncertain']
    assert hidden and {q['semantic_anchor']['object'] for q in hidden}=={0}
    assert all(q['options'][q['answer_index']]==UNDETERMINED for q in hidden)
    # One visible native frame must reject a claimed fully hidden interval,
    # even when that frame is not on the 1 FPS evaluator clock.
    visibility['counts'][101][0]=1
    changed=build_visual_pool(episode,visibility)
    assert not [q for q in changed if q['diagnostics']['epistemic_status']=='uncertain']


def test_balancing_preserves_semantic_answers_and_joint_counts():
    episode,visibility=fixture()
    rows=build_visual_pool(episode,visibility)[:16]
    revised=balanced_positions(rows)
    assert [q['options'][q['answer_index']] for q in rows]==[q['options'][q['answer_index']] for q in revised]
    assert sorted([q['answer_index'] for q in revised])==[0]*4+[1]*4+[2]*4+[3]*4
    stats=compare_distributions(rows,revised,durations={'fixture':60.05})
    assert stats['comparison']['question_type']['total_variation']==0
    assert stats['comparison']['epistemic_status']['total_variation']==0


def test_reference_uncertainty_options_preserve_grounded_answers():
    episode,visibility=fixture()
    old={q['id']:q for q in build_visual_pool(episode,visibility,uncertainty_policy='all_questions')}
    new={q['id']:q for q in build_visual_pool(episode,visibility,uncertainty_policy='reference')}
    assert old.keys()==new.keys()
    for key,q in new.items():
        source=old[key]
        assert len(q['options'])==len(set(q['options']))==4
        assert (UNDETERMINED in q['options'])==(q['diagnostics']['epistemic_status']=='uncertain')
        assert q['options'][q['answer_index']]==source['options'][source['answer_index']]
        for field in ['question','query_time','diagnostics','evidence_spans','semantic_anchor','change_intensity']:
            assert q[field]==source[field]
        if q['question_subtype']=='object_return_after_absence':
            assert 'None of these objects returned to view.' in q['options']
