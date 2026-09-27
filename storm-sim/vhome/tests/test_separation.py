"""Keep separate native recorder epochs and reject invalid pose evidence."""
import pytest
import json
from storm_virtualhome.separation import align_clock, read_hips, audit_separation


def test_actor_reset_does_not_overwrite_earlier_epoch():
    rows = []
    for unity, source, npc in [(100, 10, 9), (101, 11, 10), (200, 12, 0), (201, 13, 1)]:
        rows.extend([dict(unity_frame=unity, actor=0, source_frame=source),
                     dict(unity_frame=unity, actor=1, source_frame=npc)])
    result = align_clock(rows)
    assert result[10][1] == (0, 9)
    assert result[12][1] == (1, 0)
    assert result[13][0] == (0, 13)


def test_hips_are_read_by_name_and_nonfinite_poses_are_rejected(tmp_path):
    path = tmp_path / 'poses.txt'
    path.write_text('Head Hips\n7 4 5 6 1 2 3\n')
    assert read_hips(path) == {7: (1, 2, 3)}
    path.write_text('Hips\n7 nan 2 3\n')
    with pytest.raises(ValueError, match='Invalid skeleton'):
        read_hips(path)


def test_synchronized_roots_detect_a_mid_action_near_pass(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(motion_contract=dict(rules=dict(agent_clearance_m=.85)))))
    (tmp_path/'logs').mkdir()
    (tmp_path/'logs/native_frame_clock.jsonl').write_text('')
    frames=[dict(frame=i,rgb=f'actions/0000/action_0000/0/Action_{i:04d}_0_normal.png',
                 monitored_colors={'1':[],'2':[]},
                 camera_pose=dict(actor_positions={'0':[0,0,0],'1':[distance,0,0]}))
            for i,distance in enumerate([1.2,.7,1.2])]
    path=tmp_path/'frame_manifest.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in frames))
    report=audit_separation(tmp_path)
    assert report['exact_root_evidence'] and report['covered_frames']==3
    assert report['status']=='clearance_violation'
    assert report['violations'][0]['frame']==1
    frames[1]['camera_pose']['actor_positions']['1']=[1.1,0,0]
    path.write_text(''.join(json.dumps(row)+'\n' for row in frames))
    assert audit_separation(tmp_path)['status']=='separation_verified'
    del frames[1]['camera_pose']['actor_positions']['1']
    path.write_text(''.join(json.dumps(row)+'\n' for row in frames))
    assert audit_separation(tmp_path)['status']=='incomplete_evidence'
