"""Check the room capture gate and atomic publication helper."""
import importlib.util
import json
from pathlib import Path
import sys


SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'rollout_rooms_native.py'
sys.path.insert(0,str(SCRIPT.parent))
SPEC=importlib.util.spec_from_file_location('rollout_rooms_native_contract',SCRIPT)
MODULE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path,value):
    path.write_text(json.dumps(value))


def accepted_capture(root):
    program={'plan_id':'fixed'}
    write(root/'event_program.json',program)
    write(root/'plan_execution.json',{'accepted':True})
    write(root/'camera_continuity.json',{'accepted':True})
    write(root/'observer_validation.json',{'event_count':10,'offscreen_event_count':5,
        'duration_sec':65,'camera_overlap_samples':0,'frames':1300})
    write(root/'debug_validation.json',{'frames':1300})
    write(root/'motion_validation.json',{'acceptance_mode':'locomotion_and_turning',
        'active_frame_fraction':.91,'longest_inactive_seconds':1.2})
    write(root/'qa.json',{'questions':[{} for _ in range(60)]})
    (root/'video.mp4').write_bytes(b'video')
    (root/'debug.mp4').write_bytes(b'debug')
    return program


def test_capture_gate_rechecks_all_release_artifacts(tmp_path):
    program=accepted_capture(tmp_path)
    assert MODULE.capture_accepted(tmp_path,program)
    write(tmp_path/'observer_validation.json',{'event_count':10,'offscreen_event_count':4,
        'duration_sec':65,'camera_overlap_samples':0,'frames':1300})
    assert not MODULE.capture_accepted(tmp_path,program)


def test_video_acceptance_can_defer_qa_without_skipping_camera_checks(tmp_path):
    program=accepted_capture(tmp_path)
    (tmp_path/'qa.json').unlink()
    assert not MODULE.capture_accepted(tmp_path,program)
    assert MODULE.capture_accepted(tmp_path,program,require_qa=False)
    write(tmp_path/'camera_continuity.json',{'accepted':False})
    assert not MODULE.capture_accepted(tmp_path,program,require_qa=False)


def test_publication_link_is_idempotent_and_points_to_capture(tmp_path):
    capture=tmp_path/'capture';capture.mkdir()
    destination=tmp_path/'dataset'/'episode_0000'
    MODULE.publish_link(destination,capture)
    MODULE.publish_link(destination,capture)
    assert destination.is_symlink()
    assert destination.resolve()==capture.resolve()


def test_probe_merge_preserves_successes_from_separate_target_runs():
    first={'scene':1,'room_id':2,'status':'blocked','targets':[
        {'id':10,'name':'book','operations':[{'first':'Grab','second':'PutBack','accepted':True}]},
        {'id':11,'name':'cabinet','operations':[{'first':'Open','second':'Close','accepted':False}]},
    ]}
    second={'scene':1,'room_id':2,'status':'blocked','targets':[
        {'id':11,'name':'cabinet','operations':[{'first':'Open','second':'Close','accepted':True}]},
        {'id':12,'name':'pillow','operations':[{'first':'Grab','second':'PutBack','accepted':True}]},
    ]}
    merged=MODULE.merge_probe_entry(first,second)
    assert merged['status']=='complete'
    assert {target['id'] for target in merged['targets']}=={10,11,12}
    assert all(any(operation['accepted'] for operation in target['operations'])
               for target in merged['targets'])


def test_capture_accepts_raw_qa_only_with_complete_event_coverage(tmp_path):
    program = accepted_capture(tmp_path)
    write(tmp_path / 'qa.json', {'qa_stage': 'raw',
          'questions': [{'event_index': i} for i in range(10)]})
    assert MODULE.capture_accepted(tmp_path, program)
    write(tmp_path / 'qa.json', {'qa_stage': 'raw',
          'questions': [{'event_index': 0} for _ in range(10)]})
    assert not MODULE.capture_accepted(tmp_path, program)
