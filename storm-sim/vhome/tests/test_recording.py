"""Check instance-color matching and snapshot ordering."""
import numpy as np
from storm_virtualhome.recording import (Recorder, recorded_target_visible,
                                         target_visibility_threshold, visible_pixels)


def test_normalized_colors_and_tolerance():
    image = np.array([[[0, 128, 255], [0, 129, 255], [255, 128, 0]]], dtype=np.uint8)
    assert visible_pixels(image, [0, 0.5, 1]) == 2
    assert visible_pixels(image, [255, 128, 0]) == 1


def test_recorded_visibility_uses_the_encoded_action_frame():
    observation={'last_visible_pixels':{'41':39,'42':40}}
    assert recorded_target_visible(observation,41,40) is False
    assert recorded_target_visible(observation,42,40) is True
    assert recorded_target_visible(observation,43,40) is None


def test_small_target_visibility_threshold_is_class_specific():
    config={'evidence':{'min_visible_pixels':40,
                        'min_visible_pixels_by_class':{'lightswitch':12}}}
    assert target_visibility_threshold(config,{'class_name':'lightswitch'})==12
    assert target_visibility_threshold(config,{'class_name':'cabinet'})==40


def test_refresh_palette_before_segmentation_and_report_collisions(tmp_path):
    calls = []

    class Client:
        def graph(self):
            calls.append('graph')
            return {'nodes': [], 'edges': []}

        def colors(self):
            calls.append('colors')
            return {'1': [1, 0, 0], '2': [1, 0, 0]}

        def image(self, index, mode='normal', **kwargs):
            calls.append(mode)
            return np.full((2, 2, 3), [255, 0, 0], dtype=np.uint8)

    recorder = Recorder.__new__(Recorder)
    recorder.client = Client()
    recorder.output = tmp_path
    recorder.camera_index = 0
    recorder.config = {'render': {'width': 2, 'height': 2}, 'evidence': {'hold_seconds': 0.5}}
    recorder.fps, recorder.frames = 20, 0
    recorder.append = lambda frame, count: None
    snapshot = recorder.snapshot('test', 1)
    assert calls == ['graph', 'colors', 'normal', 'seg_inst']
    assert snapshot['visible_pixels'] == 4
    assert snapshot['color_collision_ids'] == [2]


def test_black_rgb_frame_is_rejected_before_encoding():
    import pytest

    recorder = Recorder.__new__(Recorder)
    with pytest.raises(RuntimeError, match='entirely black'):
        recorder.append(np.zeros((8, 8, 3), dtype=np.uint8))


def test_native_operation_timing_excludes_implicit_approach_and_other_actor_wait(tmp_path):
    from storm_virtualhome.recording import native_action_spans
    folder=tmp_path/'action_0003'/'1';folder.mkdir(parents=True)
    (folder/'ftaa_action_0003.txt').write_text('0 OPEN 20 85\n0 WALK 100 149\n0 CLOSE 150 215\n')
    spans=native_action_spans(tmp_path,100,239,10.0,20)['1']
    assert [span['verb'] for span in spans]==['WALK','CLOSE']
    assert spans[-1]['start']==12.5
    assert spans[-1]['end']==15.8


def test_late_actor_uses_shared_unity_clock_instead_of_observer_frame_numbers(tmp_path):
    import json
    from storm_virtualhome.recording import native_action_spans, native_frame_mapping
    clock=tmp_path/'clock.jsonl'
    rows=[]
    for i in range(10):
        rows.extend([dict(unity_frame=1000+i,actor=0,source_frame=400+i),
                     dict(unity_frame=1000+i,actor=2,source_frame=10+i)])
    clock.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    folder=tmp_path/'action_0003'/'2';folder.mkdir(parents=True)
    (folder/'ftaa_action_0003.txt').write_text('0 WALK 10 11\n0 SWITCHON 12 19\n')
    spans=native_action_spans(tmp_path,400,409,20.0,20,native_frame_mapping(clock))['2']
    assert spans[-1]['start']==20.1
    assert spans[-1]['end']==20.5


def test_native_frame_mapping_discards_previous_counter_epoch(tmp_path):
    from storm_virtualhome.recording import native_frame_mapping
    import json
    rows=[dict(unity_frame=1,actor=0,source_frame=260),
          dict(unity_frame=1,actor=1,source_frame=260),
          dict(unity_frame=2,actor=0,source_frame=715),
          dict(unity_frame=2,actor=1,source_frame=0),
          dict(unity_frame=3,actor=0,source_frame=974),
          dict(unity_frame=3,actor=1,source_frame=259)]
    path=tmp_path/'clock.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    mapping=native_frame_mapping(path)['1']
    assert mapping[259]==974
    assert 260 not in mapping
def test_half_open_native_interval_does_not_include_next_action_frame():
    from storm_virtualhome.recording import frame_interval
    assert frame_interval(46.2, 47.800000000000004, 20) == (924, 956)
    assert frame_interval(46.199999999999996, 47.8, 20) == (924, 956)
    assert frame_interval(1.001, 1.101, 20) == (21, 23)


def test_observer_frame_check_matches_final_visibility_gate():
    from storm_virtualhome.recording import observer_frame_issue
    color=[1,0,0]
    mask=np.zeros((4,4,3),dtype=np.uint8)
    assert observer_frame_issue(mask,color)=='observer_missing'
    mask[1,1]=[255,0,0]
    assert observer_frame_issue(mask,color) is None
    mask[0,1]=[255,0,0]
    assert observer_frame_issue(mask,color)=='observer_touches_upper_border'
