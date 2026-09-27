"""Check clear native paths and useful concurrent observer travel."""
from storm_virtualhome.native_navigation import NativePatrol, path_clearance, hidden_path


def test_path_clearance_checks_every_corner_segment():
    assert path_clearance([[0,0,0],[2,0,0],[2,0,2]],[[2,0,1]])==0
    assert path_clearance([[0,0,0],[2,0,0]],[[1,0,2]])==2


def test_short_joint_walk_slows_animation_and_next_walk_resets_request(tmp_path):
    import json
    import pytest
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.log_dir=tmp_path
    waypoint=dict(navigation_position=[2,0,0],navigation_path=[[0,0,0],[2,0,0]],
                  concurrent_seconds=3.4)
    patrol.set_destination([0,0,0],waypoint)
    request=json.loads((tmp_path/'observer_destination.json').read_text())
    assert request['animation_speed_scale']==.45
    waypoint['concurrent_seconds']=2.0
    patrol.set_destination([0,0,0],waypoint)
    request=json.loads((tmp_path/'observer_destination.json').read_text())
    assert request['animation_speed_scale']==pytest.approx(2/(1.7*1.7))
    del waypoint['concurrent_seconds']
    patrol.set_destination([0,0,0],waypoint)
    assert 'animation_speed_scale' not in json.loads((tmp_path/'observer_destination.json').read_text())


def test_operation_patrol_prefers_enough_travel_and_rejects_crossing_people():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[1.2,0,0],2:[3.5,0,0],3:[0,0,3.5]}
    patrol.paths={0:[dict(to=i,length=length,corners=[[0,0,0],patrol.points[i]])
                    for i,length in [(1,1.2),(2,3.5),(3,3.5)]]}
    selected=patrol.choose([0,0,0],[],[3.5,0,2],99,operation_seconds=3.3)
    assert selected['navigation_position']==[3.5,0,0]
    selected=patrol.choose([0,0,0],[[2,0,0]],[3.5,0,2],99,operation_seconds=3.3)
    assert selected['navigation_position']==[0,0,3.5]


def test_acquisition_routes_around_an_occluded_station():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[0,0,1],2:[1,0,1],3:[1,0,0]}
    patrol.paths={start:[dict(to=end,length=1.0,corners=[patrol.points[start],patrol.points[end]])]
                  for start,end in [(0,1),(1,2),(2,3)]}
    patrol.sight={99:{3}}
    waypoint=patrol.choose([0,0,0],[],[2,0,0],8,acquire=True,target_id=99)
    assert waypoint['navigation_position']==[0,0,1]
    assert waypoint['observation_station']==3


def test_patrol_rejects_a_crossing_operator_route():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[1.5,0,0],2:[0,0,1.5]}
    patrol.paths={0:[dict(to=end,length=1.5,corners=[patrol.points[0],patrol.points[end]])
                    for end in [1,2]]}
    waypoint=patrol.choose([0,0,0],[],[3,0,0],8,clearance=.5,
        reserved_segments=[([1,0,-1],[1,0,.25])])
    assert waypoint['navigation_position']==[0,0,1.5]


def test_operator_reservation_keeps_native_bends(tmp_path,monkeypatch):
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.log_dir=tmp_path
    corners=[[0,0,0],[0,0,2],[2,0,2]]
    calls=[]
    def request(*args,**kwargs):
        calls.append(args)
        return dict(paths=[dict(complete=True,corners=corners)])
    monkeypatch.setattr('storm_virtualhome.native_navigation.request_paths',request)
    segments=patrol.operator_segments(corners[0],corners[-1])
    assert segments==[(corners[0],corners[1]),(corners[1],corners[2])]
    assert patrol.operator_segments(corners[0],corners[-1])==segments
    assert len(calls)==1


def test_visibility_refresh_keeps_frozen_point_ids_and_drops_a_blocked_station(tmp_path,monkeypatch):
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.log_dir=tmp_path;patrol.camera={'position':[0,1.6,0]}
    patrol.points={2:[0,0,0],9:[1,0,0]};patrol.sight={17:{2}}
    patrol.observation_distances={17:1.0}
    patrol.camera_stations={};patrol.observed_stations={17:2}
    paths={2:[dict(to=9,length=1,corners=[[0,0,0],[1,0,0]])]}
    patrol.paths=paths
    def request(directory,points,edges,**kwargs):
        assert edges==[] and points==[[0,0,0],[1,0,0]]
        assert kwargs['sight_targets'][0]['center']==[2,1,0]
        assert kwargs['fresh'] is True
        return dict(id='fresh',sight=[
            dict(point=0,target_id=17,clear_rays=0,observer_camera_clear=True,hidden_camera_clear=True),
            dict(point=1,target_id=17,clear_rays=2,observer_camera_clear=True,hidden_camera_clear=True)])
    monkeypatch.setattr('storm_virtualhome.native_navigation.request_paths',request)
    patrol.refresh_sight(dict(id=17,class_name='mug',bounding_box=dict(center=[2,1,0],size=[.2,.2,.2])))
    assert patrol.sight[17]=={9} and 17 not in patrol.observed_stations
    assert patrol.observation_distances[17]==1.0
    assert patrol.paths is paths


def test_small_target_acquisition_prefers_a_readable_station():
    from storm_virtualhome.native_navigation import observation_distance
    assert observation_distance(dict(class_name='toothbrush',bounding_box=dict(size=[.02,.02,.16])))==1.0
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None;patrol.sight={17:{1,2}}
    patrol.observation_distances={17:1.0}
    patrol.points={0:[0,0,0],1:[.6,0,0],2:[1.6,0,0]}
    patrol.paths={0:[dict(to=i,length=length,corners=[[0,0,0],patrol.points[i]])
                    for i,length in [(1,.6),(2,1.6)]]}
    result=patrol.choose([0,0,0],[],[2.6,0,0],8,acquire=True,target_id=17)
    assert result['navigation_position']==[1.6,0,0]


def test_hidden_route_checks_intermediate_positions():
    camera={'position':[0,2.2,-1.7]}
    assert hidden_path([[0,0,0],[1,0,0]],[0,0,1],[[0,0,-5]],camera)
    assert not hidden_path([[0,0,0],[1,0,0]],[0,0,1],[[0,0,5]],camera)
    assert not hidden_path([[-4,0,0],[4,0,0]],[0,0,1],[[0,0,1]],camera)


def test_concurrent_approach_does_not_stop_at_the_first_short_edge():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[1,0,0],2:[3.5,0,0]}
    patrol.paths={0:[dict(to=i,length=length,corners=[[0,0,0],patrol.points[i]])
                    for i,length in [(1,1),(2,3.5)]]}
    patrol.sight={9:{1,2}}
    selected=patrol.choose([0,0,0],[],[2.25,0,1],8,operation_seconds=3.2,acquire=True,target_id=9)
    assert selected['navigation_position']==[3.5,0,0]


def test_joint_hidden_action_prefers_continuous_walking_over_extra_target_distance():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[-1,0,0],2:[2.7,0,0]}
    patrol.paths={0:[dict(to=i,length=length,corners=[[0,0,0],patrol.points[i]])
                    for i,length in [(1,1),(2,2.7)]]}
    selected=patrol.choose([0,0,0],[],[3,0,1.5],8,
                           operation_seconds=2.5,hidden=True)
    assert selected['navigation_position']==[2.7,0,0]


def test_pixel_verified_station_is_preferred_over_ray_only_station():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[1,0,0],2:[0,0,1.5]}
    patrol.paths={0:[dict(to=i,length=length,corners=[[0,0,0],patrol.points[i]])
                    for i,length in [(1,1),(2,1.5)]]}
    patrol.sight={9:{1}}
    patrol.observed_stations={}
    patrol.note_visible(9,[0,0,1.5])
    selected=patrol.choose([0,0,0],[],[2,0,0],8,acquire=True,target_id=9)
    assert selected['observation_station']==2


def test_short_timing_budget_prefers_the_short_edge():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[.65,0,0],2:[1.6,0,0]}
    patrol.paths={0:[dict(to=i,length=length,corners=[[0,0,0],patrol.points[i]])
                    for i,length in [(1,.65),(2,1.6)]]}
    patrol.sight={9:{1,2}}
    selected=patrol.choose([0,0,0],[],[3,0,0],8,acquire=True,target_id=9,timing_seconds=.7)
    assert selected['navigation_position']==[.65,0,0]


def test_patrol_prefers_a_station_with_a_clear_observer_camera():
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.points={0:[0,0,0],1:[1.2,0,0],2:[0,0,1.2]}
    patrol.paths={0:[dict(to=i,length=1.2,corners=[[0,0,0],patrol.points[i]]) for i in [1,2]]}
    patrol.camera_stations={9:{'visible':{2},'hidden':{1}}}
    visible=patrol.choose([0,0,0],[],[3,0,0],8,target_id=9)
    hidden=patrol.choose([0,0,0],[],[3,0,0],8,target_id=9,hidden=True)
    assert visible['navigation_position']==[0,0,1.2]
    assert hidden['navigation_position']==[1.2,0,0]


def test_dynamic_camera_probe_avoids_an_occluded_route(monkeypatch):
    patrol=NativePatrol.__new__(NativePatrol)
    patrol.previous=None
    patrol.log_dir='unused'
    patrol.camera_route_probe=True
    patrol.points={0:[0,0,0],1:[1.2,0,0],2:[0,0,1.2]}
    patrol.paths={0:[dict(to=i,length=1.2,corners=[[0,0,0],patrol.points[i]]) for i in [1,2]]}
    monkeypatch.setattr('storm_virtualhome.native_navigation.camera_route_penalties',
                        lambda directory,routes:[3,0])
    selected=patrol.choose([0,0,0],[],[3,0,0],8,target_id=9)
    assert selected['navigation_position']==[0,0,1.2]
    patrol.sight={9:{1,2}}
    selected=patrol.choose([0,0,0],[],[3,0,0],8,target_id=9,acquire=True)
    assert selected['navigation_position']==[0,0,1.2]
