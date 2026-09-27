"""Check frozen route geometry and endpoint clearance."""
import pytest

from storm_virtualhome.motion import (
    assert_agent_clearance,
    build_motion_contract,
    offscreen_view_direction,
    patrol_cycle,
    point_segment_distance,
    role_aware_waypoint_score,
    route_turn_degrees,
    segment_segment_distance,
    select_patrol_waypoint,
    waypoint_escape_score,
    waypoint_is_navigable,
    waypoint_path_is_clear,
)


def node(key, name, x, z):
    return dict(id=key,class_name=name,properties=[],states=[],
                bounding_box=dict(center=[x,0,z],size=[.2,.2,.2]))


def test_route_geometry_has_expected_turn_and_clearance():
    assert route_turn_degrees([0,0,0],[1,0,0],[1,0,1]) == pytest.approx(90)
    assert point_segment_distance([1,0,1],[0,0,0],[2,0,0]) == pytest.approx(1)


def test_parallel_agent_routes_keep_clear_and_crossing_routes_do_not():
    assert segment_segment_distance([0,0,0],[0,0,4],[1.25,0,0],[1.25,0,4]) == pytest.approx(1.25)
    assert segment_segment_distance([0,0,0],[4,0,4],[0,0,4],[4,0,0]) == pytest.approx(0)


def test_offscreen_view_keeps_target_behind_and_follows_travel():
    direction=offscreen_view_direction([0,0,0],[2,0,0],[0,0,3])
    assert direction[2] > 0
    assert direction[0]*2+direction[2]*0 < 0


def test_motion_contract_freezes_safe_waypoints():
    graph=dict(nodes=[node(10,'cabinet',0,0),node(20,'table',2,0),
                      node(21,'desk',2,2),node(22,'sofa',0,2)],edges=[])
    events=[dict(object_id=10,surface_id=20,verb='Open',requested_visibility='in_view',
                 operator_approach_id=21),
            dict(object_id=10,surface_id=20,verb='Close',requested_visibility='offscreen')]
    contract=build_motion_contract(graph,events,[20,21,22],[-1,0,-1],
                                   rules=dict(stationary_operator_clearance_m=.5))
    assert len(contract['event_routes'])==2
    assert contract['event_routes'][0]['operator_approach_id']==21
    assert contract['event_routes'][1]['operator_approach_id']==10
    assert all(route['approach_waypoints'] for route in contract['event_routes'])
    assert contract['event_routes'][1]['hidden_waypoints']
    assert all(step['planned_turn_degrees']<=180 for route in contract['event_routes']
               for key in ('approach_waypoints','hidden_waypoints') for step in route[key])


def test_character_endpoint_overlap_is_rejected():
    graph=dict(nodes=[node(1,'character',0,0),node(2,'character',.4,0)],edges=[])
    with pytest.raises(ValueError,match='0.40 m apart'):
        assert_agent_clearance(graph,.85)


def test_patrol_cycle_reverses_at_endpoints_and_resumes_after_current_anchor():
    route=[{'anchor_id':20},{'anchor_id':21},{'anchor_id':22}]
    assert [step['anchor_id'] for step in patrol_cycle(route)]==[20,21,22,21]
    assert [step['anchor_id'] for step in patrol_cycle(route,22)]==[21,20,21,22]


def test_patrol_selection_keeps_a_partial_walk_destination_until_advanced():
    route=[{'anchor_id':20},{'anchor_id':21},{'anchor_id':22}]
    assert select_patrol_waypoint(route,1)['anchor_id']==21
    assert select_patrol_waypoint(route,0)['anchor_id']==21
    assert select_patrol_waypoint(route,1)['anchor_id']==22


def test_open_container_is_not_a_navigation_anchor():
    assert not waypoint_is_navigable({'properties':['CAN_OPEN'],'states':['OPEN']})
    assert waypoint_is_navigable({'properties':['CAN_OPEN'],'states':['CLOSED']})
    assert waypoint_is_navigable({'properties':['SURFACES'],'states':['OPEN']})


def test_waypoint_path_rejects_character_crossing_and_accepts_clear_detour():
    nodes={20:node(20,'table',2,0),21:node(21,'desk',0,2)}
    start=[0,0,0];occupied=[[1,0,0]]
    assert not waypoint_path_is_clear(nodes,{'anchor_id':20},start,occupied,.85,.75)
    assert waypoint_path_is_clear(nodes,{'anchor_id':21},start,occupied,.85,.75)


def test_waypoint_escape_allows_motion_away_from_a_nearby_character():
    nodes={20:node(20,'table',3,0),21:node(21,'desk',-3,0)}
    start=[0,0,0];occupied=[[1.3,0,0]]
    score=waypoint_escape_score(nodes,{'anchor_id':21},start,occupied,.85,1.7,.75)
    assert score is not None
    assert score[0] is False
    assert waypoint_escape_score(nodes,{'anchor_id':20},start,occupied,.85,1.7,.75) is None


def test_waypoint_escape_can_leave_runtime_margin_without_crossing_hard_limit():
    nodes={20:node(20,'table',-3,0)}
    score=waypoint_escape_score(nodes,{'anchor_id':20},[0,0,0],[[.92,0,0]],
                                1.0,1.2,.75,absolute_clearance=.85)
    assert score is not None
    assert score[0] is False


def test_manipulation_margin_applies_only_to_the_active_operator():
    nodes={20:node(20,'table',0,4)}
    start=[0,0,0];helper=[1.25,0,2];operator=[4,0,2]
    score=role_aware_waypoint_score(
        nodes,{'anchor_id':20},start,[helper,operator],operator,
        1.1,1.2,1.65,.75,absolute_clearance=.85)
    assert score is not None
    assert role_aware_waypoint_score(
        nodes,{'anchor_id':20},start,[helper,operator],helper,
        1.1,1.2,1.65,.75,absolute_clearance=.85) is None


def test_motion_contract_reserves_the_recorded_operator_endpoint():
    graph=dict(nodes=[node(10,'cabinet',0,0),node(20,'table',2,0),
                      node(21,'desk',2,2),node(22,'sofa',0,2)],edges=[])
    events=[dict(object_id=10,surface_id=20,verb='Open',requested_visibility='in_view',
                 operator_position=[2,0,2])]
    contract=build_motion_contract(graph,events,[20,21,22],[-1,0,-1],
                                   rules=dict(agent_clearance_m=.85,
                                              stationary_operator_clearance_m=1.0))
    anchors=[step['anchor_id'] for step in contract['event_routes'][0]['approach_waypoints']]
    assert 21 not in anchors


def test_motion_contract_reserves_the_target_and_approach_object():
    graph=dict(nodes=[node(10,'cabinet',0,0),node(11,'counter',2,0),
                      node(20,'table',.4,0),node(21,'desk',2.2,0),node(22,'sofa',0,2)],edges=[])
    events=[dict(object_id=10,surface_id=20,verb='Open',requested_visibility='in_view',
                 operator_approach_id=11,operator_position=[3,0,3])]
    contract=build_motion_contract(graph,events,[20,21,22],[-1,0,-1],
                                   rules=dict(stationary_operator_clearance_m=1.0))
    anchors=[step['anchor_id'] for step in contract['event_routes'][0]['approach_waypoints']]
    assert 20 not in anchors
    assert 21 not in anchors


def test_motion_contract_never_uses_the_current_workspace_as_an_anchor():
    graph=dict(nodes=[node(10,'cabinet',0,0),node(11,'counter',2,0),
                      node(20,'table',-2,0),node(21,'desk',0,2),node(22,'sofa',-2,2)],edges=[])
    events=[dict(object_id=10,surface_id=20,verb='Open',requested_visibility='offscreen',
                 operator_approach_id=11,operator_position=[3,0,3])]
    contract=build_motion_contract(graph,events,[10,11,20,21,22],[-1,0,-1],
                                   rules=dict(stationary_operator_clearance_m=.5))
    route=contract['event_routes'][0]
    anchors=[step['anchor_id'] for key in ('approach_waypoints','hidden_waypoints')
             for step in route[key]]
    assert 10 not in anchors
    assert 11 not in anchors
