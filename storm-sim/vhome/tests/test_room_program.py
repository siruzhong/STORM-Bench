"""Check legal randomized schedules and retained observation requirements."""
import copy
from storm_virtualhome.room_program import plan_room_routine
from storm_virtualhome.planning import create_program, validate_program


def fixture():
    nodes=[dict(id=i,class_name=name,properties=props,states=states) for i,name,props,states in [
        (10,'cabinet',['CAN_OPEN'],['CLOSED']),(11,'toilet',['CAN_OPEN'],['CLOSED']),
        (12,'faucet',['HAS_SWITCH'],['OFF']),(20,'sink',['SURFACES'],[])]]
    pool=[dict(id=10,actor=1,operations=['Open','Close'],surface_id=20,
               operator_approach_id=20,operator_find_solution=True,
               state_verification='native_recorded_animation'),
          dict(id=11,actor=2,operations=['Open','Close'],surface_id=20),
          dict(id=12,actor=1,operations=['SwitchOn','SwitchOff'],surface_id=20)]
    return dict(nodes=nodes,edges=[]),dict(seed=1,room_action_pool=pool,event_count=10,offscreen_event_count=5,
        event_interval_seconds=6,event_interval_tolerance_seconds=2.5,duration_range_seconds=[60,70])


def test_sparse_room_plan_is_legal_and_deterministic():
    graph,config=fixture();original=copy.deepcopy(graph)
    for seed in range(25):
        config['seed']=seed
        plan=create_program(graph,config)
        validate_program(plan,graph,config)
        assert plan==create_program(graph,config)
        assert len(plan['events'])==10
        assert len({e['object_id'] for e in plan['events']})==3
        assert sum(e['requested_visibility']=='offscreen' for e in plan['events'])==5
        assert plan['events'][-1]['requested_visibility']=='in_view'
        assert not any(first['requested_visibility']==second['requested_visibility']=='offscreen'
                       and first['injector_character']==second['injector_character']
                       and first['object_id']==second['object_id']
                       for first,second in zip(plan['events'],plan['events'][1:]))
        assert {e['injector_character'] for e in plan['events']}=={1,2}
        assert all(e['operator_approach_id']==20 for e in plan['events'] if e['object_id']==10)
        assert all(e['operator_find_solution'] for e in plan['events'] if e['object_id']==10)
        assert all(e['state_verification']=='native_recorded_animation'
                   for e in plan['events'] if e['object_id']==10)
        assert max(sum(e['object_id']==key for e in plan['events']) for key in (10,11,12))<=4
    assert graph==original


def test_room_plan_changes_order_and_hidden_event_types():
    graph,config=fixture();orders=set();hidden=set()
    for seed in range(25):
        config['seed']=seed;events=plan_room_routine(graph,config)
        orders.add(tuple((e['object_id'],e['verb']) for e in events))
        hidden.update(e['verb'] for e in events if e['requested_visibility']=='offscreen')
    assert len(orders)>10
    assert hidden=={'Open','Close','SwitchOn','SwitchOff'}


def test_delayed_operator_enters_by_the_fifth_event_and_finishes_its_pair():
    graph,config=fixture();config['defer_late_operators']=True
    for seed in range(25):
        config['seed']=seed;events=plan_room_routine(graph,config)
        first_actor=events[0]['injector_character']
        delayed=next(index for index,event in enumerate(events)
                     if event['injector_character']!=first_actor)
        assert delayed<=4
        assert events[delayed]['requested_visibility']=='offscreen'
        assert events[delayed+1]['requested_visibility']=='in_view'
        assert events[delayed+1]['injector_character']==events[delayed]['injector_character']
        assert events[delayed+1]['object_id']==events[delayed]['object_id']


def test_repeated_names_need_separated_reference_geometry():
    from storm_virtualhome.room_program import target_descriptions
    def node(key,name,x):
        return dict(id=key,class_name=name,bounding_box=dict(center=[x,0,0],size=[.2,.2,.2]))
    members=[node(1,'tablelamp',0),node(2,'tablelamp',4),node(3,'desk',-1)]
    labels=target_descriptions(members)
    assert labels[1]!=labels[2]
    assert 'closest to' in labels[1] and 'farthest from' in labels[2]
    members[1]['bounding_box']['center']=[.1,0,0]
    labels=target_descriptions(members)
    assert 1 not in labels and 2 not in labels


def test_small_props_can_use_room_specific_reference_objects():
    from storm_virtualhome.room_program import target_descriptions
    def node(key,name,x,size=.1):
        return dict(id=key,class_name=name,bounding_box=dict(center=[x,0,0],size=[size,size,size]))
    members=[node(1,'waterglass',0),node(2,'waterglass',.3),node(3,'washingmachine',2,1)]
    labels=target_descriptions(members)
    assert labels[1]!=labels[2]
    assert 'washing machine' in labels[1] and 'washing machine' in labels[2]


def test_individually_clear_target_is_retained_after_three_other_targets():
    from storm_virtualhome.room_program import describable_probe_targets, extreme_target_description
    def node(key,name,x,category='Objects'):
        return dict(id=key,class_name=name,category=category,
                    bounding_box=dict(center=[x,0,0],size=[.1,.1,.1]))
    members=[node(1,'towel',0),node(2,'towel',.1),node(3,'towel',4),
             node(4,'toilet',10),node(5,'lightswitch',8),node(6,'faucet',9)]
    graph=dict(nodes=[node(100,'bathroom',0,'Rooms'),*members],
               edges=[dict(from_id=n['id'],to_id=100,relation_type='INSIDE') for n in members])
    probe=dict(room_id=100,targets=[dict(id=n['id'],name=n['class_name'],operations=[dict(accepted=True)])
                                    for n in members if n['id'] in (3,4,5,6)])
    candidates,descriptions=describable_probe_targets(graph,probe)
    assert {t['id'] for t in candidates}=={3,4,5,6}
    assert descriptions[3]=='towel closest to the toilet'
    assert extreme_target_description(members,1) is None


def test_native_probe_readiness_accepts_only_describable_repeated_targets():
    from storm_virtualhome.room_program import native_probe_ready
    def node(key,name,x,category='Objects'):
        return dict(id=key,class_name=name,category=category,
                    bounding_box=dict(center=[x,0,0],size=[.4,.4,.4]))
    room=node(100,'bedroom',0,'Rooms')
    objects=[node(1,'cabinet',0),node(2,'cabinet',4),node(3,'computer',2),node(4,'bed',-1)]
    graph={'nodes':[room,*objects],
           'edges':[dict(from_id=item['id'],to_id=100,relation_type='INSIDE') for item in objects]}
    def target(item,first,second):
        return dict(id=item['id'],name=item['class_name'],operations=[dict(
            first=first,second=second,accepted=True,spawn=[0,0,0],approach_position=[0,0,0])])
    probe={'room_id':100,'targets':[target(objects[0],'Open','Close'),
                                    target(objects[1],'Open','Close'),
                                    target(objects[2],'SwitchOn','SwitchOff')]}
    assert native_probe_ready(graph,probe)
    graph['nodes'].remove(objects[3])
    graph['edges']=[edge for edge in graph['edges'] if edge['from_id']!=objects[3]['id']]
    assert not native_probe_ready(graph,probe)


def test_single_operator_uses_the_first_non_observer_character():
    from storm_virtualhome.room_program import normalize_operator_ids
    assignments = [dict(actor=2), dict(actor=2)]
    assert {item['actor'] for item in normalize_operator_ids(assignments)} == {1}
    assignments = [dict(actor=1), dict(actor=2)]
    assert [item['actor'] for item in normalize_operator_ids(assignments, first_actor=2)] == [2, 1]


def test_room_startup_budget_is_frozen_before_capture():
    graph,config=fixture()
    config['first_event_seconds']=4.0
    plan=create_program(graph,config)
    assert 3.7<=plan['events'][0]['planned_start_seconds']<=4.3
    assert 57.7<=plan['events'][-1]['planned_start_seconds']<=58.3
    validate_program(plan,graph,config)


def test_visibility_adaptive_cadence_reserves_hidden_travel():
    graph,config=fixture()
    config.update(first_event_seconds=4.0,visibility_adaptive_cadence=True,
                  in_view_event_interval_seconds=6.5,offscreen_event_interval_seconds=7.0,
                  event_interval_tolerance_seconds=3.0)
    plan=create_program(graph,config)
    for previous,event in zip(plan['events'],plan['events'][1:]):
        if previous['verb']=='Grab' and event['verb']=='PutBack' and previous['object_id']==event['object_id']:
            continue
        expected=7.0 if event['requested_visibility']=='offscreen' else 6.5
        assert event['planned_gap_seconds']==expected
    assert 63<=plan['events'][-1]['planned_start_seconds']<66
