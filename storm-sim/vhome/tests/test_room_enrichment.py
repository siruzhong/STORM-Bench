"""Keep enrichment proposals separate from existing scene content."""
import copy
from storm_virtualhome.room_enrichment import propose_additions


def test_enrichment_preserves_scene_and_places_props_on_room_surfaces():
    room=dict(id=10,class_name='bedroom',category='Rooms',properties=[],states=[],bounding_box=dict(center=[0,1,0],size=[5,3,5]))
    desk=dict(id=20,class_name='desk',category='Furniture',properties=['SURFACES'],states=[],bounding_box=dict(center=[0,.4,0],size=[2,.8,1]))
    graph=dict(nodes=[room,desk],edges=[dict(from_id=20,to_id=10,relation_type='INSIDE')])
    mug=dict(id=30,class_name='mug',category='Props',properties=['GRABBABLE'],states=[],prefab_name='mug',
             bounding_box=dict(center=[3,.1,3],size=[.15,.2,.15]),obj_transform=dict(position=[3,0,3],rotation=[0,0,0,1],scale=[1,1,1]))
    original=copy.deepcopy(graph)
    additions=propose_additions(graph,{'mug':mug})
    assert graph==original
    assert len(additions['nodes'])==1
    added=additions['nodes'][0]
    assert added['id']>30 and added['bounding_box']['center'][1]>.8
    assert dict(from_id=added['id'],to_id=20,relation_type='ON') in additions['edges']
    assert dict(from_id=added['id'],to_id=10,relation_type='INSIDE') in additions['edges']


def test_missing_bedroom_cabinet_is_placed_against_a_clear_wall():
    room=dict(id=10,class_name='bedroom',category='Rooms',properties=[],states=[],
              bounding_box=dict(center=[0,1.5,0],size=[8,3,8]))
    floor=dict(id=11,class_name='floor',category='Structural',properties=[],states=[],
               bounding_box=dict(center=[0,0,0],size=[8,.1,8]))
    bed=dict(id=12,class_name='bed',category='Furniture',properties=['SITTABLE'],states=[],
             bounding_box=dict(center=[0,.5,0],size=[2,1,2]))
    graph=dict(nodes=[room,floor,bed],edges=[
        dict(from_id=11,to_id=10,relation_type='INSIDE'),
        dict(from_id=12,to_id=10,relation_type='INSIDE'),
    ])
    cabinet=dict(id=20,class_name='cabinet',category='Furniture',
                 properties=['SURFACES','CAN_OPEN','CONTAINERS'],states=['CLOSED'],
                 bounding_box=dict(center=[0,.5,0],size=[.4,1,1.2]),
                 obj_transform=dict(position=[0,0,0],rotation=[0,0,0,1],scale=[1,1,1]))
    additions=propose_additions(graph,{'cabinet':cabinet})
    added=additions['nodes']
    assert len(added)==2 and {node['class_name'] for node in added}=={'cabinet'}
    assert all(abs(node['bounding_box']['center'][0])>3 for node in added)
    assert all(placement['surface_id'] is None for placement in additions['placements'])
    assert all(dict(from_id=node['id'],to_id=10,relation_type='INSIDE') in additions['edges']
               for node in added)


def test_targeted_props_only_change_requested_rooms():
    rooms=[dict(id=i,class_name='livingroom',category='Rooms',properties=[],states=[],
                bounding_box=dict(center=[i*5,1.5,0],size=[4,3,4])) for i in (10,11)]
    tables=[dict(id=i+10,class_name='coffeetable',category='Furniture',properties=['SURFACES'],states=[],
                 bounding_box=dict(center=[(i-10)*5,.4,0],size=[2,.8,1])) for i in (10,11)]
    graph=dict(nodes=rooms+tables,edges=[
        dict(from_id=20,to_id=10,relation_type='INSIDE'),
        dict(from_id=21,to_id=11,relation_type='INSIDE'),
    ])
    mug=dict(id=30,class_name='mug',category='Props',properties=['GRABBABLE'],states=[],prefab_name='mug',
             bounding_box=dict(center=[0,.1,0],size=[.15,.2,.15]),
             obj_transform=dict(position=[0,0,0],rotation=[0,0,0,1],scale=[1,1,1]))
    glass=copy.deepcopy(mug);glass.update(id=31,class_name='waterglass',prefab_name='waterglass')
    additions=propose_additions(graph,{'mug':mug,'waterglass':glass},per_room=0,
                                room_prop_classes={11:('mug','waterglass')})
    assert [placement['room_id'] for placement in additions['placements']]==[11,11]
    assert [placement['class_name'] for placement in additions['placements']]==['mug','waterglass']


def test_native_identity_mapping_excludes_existing_objects():
    from storm_virtualhome.room_enrichment import match_added_nodes
    old=dict(id=10,class_name='mug',bounding_box=dict(center=[0,1,0]))
    requested=dict(id=1010,class_name='mug',bounding_box=dict(center=[.2,1,0]))
    native=dict(id=11,class_name='mug',bounding_box=dict(center=[.21,1,0]))
    assert match_added_nodes({'nodes':[old]}, {'nodes':[old,native]}, {'nodes':[requested]}) == {'1010':11}


def test_ambiguous_native_identity_is_rejected():
    import pytest
    from storm_virtualhome.room_enrichment import match_added_nodes
    requested=dict(id=1010,class_name='mug',bounding_box=dict(center=[0,1,0]))
    native=[dict(id=i,class_name='mug',bounding_box=dict(center=[0,1,0])) for i in (11,12)]
    with pytest.raises(ValueError,match='Ambiguous'):
        match_added_nodes({'nodes':[]}, {'nodes':native}, {'nodes':[requested]})
