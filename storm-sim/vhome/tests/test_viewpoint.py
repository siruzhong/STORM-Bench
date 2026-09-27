"""Preserve frozen event semantics when deriving a first-person recording."""
from copy import deepcopy
from storm_virtualhome.planning import digest, planning_config
from storm_virtualhome.viewpoint import first_person_plan


def test_camera_conversion_keeps_events_and_does_not_edit_source():
    config = dict(camera={'position':[.35,2.2,-1.7], 'rotation':[28,0,0]},
                  camera_continuity={'position_min':[-2,1.8,-3], 'position_max':[2,2.28,3]},
                  observer={'position':[0,0,0]}, seed=42)
    program = dict(plan_id='source', events=[{'event_index':0,'object_id':23,'verb':'Open'}], seed=42)
    source_config, source_program = deepcopy(config), deepcopy(program)
    converted, derived = first_person_plan(config, program)
    assert config == source_config and program == source_program
    assert converted['camera']['position'] == [0,1.6,0]
    assert converted['camera_continuity']['position_min'] == [-2,1.35,-3]
    assert derived['events'] == source_program['events']
    assert derived['derived_from_plan_id'] == 'source'
    assert derived['config_signature'] == digest(planning_config(converted))
    assert derived['plan_id'] == digest({k:v for k,v in derived.items() if k != 'plan_id'})
    assert first_person_plan(converted, derived) == (converted, derived)
