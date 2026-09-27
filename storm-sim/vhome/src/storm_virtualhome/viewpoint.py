"""Derive a new camera configuration without resampling recorded event plans."""
import copy
from .planning import digest, planning_config


def first_person_plan(config, program):
    """Preserve event identities, order, windows, and routes under a new camera."""
    config = copy.deepcopy(config)
    program = copy.deepcopy(program)
    if config.get('viewpoint') == 'first_person':
        return config, program
    parent = program['plan_id']
    config['viewpoint'] = 'first_person'
    config['camera'].update(position=[0.0, 1.6, 0.0], rotation=[0.0, 0.0, 0.0])
    rules = config['camera_continuity']
    rules.update(viewpoint='first_person', position_smooth_seconds=.12)
    floor = config['observer']['position'][1]
    rules['position_min'][1] = floor + 1.35
    rules['position_max'][1] = floor + 1.8
    program['config_signature'] = digest(planning_config(config))
    program['derived_from_plan_id'] = parent
    program['plan_id'] = digest({k: v for k, v in program.items() if k != 'plan_id'})
    return config, program
