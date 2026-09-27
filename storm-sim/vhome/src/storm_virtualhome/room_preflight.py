"""Propose native interaction tests without assuming a kitchen layout."""
import math
from .scene import position, room_id_of


def spawn_points(graph, room_id, spacing=.5):
    nodes = {n['id']: n for n in graph['nodes']}
    room = nodes[room_id]
    members = [n for n in graph['nodes'] if room_id_of(graph, n['id']) == room_id]
    floors = [n for n in members if n['class_name'] == 'floor']
    if not floors:
        return []
    floor_y = min(position(n)[1] for n in floors)
    center, size = room['bounding_box']['center'], room['bounding_box']['size']
    obstacles = [n for n in members if n['class_name'] not in ('floor', 'wall', 'ceiling')
                 and n.get('category', '').lower() != 'rooms'
                 and 'GRABBABLE' not in n.get('properties', [])
                 and position(n)[1] + n['bounding_box']['size'][1]/2 > floor_y+.1
                 and position(n)[1] - n['bounding_box']['size'][1]/2 < floor_y+1.7]
    result=[]
    for ix in range(1, int(size[0]/spacing)):
        for iz in range(1, int(size[2]/spacing)):
            point=[center[0]-size[0]/2+ix*spacing, floor_y, center[2]-size[2]/2+iz*spacing]
            if any(all(abs(point[a]-position(n)[a]) < n['bounding_box']['size'][a]/2+.35 for a in (0,2)) for n in obstacles):
                continue
            result.append(point)
    return result


def probe_candidates(room, limit=14):
    """Prefer visible household changes over tiny switches or decorative assets."""
    candidates=[n for n in room['candidates'] if n['unique_mask']]
    family_preferences={
        'SwitchOn':('tv','computer','lightswitch','microwave','fridge','dishwasher',
                    'washingmachine','radio','faucet','tablelamp'),
        'Open':('microwave','bathroomcabinet','cabinet','kitchencabinet','fridge',
                'dishwasher','washingmachine','toilet','closet','nightstand','desk'),
        'Grab':('cookingpot','fryingpan','dishbowl','box','book','pillow','towel','plate',
                'mug','waterglass','remotecontrol','barsoap','toothpaste','facecream','toothbrush'),
    }
    def rank(candidate,family):
        preference=family_preferences[family]
        return (preference.index(candidate['name']) if candidate['name'] in preference else len(preference),
                -max(candidate['bounds']['size']))
    families=('Open','SwitchOn','Grab')
    ranked={}
    for family in families:
        family_candidates=[candidate for candidate in candidates if family in candidate['operations']]
        family_candidates.sort(key=lambda candidate:rank(candidate,family))
        ranked[family]=family_candidates
    selected=[]
    for offset in range(max(2,limit//3)):
        for family in families:
            if offset>=len(ranked[family]):continue
            candidate=ranked[family][offset]
            if not any(item['id']==candidate['id'] for item in selected):
                selected.append(dict(candidate,probe_family=family))
    selected_ids={candidate['id'] for candidate in selected}
    remaining=sorted((candidate for candidate in candidates if candidate['id'] not in selected_ids),
                     key=lambda candidate:-max(candidate['bounds']['size']))
    selected.extend(remaining)
    return selected[:limit]


def probe_result_ready(targets, minimum_targets=3, minimum_verbs=4):
    accepted=[target for target in targets
              if any(operation.get('accepted') for operation in target.get('operations',[]))]
    verbs={verb for target in accepted for operation in target['operations']
           if operation.get('accepted') for verb in (operation['first'],operation['second'])}
    return len(accepted)>=minimum_targets and len(verbs)>=minimum_verbs


def switch_animation_ready(timing, minimum_shared_pixels=20, minimum_target_delta=.5):
    return (timing.get('recorded_frames',0)>=2
            and timing.get('target_shared_pixels',0)>=minimum_shared_pixels
            and timing.get('target_mean_frame_delta',0)>=minimum_target_delta)
