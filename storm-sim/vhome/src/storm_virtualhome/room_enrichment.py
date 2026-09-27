"""Place a small number of household props without moving existing furniture."""
import copy
import math
from collections import Counter
from .scene import position, room_id_of


ALLOWED = {
    'bedroom': ('waterglass','mug','radio','tablelamp','remotecontrol','clock','plate','book'),
    'livingroom': ('mug','waterglass','radio','tablelamp','remotecontrol','clock','candle','dishbowl','plate','book'),
    'kitchen': ('mug','waterglass','dishbowl','plate','coffeepot'),
    'bathroom': ('waterglass','barsoap','toothbrush','facecream'),
}

FLOOR_FURNITURE = {
    'bedroom': ('cabinet',),
}


def _place_node(template, node_id, goal):
    node=copy.deepcopy(template);node['id']=node_id
    shift=[goal[a]-node['bounding_box']['center'][a] for a in range(3)]
    node['bounding_box']['center']=list(goal)
    node['obj_transform']['position']=[node['obj_transform']['position'][a]+shift[a] for a in range(3)]
    return node


def _floor_furniture_goal(room, members, additions, template):
    center,size=room['bounding_box']['center'],room['bounding_box']['size']
    object_size=template['bounding_box']['size']
    floors=[node for node in members if node['class_name']=='floor']
    if not floors:return None
    floor_y=min(position(node)[1] for node in floors)
    x_bounds=(center[0]-size[0]/2+object_size[0]/2+.15,
              center[0]+size[0]/2-object_size[0]/2-.15)
    z_min=center[2]-size[2]/2+object_size[2]/2+.15
    z_max=center[2]+size[2]/2-object_size[2]/2-.15
    steps=max(1,int((z_max-z_min)/.25))
    candidates=[]
    for offset_step in range(5):
        offset=offset_step*.25
        for side,x in enumerate(x_bounds):
            x=x+offset if side==0 else x-offset
            for step in range(steps+1):
                z=z_min+(z_max-z_min)*step/steps
                candidates.append((offset,[x,floor_y+object_size[1]/2,z]))
    obstacles=[node for node in members+additions['nodes']
               if node['class_name'] not in ('wall','floor','ceiling')
               and node.get('category','').lower()!='rooms']
    valid=[]
    for offset,goal in candidates:
        if any(all(abs(goal[a]-position(node)[a])
                   <(object_size[a]+node['bounding_box']['size'][a])/2+.12
                   for a in range(3)) for node in obstacles):
            continue
        clearance=min(math.dist(goal,position(node)) for node in obstacles) if obstacles else 10
        valid.append((-offset,clearance,goal))
    return max(valid)[2] if valid else None


def propose_additions(graph, templates, per_room=2, room_prop_classes=None):
    additions={'nodes':[],'edges':[],'placements':[]}
    next_id=max(n['id'] for n in graph['nodes'])+1000
    for room in graph['nodes']:
        if room.get('category','').lower()!='rooms':continue
        members=[n for n in graph['nodes'] if room_id_of(graph,n['id'])==room['id']]
        names={n['class_name'] for n in members}
        counts=Counter(n['class_name'] for n in members)
        if room_prop_classes is None:
            classes=[name for name in ALLOWED.get(room['class_name'],()) if name not in names and name in templates]
            # Prefer missing prop classes, then admit a duplicate only on a separated surface.
            classes += [name for name in ALLOWED.get(room['class_name'],()) if name in names and name in templates]
            limit=per_room
        else:
            requested=room_prop_classes.get(room['id'],())
            invalid=[name for name in requested if name not in ALLOWED.get(room['class_name'],()) or name not in templates]
            if invalid:raise ValueError(f"Unsupported props for room {room['id']}: {invalid}")
            classes=list(requested)
            limit=len(classes)
        surfaces=[n for n in members if n['class_name'] in ('desk','coffeetable','kitchentable','kitchencounter','bathroomcounter','nightstand','tvstand')
                  and 'SURFACES' in n.get('properties',[]) and .30<position(n)[1]+n['bounding_box']['size'][1]/2<1.35]
        used=0
        for name in classes:
            if used>=limit:break
            template=templates[name];size=template['bounding_box']['size']
            if max(size)>.65:continue
            proposal=None
            for surface in surfaces:
                center=position(surface);extent=surface['bounding_box']['size']
                offsets=[(u,v) for u in (0,-.2,.2,-.38,.38)
                         for v in (0,-.2,.2,-.38,.38)]
                offsets.sort(key=lambda pair:(abs(pair[0])+abs(pair[1]),pair))
                for u,v in offsets:
                    x,z=center[0]+u*extent[0],center[2]+v*extent[2]
                    if abs(u)*extent[0]+size[0]/2+.03>extent[0]/2 or abs(v)*extent[2]+size[2]/2+.03>extent[2]/2:continue
                    top=center[1]+extent[1]/2
                    goal=[x,top+size[1]/2+.015,z]
                    obstacles=[n for n in members+additions['nodes'] if n['id']!=surface['id'] and n['class_name'] not in ('wall','floor','ceiling') and n.get('category','').lower()!='rooms']
                    if any(all(abs(goal[a]-position(n)[a])<(size[a]+n['bounding_box']['size'][a])/2+.04 for a in range(3)) for n in obstacles):continue
                    proposal=(surface,goal);break
                if proposal:break
            if proposal is None:continue
            surface,goal=proposal
            node=_place_node(template,next_id,goal);next_id+=1
            if 'HAS_SWITCH' in node['properties']:node['states']=['OFF' if s=='ON' else s for s in node['states']]
            additions['nodes'].append(node)
            additions['edges'] += [dict(from_id=node['id'],to_id=room['id'],relation_type='INSIDE'),dict(from_id=node['id'],to_id=surface['id'],relation_type='ON')]
            additions['placements'].append(dict(object_id=node['id'],class_name=name,room_id=room['id'],surface_id=surface['id']))
            used+=1
        for name in FLOOR_FURNITURE.get(room['class_name'],()):
            if name not in templates:continue
            additions_needed=2 if counts[name]==0 else 1
            for _ in range(additions_needed):
                goal=_floor_furniture_goal(room,members,additions,templates[name])
                if goal is None:break
                node=_place_node(templates[name],next_id,goal);next_id+=1
                additions['nodes'].append(node)
                additions['edges'].append(dict(from_id=node['id'],to_id=room['id'],relation_type='INSIDE'))
                additions['placements'].append(dict(object_id=node['id'],class_name=name,
                                                     room_id=room['id'],surface_id=None))
    return additions


def apply_additions(client, additions, allow_partial=False):
    before=client.graph()
    graph=copy.deepcopy(before)
    existing={n['id'] for n in graph['nodes']}
    if existing & {n['id'] for n in additions['nodes']}:raise ValueError('Added object IDs overlap existing IDs')
    graph['nodes']+=additions['nodes'];graph['edges']+=additions['edges']
    import json
    config=dict(randomize=False,random_seed=0,animate_character=False,ignore_obstacles=False,transfer_transform=True)
    try:
        response=client.command('expand_scene',strings=[json.dumps(config),json.dumps(graph)])
    except Exception as exc:
        if not allow_partial:raise
        try:detail=json.loads(str(exc).split(': ',1)[1])
        except (ValueError,IndexError):raise exc
        if set(detail)!={'unplaced'}:raise
        response=dict(success=False,placement_failures=detail['unplaced'])
    after=client.graph();nodes={n['id']:n for n in after['nodes']}
    for old in before['nodes']:
        if old['id'] not in nodes or math.dist(position(old),position(nodes[old['id']]))>.05:
            raise ValueError(f"Existing layout changed at object {old['id']}")
    mapping = match_added_nodes(before, after, additions, allow_partial)
    expected = additions.get('id_mapping')
    if expected is not None and mapping != expected:
        raise ValueError(f'Prop identity changed on replay: {mapping} != {expected}')
    return after, dict(native_response=response, id_mapping=mapping)


def match_added_nodes(before, after, additions, allow_partial=False):
    """Match native IDs by class and position without reusing an existing object."""
    existing = {node['id'] for node in before['nodes']}
    available = {node['id']: node for node in after['nodes'] if node['id'] not in existing}
    mapping = {}
    for added in additions['nodes']:
        matches = [node for node in available.values()
                   if node['class_name'] == added['class_name']
                   and math.dist(position(added), position(node)) <= .15]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous added prop: {added['id']}")
        if not matches:
            if allow_partial:
                continue
            raise ValueError(f"Added prop missing: {added['id']}")
        actual = matches[0]
        mapping[str(added['id'])] = actual['id']
        del available[actual['id']]
    if available:
        raise ValueError(f'Unexpected scene objects: {sorted(available)}')
    return mapping


def reset_enriched_scene(client, scene, additions=None):
    """Restore the frozen scene before native actions or capture."""
    client.reset(scene)
    if additions and additions['nodes']:
        apply_additions(client, additions, allow_partial=False)
