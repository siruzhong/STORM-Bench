"""Plan room routines from reversible operations verified in the native simulator."""
import copy
import math
import random
import itertools
from collections import Counter
from .mixed import STATE_EFFECTS
from .qa import label
from .room_preflight import spawn_points
from .scene import position, room_id_of, support_of
from .motion import build_motion_contract, planar_distance


def normalize_operator_ids(assignments, first_actor=None):
    """Keep native character IDs contiguous and start with operator one."""
    actors = {assignment['actor'] for assignment in assignments}
    if actors == {2}:
        for assignment in assignments:
            assignment['actor'] = 1
    elif actors == {1, 2} and first_actor == 2:
        for assignment in assignments:
            assignment['actor'] = 3 - assignment['actor']
    return assignments


def plan_room_routine(graph, config):
    rng=random.Random(config['seed'])
    nodes={n['id']:n for n in graph['nodes']}
    pool=config['room_action_pool']
    pending=[]
    sampled=list(pool)
    while len(sampled)<5:
        eligible=[t for t in pool if sum(x['id']==t['id'] for x in sampled)<2]
        sampled.append(rng.choice(eligible))
    rng.shuffle(sampled)
    previous={}
    for index,target in enumerate(sampled):
        actor=target['actor']
        for phase,verb in enumerate(target['operations']):
            key=f"pair_{index}_{phase}"
            pending.append(dict(task=key,verb=verb,object_id=target['id'],object=nodes[target['id']]['class_name'],
                event_family='state_change' if verb in STATE_EFFECTS else 'movement',
                event_action=STATE_EFFECTS[verb][2] if verb in STATE_EFFECTS else 'move',operation=verb,
                injector_character=actor,injector_node_id=actor+1,surface_id=target['surface_id'],
                surface=nodes[target['surface_id']]['class_name'],
                operator_approach_id=target.get('operator_approach_id',target['id']),
                operator_position=target.get('operator_position'),
                measured_operation_seconds=target.get('operation_video_seconds',{}).get(verb),
                operator_find_solution=target.get('operator_find_solution',False),
                state_verification=target.get('state_verification','graph_state'),
                depends_on=[f"pair_{index}_0"] if phase else ([previous[target['id']]] if target['id'] in previous else [])))
        previous[target['id']]=f"pair_{index}_1"
    events=[];done=set();held={}
    while pending:
        eligible=[e for e in pending if set(e['depends_on'])<=done and
                  (e['injector_character'] not in held or
                   (e['verb']=='PutBack' and held[e['injector_character']]==e['object_id']))]
        if not eligible:raise ValueError('No hand-safe event ordering exists')
        if config.get('defer_late_operators') and events:
            used={event['injector_character'] for event in events}
            if len(events)<4:
                existing=[event for event in eligible
                          if event['injector_character'] in used]
                if existing:
                    eligible=existing
            else:
                unseen=[event for event in eligible
                        if event['injector_character'] not in used]
                if unseen:
                    eligible=unseen
            previous_event=events[-1]
            first_index=next(index for index,event in enumerate(events)
                             if event['injector_character']==previous_event['injector_character'])
            if first_index>0 and first_index==len(events)-1:
                paired=[event for event in eligible
                        if event['injector_character']==previous_event['injector_character']
                        and event['object_id']==previous_event['object_id']]
                if paired:
                    eligible=paired
        def travel_cost(candidate):
            earlier=events[-1] if events else None
            travel=0 if earlier is None else math.dist(position(nodes[earlier['object_id']]),
                                                        position(nodes[candidate['object_id']]))
            repeat_actor=.35 if events and events[-1]['injector_character']==candidate['injector_character'] else 0
            reversal=.75 if events and events[-1]['object_id']==candidate['object_id'] else 0
            return travel+repeat_actor+reversal+rng.random()*.35
        if not events:
            readable=[event for event in eligible
                      if sorted(nodes[event['object_id']].get('bounding_box',{}).get('size',[0,0,0]))[-1]
                      *sorted(nodes[event['object_id']].get('bounding_box',{}).get('size',[0,0,0]))[-2]>=.025]
            eligible=readable or eligible
        event=min(eligible,key=travel_cost)
        events.append(event);pending.remove(event);done.add(event['task'])
        if event['verb']=='Grab':held[event['injector_character']]=event['object_id']
        elif event['verb']=='PutBack':del held[event['injector_character']]
    required_hidden=set()
    required_visible=set()
    if config.get('defer_late_operators'):
        for actor in {event['injector_character'] for event in events}:
            first=next(index for index,event in enumerate(events)
                       if event['injector_character']==actor)
            if first:
                required_hidden.add(first)
                if (first+1<len(events)
                        and events[first+1]['injector_character']==actor
                        and events[first+1]['object_id']==events[first]['object_id']):
                    required_visible.add(first+1)
    remaining=[index for index in range(len(events))
               if index not in (required_hidden|required_visible)]
    needed=config['offscreen_event_count']-len(required_hidden)
    if needed<0:
        raise ValueError('Delayed operators exceed the offscreen event budget')
    choices=[]
    for candidate in itertools.combinations(remaining,needed):
        selected=required_hidden|set(candidate)
        adjacent=sum(index-1 in selected for index in selected)
        repeated=sum(index-1 in selected
                     and events[index-1]['injector_character']==events[index]['injector_character']
                     and events[index-1]['object_id']==events[index]['object_id']
                     for index in selected)
        first_phase=sum(not events[index]['task'].endswith('_1')
                        for index in selected-required_hidden)
        penalty=(int(len(events)-1 in selected),repeated,first_phase,adjacent)
        choices.append((penalty,candidate))
    if not choices:
        raise ValueError('No valid offscreen event allocation exists')
    best=min(item[0] for item in choices)
    hidden=required_hidden|set(rng.choice([item[1] for item in choices if item[0]==best]))
    for index,event in enumerate(events):
        if index and event['injector_character']==events[index-1]['injector_character']:
            event['operator_transfer_distance_m']=round(planar_distance(
                position(nodes[events[index-1]['object_id']]),
                position(nodes[event['object_id']])),3)
        else:
            event['operator_transfer_distance_m']=0.0
        event['requested_visibility']='offscreen' if index in hidden else 'in_view'
        contract=config.get('motion_contract')
        if contract:
            route=contract['event_routes'][index]
            event['observer_approach_waypoints']=route['approach_waypoints']
            event['observer_hidden_waypoints']=route['hidden_waypoints']
            event['observer_action_waypoint']=route['action_waypoint']
            event['operator_approach_id']=route['operator_approach_id']
    return events


def target_descriptions(members):
    counts=Counter(n['class_name'] for n in members)
    descriptions={n['id']:label(n['class_name']) for n in members if counts[n['class_name']]==1}
    anchors=[n for n in members if counts[n['class_name']]==1 and n['class_name'] in
             ('bed','desk','sofa','tv','cabinet','bathroomcabinet','bathroomcounter',
              'bathtub','toilet','washingmachine','fridge','microwave','kitchencounter',
              'coffeetable','nightstand','door')]
    for name,count in counts.items():
        if count<2 or not anchors:continue
        group=[n for n in members if n['class_name']==name]
        ranked=[]
        for anchor in anchors:
            distances=sorted((math.dist(position(n),position(anchor)),n['id']) for n in group)
            margin=min(distances[1][0]-distances[0][0],distances[-1][0]-distances[-2][0])
            ranked.append((margin,anchor['id'],distances))
        margin,anchor_id,distances=max(ranked)
        footprint=max(max(n.get('bounding_box',{}).get('size',[.2,.2,.2])[a] for a in (0,2)) for n in group)
        minimum_margin=max(.22,min(.4,footprint*1.5))
        if margin<minimum_margin:continue
        anchor=next(n for n in anchors if n['id']==anchor_id)
        descriptions[distances[0][1]]=f"{label(name)} closest to the {label(anchor['class_name'])}"
        descriptions[distances[-1][1]]=f"{label(name)} farthest from the {label(anchor['class_name'])}"
    return descriptions


def extreme_target_description(members, target_id):
    """Describe a repeated target by a unique room anchor when an extreme is unambiguous."""
    counts=Counter(n['class_name'] for n in members)
    target=next(n for n in members if n['id']==target_id)
    group=[n for n in members if n['class_name']==target['class_name']]
    if len(group)<2:return label(target['class_name'])
    anchors=[n for n in members if counts[n['class_name']]==1 and n['class_name'] in
             ('bed','desk','sofa','tv','cabinet','bathroomcabinet','bathroomcounter',
              'bathtub','toilet','washingmachine','fridge','microwave','kitchencounter',
              'coffeetable','nightstand','door')]
    choices=[]
    for anchor in anchors:
        distances=sorted((math.dist(position(n),position(anchor)),n['id']) for n in group)
        if distances[0][1]==target_id:
            choices.append((distances[1][0]-distances[0][0],'closest to',anchor))
        if distances[-1][1]==target_id:
            choices.append((distances[-1][0]-distances[-2][0],'farthest from',anchor))
    if not choices:return None
    margin,relation,anchor=max(choices,key=lambda item:item[0])
    footprint=max(max(n.get('bounding_box',{}).get('size',[.2,.2,.2])[a] for a in (0,2)) for n in group)
    if margin<max(.22,min(.4,footprint*1.5)):return None
    return f"{label(target['class_name'])} {relation} the {label(anchor['class_name'])}"


def describable_probe_targets(graph, probe):
    """Return accepted targets that can be named without visual ambiguity."""
    nodes={n['id']:n for n in graph['nodes']}
    room_id=probe['room_id'];room=nodes[room_id]
    members=[n for n in graph['nodes'] if room_id_of(graph,n['id'])==room_id]
    accepted=[t for t in probe['targets'] if t['id'] in nodes
              and nodes[t['id']]['class_name']==t.get('name')
              and any(o['accepted'] for o in t['operations'])]
    descriptions=target_descriptions(members)
    candidates=[t for t in accepted if t['id'] in descriptions]
    for target in accepted:
        if target['id'] in descriptions:continue
        description=extreme_target_description(members,target['id'])
        if description is None:continue
        descriptions[target['id']]=description;candidates.append(target)
    return candidates,descriptions


def native_probe_ready(graph, probe, minimum_targets=3, minimum_verbs=4):
    candidates,_=describable_probe_targets(graph,probe)
    if len(candidates)<minimum_targets:return False
    for size in range(min(5,len(candidates)),minimum_targets-1,-1):
        for group in itertools.combinations(candidates,size):
            verbs={verb for target in group for operation in target['operations'] if operation['accepted']
                   for verb in (operation['first'],operation['second'])}
            positioned=all(any(operation.get('accepted')
                               and (operation.get('spawn') or target.get('spawn')) is not None
                               and (operation.get('approach_position') or target.get('approach_position')) is not None
                               for operation in target['operations']) for target in group)
            if len(verbs)>=minimum_verbs and positioned:return True
    return False


def build_native_profile(graph, template, probe, seed):
    rng=random.Random(seed)
    nodes={n['id']:n for n in graph['nodes']}
    room_id=probe['room_id'];room=nodes[room_id]
    members=[n for n in graph['nodes'] if room_id_of(graph,n['id'])==room_id]
    counts=Counter(n['class_name'] for n in members)
    candidates,descriptions=describable_probe_targets(graph,probe)
    if len(candidates)<3:raise ValueError(f"Only {len(candidates)} distinct named native targets passed; three are required")
    subsets=[]
    for size in range(min(5,len(candidates)),2,-1):
        for group in itertools.combinations(candidates,size):
            verbs={verb for target in group for operation in target['operations'] if operation['accepted']
                   for verb in (operation['first'],operation['second'])}
            if len(verbs)<4:continue
            spread=sum(math.dist(position(nodes[a['id']]),position(nodes[b['id']]))
                       for a,b in itertools.combinations(group,2))
            small=sum(sorted(nodes[target['id']]['bounding_box']['size'])[-1]
                      *sorted(nodes[target['id']]['bounding_box']['size'])[-2]<.012
                      and nodes[target['id']]['class_name']!='lightswitch' for target in group)
            floor_returns=sum(
                any(operation['accepted'] and operation['second']=='PutBack' for operation in target['operations'])
                and (support_of(graph,target['id']) or {}).get('class_name')=='floor'
                for target in group)
            # Favor a compact routine before adding more independently reachable objects.
            # Floor placement remains a fallback when a room has few usable props.
            compactness=spread/(size*(size-1)/2)+1.2*(size-3)+rng.random()*.5
            subsets.append((floor_returns,small,-len(verbs),compactness,group))
    if not subsets:raise ValueError('Native targets do not supply two operation pairs')
    selected=list(min(subsets,key=lambda item:item[:4])[4])
    cluster=min(selected,key=lambda t:sum(math.dist(position(nodes[t['id']]),position(nodes[u['id']])) for u in selected))
    selected.sort(key=lambda t:math.dist(position(nodes[t['id']]),position(nodes[cluster['id']])))
    surfaces=[n for n in members if 'SURFACES' in n.get('properties',[]) and n['class_name'] not in ('floor','ceiling','wall')]
    if not surfaces:raise ValueError('No support surface exists in the selected room')
    assignments=[]
    used_verbs=set()
    for index,target in enumerate(selected):
        possible=[o for o in target['operations'] if o['accepted']]
        operation=max(possible,key=lambda item:(len({item['first'],item['second']}-used_verbs),rng.random()))
        used_verbs.update((operation['first'],operation['second']))
        support=support_of(graph,target['id']) or min(surfaces,key=lambda n:math.dist(position(n),position(nodes[target['id']])))
        operator_spawn=operation.get('spawn') or target.get('spawn')
        operator_position=operation.get('approach_position') or target.get('approach_position')
        if operator_spawn is None or operator_position is None:
            raise ValueError(f"Native target {target['id']} has no verified operator position")
        assignments.append(dict(id=target['id'],actor=1 if math.dist(position(nodes[target['id']]),position(nodes[selected[0]['id']]))<=math.dist(position(nodes[target['id']]),position(nodes[selected[-1]['id']])) else 2,operations=[operation['first'],operation['second']],surface_id=support['id'],
                                operator_approach_id=operation.get('approach_id',target.get('approach_id',target['id'])),
                                operator_find_solution=operation.get('find_solution',False),
                                state_verification=operation.get('state_verification','graph_state'),
                                operation_video_seconds=operation.get('operation_video_seconds',{}),
                                operator_resource=operation.get('character_resource','Chars/Male1'),
                                operator_spawn=operator_spawn,
                                operator_position=operator_position))
    if len({v for a in assignments for v in a['operations']})<4:raise ValueError('Native targets do not supply two operation pairs')
    config=copy.deepcopy(template)
    config.update(scene=probe['scene'],room_id=room_id,room_name=room['class_name'],seed=seed,
                  episode_id=f"scene_{probe['scene']}_room_{room_id}_seed_{seed}",
                  room_action_pool=assignments,room_adaptive_view=True,scene_signature_mode='native_room',
                  minimum_distinct_operations=4,minimum_distinct_objects=len(selected),first_event_seconds=5.5,
                  timing_policy='planned_start_windows_without_retiming',
                  visibility_adaptive_cadence=True,in_view_event_interval_seconds=4.5,
                  offscreen_event_interval_seconds=8.0,event_interval_tolerance_seconds=2.5,
                  object_transition_interval_extra_seconds=2.0,event_span_seconds=54.5,
                  operator_transfer_adaptive_cadence=True,
                  operator_transfer_distance_threshold_m=3.0,
                  operator_transfer_seconds_per_meter=.75,
                  operator_transfer_max_extra_seconds=1.5,
                  defer_late_operators=True,
                  observer_turns_during_manipulation=False,
                  delayed_operator_first_event_turn_only=False)
    config['object_descriptions']={t['id']:descriptions[t['id']] for t in selected}
    config['evidence']['min_visible_pixels_by_class']={'lightswitch':12}
    config['render']['time_scale']=2.4
    config['viewpoint']='first_person'
    config['camera'].update(position=[0.0,1.6,0.0],rotation=[0.0,0.0,0.0])
    config['surface_id']=assignments[0]['surface_id']
    config['mixed_candidates']=dict(portable=[],cabinets=[],appliance=selected[0]['id'],switch=selected[1]['id'],secondary_surface_id=surfaces[0]['id'])
    anchors=[n for n in members if n['class_name'] in
             ('desk','cabinet','coffeetable','kitchentable','kitchencounter','sofa',
              'tvstand','bench','bathroomcabinet','bathroomcounter','sink',
              'bathroomsink','bathtub','toilet','washingmachine','bed','nightstand',
              'door','bookshelf','wallshelf','closet')]
    if len(anchors)<2:raise ValueError('Two distinct patrol anchors are required')
    target_zone_span=max(math.dist(position(nodes[first['id']]),position(nodes[second['id']]))
                         for first,second in itertools.combinations(assignments,2))
    # Compact workspaces do not need a second idle body blocking the observer.
    same_avatar=len({assignment['operator_resource'] for assignment in assignments})==1
    if target_zone_span<3.0 and same_avatar:
        for assignment in assignments:assignment['actor']=1
    elif len({assignment['actor'] for assignment in assignments})<2:
        assignments[-1]['actor']=2 if assignments[0]['actor']==1 else 1
    normalize_operator_ids(assignments)
    active_actors=sorted({assignment['actor'] for assignment in assignments})
    config['active_operator_characters']=active_actors
    config['operator_zone_span_m']=round(target_zone_span,3)
    role_by_actor={1:'injector',2:'helper'}
    config['character_roles']=['observer',*(role_by_actor[actor] for actor in active_actors)]
    config['monitor_ids']=[1,*(actor+1 for actor in active_actors)]
    config['room_patrol_anchors']=[n['id'] for n in sorted(anchors,key=lambda n:math.dist(position(n),position(nodes[cluster['id']])))]
    points=spawn_points(graph,room_id)
    preview=plan_room_routine(graph,config)
    if preview[0]['injector_character']==2:
        normalize_operator_ids(assignments,first_actor=2)
        preview=plan_room_routine(graph,config)
    first_event_index={actor:next(index for index,event in enumerate(preview)
                                  if event['injector_character']==actor)
                       for actor in active_actors}
    config['delayed_operator_characters']=[
        actor for actor,index in first_event_index.items() if index>0]
    occupied=[]
    for role,actor in [('injector',1),('helper',2)]:
        if actor not in active_actors:
            continue
        target_id=next(e['object_id'] for e in preview if e['injector_character']==actor)
        assignment=next(item for item in assignments if item['id']==target_id)
        fallback=sorted(points,key=lambda point:math.dist(point,position(nodes[target_id])))
        choices=[assignment['operator_position'],assignment['operator_spawn'],*fallback]
        choices=[p for p in choices if all(planar_distance(p,other)>=1.2 for other in occupied)]
        if not choices:raise ValueError('Verified operator positions overlap')
        chosen=list(choices[0])
        chosen[1]=assignment['operator_spawn'][1]
        resources={item['operator_resource'] for item in assignments if item['actor']==actor}
        if len(resources)!=1:raise ValueError('One operator cannot mix avatar-specific calibrations')
        config[role].update(room=room['class_name'],position=chosen,resource=resources.pop())
        occupied.append(chosen)
    points=[p for p in points if all(math.dist(p,other)>=1.3 for other in occupied)]
    if not points:raise ValueError('No observer position clear of both operators')
    chosen=min(points,key=lambda p:sum(abs(math.dist(p,other)-2.5) for other in occupied))
    config['observer'].update(room=room['class_name'],position=chosen,resource='Chars/Female1')
    preview=plan_room_routine(graph,config)
    config['motion_contract']=build_motion_contract(
        graph,preview,config['room_patrol_anchors'],chosen,
        rules=dict(agent_clearance_m=.85,stationary_operator_clearance_m=1.0,
                   runtime_hard_clearance_m=1.1,
                   dynamic_path_clearance_m=1.20,
                   operation_start_clearance_m=1.45,
                   operation_path_clearance_m=1.45,
                   minimum_observer_step_m=.75,short_relocation_minimum_step_m=.5,
                   maximum_route_turn_degrees=180.0,
                   waypoints_per_phase=6,low_motion_anchor_budget_seconds=.5))
    active_roles=[role_by_actor[actor] for actor in active_actors]
    for first,second in itertools.combinations(active_roles,2):
        if planar_distance(config[first]['position'],config[second]['position'])<config['motion_contract']['rules']['agent_clearance_m']:
            raise ValueError('Operator start positions violate the motion clearance contract')
    config['camera_continuity'].update(
        viewpoint='first_person',max_speed_mps=2.6,max_turn_rate_dps=80.0,position_smooth_seconds=.12,
        goal_inertia_weight=.25,
        rotation_smooth_seconds=.28,heading_smooth_seconds=.22,acceleration_mps2=6.0,
        max_p95_linear_acceleration_mps2=8.0,max_p95_angular_acceleration_dps2=500.0,
        max_boundary_rotation_degrees=4.5)
    center,size=room['bounding_box']['center'],room['bounding_box']['size']
    floor=config['observer']['position'][1]
    config['camera_continuity']['position_min']=[center[0]-size[0]/2+.35,floor+1.35,center[2]-size[2]/2+.35]
    config['camera_continuity']['position_max']=[center[0]+size[0]/2-.35,floor+1.8,center[2]+size[2]/2-.35]
    return config
