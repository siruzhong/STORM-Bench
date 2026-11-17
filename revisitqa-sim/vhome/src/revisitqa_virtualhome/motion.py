"""Freeze collision-aware character routes before native rendering."""
import json
import math
from pathlib import Path
import re

import numpy as np

from .scene import position


def planar_distance(first, second):
    return math.hypot(first[0] - second[0], first[2] - second[2])


def offscreen_view_direction(origin, target, destination):
    """Keep a hidden target behind the side view that best follows travel."""
    radial = (target[0] - origin[0], target[2] - origin[2])
    travel = (destination[0] - origin[0], destination[2] - origin[2])
    if math.hypot(*radial) <= 1e-8:
        return [travel[0], 0, travel[1]]
    angle = math.radians(120.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    candidates = [(cosine*radial[0]-sine*radial[1], sine*radial[0]+cosine*radial[1]),
                  (cosine*radial[0]+sine*radial[1], -sine*radial[0]+cosine*radial[1])]
    chosen = max(candidates, key=lambda direction: direction[0] * travel[0] + direction[1] * travel[1])
    return [chosen[0], 0, chosen[1]]


def route_turn_degrees(first, pivot, last):
    incoming = (pivot[0] - first[0], pivot[2] - first[2])
    outgoing = (last[0] - pivot[0], last[2] - pivot[2])
    lengths = math.hypot(*incoming) * math.hypot(*outgoing)
    if lengths <= 1e-8:
        return 0.0
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(incoming, outgoing)) / lengths))
    return math.degrees(math.acos(cosine))


def point_segment_distance(point, start, end):
    vector = (end[0] - start[0], end[2] - start[2])
    length_squared = vector[0] ** 2 + vector[1] ** 2
    if length_squared <= 1e-8:
        return planar_distance(point, start)
    offset = (point[0] - start[0], point[2] - start[2])
    ratio = max(0.0, min(1.0, (offset[0] * vector[0] + offset[1] * vector[1]) / length_squared))
    closest = (start[0] + ratio * vector[0], point[1], start[2] + ratio * vector[1])
    return planar_distance(point, closest)


def segment_segment_distance(first_start, first_end, second_start, second_end):
    """Return the planar distance between two closed route segments."""
    def cross(first, second, third):
        return ((second[0]-first[0])*(third[2]-first[2])-
                (second[2]-first[2])*(third[0]-first[0]))
    def contains(first, second, point):
        return (min(first[0],second[0])-1e-9<=point[0]<=max(first[0],second[0])+1e-9 and
                min(first[2],second[2])-1e-9<=point[2]<=max(first[2],second[2])+1e-9)
    values=(cross(first_start,first_end,second_start),
            cross(first_start,first_end,second_end),
            cross(second_start,second_end,first_start),
            cross(second_start,second_end,first_end))
    intersects=(values[0]*values[1]<0 and values[2]*values[3]<0)
    if not intersects:
        pairs=((values[0],first_start,first_end,second_start),
               (values[1],first_start,first_end,second_end),
               (values[2],second_start,second_end,first_start),
               (values[3],second_start,second_end,first_end))
        intersects=any(abs(value)<=1e-9 and contains(start,end,point)
                       for value,start,end,point in pairs)
    if intersects:
        return 0.0
    return min(point_segment_distance(first_start,second_start,second_end),
               point_segment_distance(first_end,second_start,second_end),
               point_segment_distance(second_start,first_start,first_end),
               point_segment_distance(second_end,first_start,first_end))


def patrol_cycle(waypoints, current_anchor_id=None):
    """Return a repeatable out-and-back cycle without duplicate turn points."""
    unique = []
    for waypoint in waypoints:
        if waypoint['anchor_id'] not in {item['anchor_id'] for item in unique}:
            unique.append(dict(waypoint))
    if len(unique) < 2:
        return unique
    cycle = unique + list(reversed(unique[1:-1]))
    if current_anchor_id is None:
        return cycle
    matches = [index for index, item in enumerate(cycle)
               if item['anchor_id'] == current_anchor_id]
    if not matches:
        return cycle
    start = matches[-1] + 1
    return cycle[start:] + cycle[:start]


def select_patrol_waypoint(route, offset):
    """Select a frozen destination and keep it first until the caller advances it."""
    if not route or not 0 <= offset < len(route):
        raise ValueError('Invalid patrol waypoint selection')
    for _ in range(offset):
        route.append(route.pop(0))
    return route[0]


def waypoint_is_navigable(node):
    """Reject dynamic object states that VirtualHome cannot approach safely."""
    return not ('CAN_OPEN' in node.get('properties', ()) and 'OPEN' in node.get('states', ()))


def waypoint_path_is_clear(nodes, waypoint, start, occupied, minimum_clearance,
                           minimum_step=.0):
    """Check a frozen waypoint against current character endpoints and paths."""
    goal = position(nodes[waypoint['anchor_id']])
    if planar_distance(start, goal) < minimum_step:
        return False
    return all(planar_distance(point, goal) >= minimum_clearance and
               point_segment_distance(point, start, goal) >= minimum_clearance
               for point in occupied)


def waypoint_escape_score(nodes, waypoint, start, occupied, hard_clearance,
                          preferred_clearance, minimum_step=.0, escape_slack=.05,
                          absolute_clearance=None):
    """Score a safe path, allowing an observer already inside the preferred margin to leave."""
    absolute_clearance=hard_clearance if absolute_clearance is None else absolute_clearance
    goal = position(nodes[waypoint['anchor_id']])
    if planar_distance(start, goal) < minimum_step:
        return None
    preferred_safe = True
    goal_clearances = []
    path_clearances = []
    for point in occupied:
        current = planar_distance(point, start)
        goal_clearance = planar_distance(point, goal)
        path_clearance = point_segment_distance(point, start, goal)
        if current < hard_clearance:
            preferred_safe = False
            if path_clearance < max(absolute_clearance,current-escape_slack):
                return None
            if goal_clearance < preferred_clearance:
                return None
        elif current >= preferred_clearance:
            if min(goal_clearance, path_clearance) < preferred_clearance:
                return None
        else:
            preferred_safe = False
            if path_clearance < max(hard_clearance, current - escape_slack):
                return None
            if goal_clearance < preferred_clearance:
                return None
        goal_clearances.append(goal_clearance)
        path_clearances.append(path_clearance)
    minimum_goal = min(goal_clearances, default=float('inf'))
    minimum_path = min(path_clearances, default=float('inf'))
    return preferred_safe, minimum_goal, minimum_path


def role_aware_waypoint_score(nodes, waypoint, start, occupied, active_operator,
                              hard_clearance, ordinary_clearance,
                              active_clearance=None, minimum_step=.0,
                              absolute_clearance=None):
    """Apply the larger manipulation margin only to the active operator."""
    score=waypoint_escape_score(
        nodes,waypoint,start,occupied,hard_clearance,ordinary_clearance,
        minimum_step,absolute_clearance=absolute_clearance)
    if score is None or active_clearance is None or active_operator is None:
        return score
    active_score=waypoint_escape_score(
        nodes,waypoint,start,[active_operator],hard_clearance,active_clearance,
        minimum_step,absolute_clearance=absolute_clearance)
    if active_score is None:
        return None
    return (score[0] and active_score[0],min(score[1],active_score[1]),
            min(score[2],active_score[2]))


def _ordered_waypoints(nodes, anchor_ids, start, previous, target, hidden, rules,
                       operator_positions=()):
    remaining = list(dict.fromkeys(anchor_ids))
    result = []
    current = start
    heading_start = previous
    while remaining and len(result) < rules['waypoints_per_phase']:
        safe = []
        for key in remaining:
            goal = position(nodes[key])
            travel = planar_distance(current, goal)
            clearance = (point_segment_distance(target, current, goal) if hidden
                         else planar_distance(target, goal))
            turn = 0.0 if heading_start is None else route_turn_degrees(heading_start, current, goal)
            if travel < rules['minimum_observer_step_m']:
                continue
            if clearance < rules['stationary_operator_clearance_m']:
                continue
            if any(planar_distance(point,goal)<rules['stationary_operator_clearance_m']
                   for point in operator_positions):
                continue
            if turn > rules['maximum_route_turn_degrees']:
                continue
            target_distance = planar_distance(goal, target)
            visibility_cost = -target_distance if hidden else target_distance
            score = visibility_cost + 0.20 * travel + 0.012 * turn
            safe.append((score, key, goal, turn, clearance))
        if not safe:
            break
        _, key, goal, turn, clearance = min(safe)
        result.append(dict(anchor_id=key, planned_turn_degrees=round(turn, 3),
                           target_clearance_m=round(clearance, 3)))
        remaining.remove(key)
        heading_start, current = current, goal
    return result


def build_motion_contract(graph, events, anchor_ids, observer_start, rules=None):
    """Build deterministic observer routes and operator reservations for an episode."""
    defaults = dict(agent_clearance_m=.85, stationary_operator_clearance_m=1.0,
                    runtime_hard_clearance_m=1.1,
                    dynamic_path_clearance_m=1.7,
                    operation_start_clearance_m=1.45,
                    operation_path_clearance_m=1.65,
                    minimum_observer_step_m=.75, short_relocation_minimum_step_m=.5,
                    maximum_route_turn_degrees=180.0,
                    waypoints_per_phase=3, concurrent_operator_motion=False,
                    observer_moves_during_manipulation=True,
                    maximum_concurrent_observer_step_m=1.5,
                    low_motion_anchor_budget_seconds=.5)
    defaults.update(rules or {})
    nodes = {node['id']: node for node in graph['nodes']}
    if any(key not in nodes for key in anchor_ids):
        raise ValueError('A motion anchor is absent from the scene graph')
    current = observer_start
    previous = None
    routes = []
    for event in events:
        target = position(nodes[event['object_id']])
        reservations=[target]
        if event.get('operator_position') is not None:
            reservations.append(event['operator_position'])
        approach_id=event.get('operator_approach_id')
        if approach_id in nodes:
            reservations.append(position(nodes[approach_id]))
        reserved_anchor_ids={event['object_id'],approach_id}
        event_anchor_ids=[key for key in anchor_ids if key not in reserved_anchor_ids]
        approach = _ordered_waypoints(nodes, event_anchor_ids, current, previous, target, False, defaults,
                                      reservations)
        if not approach:
            raise ValueError(f"No collision-safe observer approach route for object {event['object_id']}")
        approach_primary = position(nodes[approach[0]['anchor_id']])
        hidden = _ordered_waypoints(nodes, event_anchor_ids, approach_primary, current, target, True, defaults,
                                    reservations)
        if event['requested_visibility'] == 'offscreen' and not hidden:
            raise ValueError(f"No collision-safe hidden route for object {event['object_id']}")
        selected = hidden if event['requested_visibility'] == 'offscreen' else approach
        primary = selected[0]['anchor_id']
        previous, current = (approach_primary if event['requested_visibility'] == 'offscreen' else current), position(nodes[primary])
        routes.append(dict(event_index=len(routes), approach_waypoints=approach,
                           hidden_waypoints=hidden, action_waypoint=primary,
                           operator_approach_id=event.get('operator_approach_id',
                               event['surface_id'] if event['verb'] == 'PutBack' else event['object_id'])))
    return dict(schema_version=1, rules=defaults, event_routes=routes)


def assert_agent_clearance(graph, minimum):
    """Reject overlapping character endpoints after every native action."""
    characters = [node for node in graph['nodes'] if node['class_name'] == 'character']
    for index, first in enumerate(characters):
        for second in characters[index + 1:]:
            distance = planar_distance(position(first), position(second))
            if distance < minimum:
                raise ValueError(f"Characters {first['id']} and {second['id']} are {distance:.2f} m apart")


def movement_metrics(positions, fps, min_speed=.15):
    positions=np.asarray(positions,dtype=float)
    radius=max(1,round(fps*.25));index=np.arange(len(positions))
    left,right=np.maximum(0,index-radius),np.minimum(len(positions)-1,index+radius)
    elapsed=np.maximum((right-left)/fps,1/fps)
    speed=np.linalg.norm(positions[right][:,[0,2]]-positions[left][:,[0,2]],axis=1)/elapsed
    moving=speed>=min_speed;longest=run=0
    for value in moving:
        run=0 if value else run+1;longest=max(longest,run)
    return {'moving_frame_fraction':float(moving.mean()),'stationary_seconds':float((~moving).sum()/fps),
            'longest_stationary_seconds':longest/fps,'speed_threshold_mps':min_speed,
            'position_window_seconds':2*radius/fps},speed


def active_metrics(positions,headings,fps,min_speed=.15,min_turn_rate=30.0):
    report,speeds=movement_metrics(positions,fps,min_speed)
    angles=np.unwrap(np.asarray(headings,dtype=float));radius=max(1,round(fps*.25));index=np.arange(len(angles))
    left,right=np.maximum(0,index-radius),np.minimum(len(angles)-1,index+radius)
    elapsed=np.maximum((right-left)/fps,1/fps)
    turn_rates=np.abs(np.degrees(angles[right]-angles[left]))/elapsed
    active=(speeds>=min_speed)|(turn_rates>=min_turn_rate);longest=run=0
    for value in active:
        run=0 if value else run+1;longest=max(longest,run)
    report.update(active_frame_fraction=float(active.mean()),inactive_seconds=float((~active).sum()/fps),
                  longest_inactive_seconds=longest/fps,turn_threshold_degrees_per_second=min_turn_rate)
    return report,speeds,turn_rates


def audit_motion(root):
    """Measure continuous observer travel from skeleton poses paired with video frames."""
    root=Path(root);config=json.loads((root/'config.json').read_text());fps=config['render']['fps']
    files=sorted((root/'actions').glob('*/action_*/0/pd_*.txt'))
    if not files:raise ValueError('Observer motion audit requires skeleton recordings')
    poses,headings={},{};lines=files[-1].read_text().splitlines();joints=lines[0].split()
    left_hip,right_hip=joints.index('LeftUpperLeg'),joints.index('RightUpperLeg')
    for line in lines[1:]:
        values=line.split();frame=int(values[0]);points=np.asarray(list(map(float,values[1:])),dtype=float).reshape(-1,3)
        poses[frame]=points[0].tolist();axis=points[right_hip]-points[left_hip]
        if not np.isfinite(axis).all() or np.linalg.norm(axis[[0,2]])<.03:
            raise ValueError('Invalid hip orientation in skeleton recording')
        headings[frame]=float(np.arctan2(axis[2],axis[0]))
    first_pose_frame=min(poses)
    positions,orientations,trace=[],[],[]
    previous,heading=poses[first_pose_frame],headings[first_pose_frame]
    for line in (root/'frame_manifest.jsonl').open():
        row=json.loads(line);match=re.search(r'Action_(\d+)_0_normal\.png$',row['rgb'])
        frame_id=int(match[1]) if match else None
        if frame_id in poses:previous=poses[frame_id];heading=headings[frame_id]
        positions.append(previous);orientations.append(heading)
        trace.append({'frame':row['frame'],'time':row['frame']/fps,'hips_position':previous})
    report,speeds,turns=active_metrics(positions,orientations,fps)
    for row,speed,turn in zip(trace,speeds,turns):
        row['horizontal_speed_mps']=float(speed);row['turn_rate_degrees_per_second']=float(turn)
    (root/'motion_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trace))
    report.update(acceptance_mode=config.get('motion_mode','translation'),
                  initial_pose_frame=first_pose_frame)
    (root/'motion_validation.json').write_text(json.dumps(report,indent=2))
    active_mode=config.get('motion_mode')=='locomotion_and_turning'
    fraction=report['active_frame_fraction'] if active_mode else report['moving_frame_fraction']
    longest=report['longest_inactive_seconds'] if active_mode else report['longest_stationary_seconds']
    if fraction<.8 or longest>2.0:raise ValueError(f'Observer pauses exceed the patrol budget: {report}')
    return report
