#!/usr/bin/env python3
"""Render a frozen room routine with separate observers and operating actors."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import yaml
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.room_enrichment import reset_enriched_scene
from storm_virtualhome.planning import (check_injection_start, event_window_deadline,
                                        preferred_start, validate_program)
from storm_virtualhome.recording import (Recorder, recorded_target_visible,
                                         target_visibility_threshold, visible_pixels)
from storm_virtualhome.scene import action, position
from storm_virtualhome.native_navigation import NativePatrol
from storm_virtualhome.mixed import check_effect
from storm_virtualhome.motion import (assert_agent_clearance, patrol_cycle,
                                      offscreen_view_direction, planar_distance,
                                      point_segment_distance,
                                      role_aware_waypoint_score, select_patrol_waypoint,
                                      segment_segment_distance, waypoint_is_navigable)
from rollout_mixed import finalize
from storm_virtualhome.planning import operation_command_time


class ObserverNavigationRejected(RuntimeError):
    """Mark a failed observer-only route so the frozen alternatives can continue."""


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--executable',required=True)
    p.add_argument('--xorg-root',type=Path)
    p.add_argument('--gpu-index',type=int,default=3)
    p.add_argument('--defer-qa',action='store_true',
                   help='Keep per-video draft QA; defer batch curation')
    args=p.parse_args()
    config=yaml.safe_load(args.config.read_text());program=json.loads(args.plan.read_text())
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(args.executable).parent/'camera_guard.json',root/'runtime_manifest.json')
    with UnityProcess(args.executable,root/'logs',gpu_index=args.gpu_index,xorg_root=args.xorg_root,camera_rules=config['camera_continuity']) as c:
        reset_enriched_scene(c,config['scene'],config.get('scene_additions'));c.follow_camera(**config['camera'])
        names=c.command('character_cameras')['message']
        if isinstance(names,str):names=json.loads(names)
        camera=c.command('camera_count')['value']+names.index(config['camera']['name'])
        graph=c.graph()
        camera_keepouts = [dict(node['bounding_box'], object_id=node['id'])
                           for node in graph['nodes'] if node['class_name'] in ('ceilinglamp', 'tablelamp')]
        (root/'logs/camera_keepouts.json').write_text(json.dumps(dict(obstacles=camera_keepouts)))
        delayed_actors=set(config.get('delayed_operator_characters',()))
        actor_by_role={'observer':0,'injector':1,'helper':2}
        for role in config.get('character_roles',('observer','injector','helper')):
            if actor_by_role[role] in delayed_actors:
                continue
            try:c.character(**config[role])
            except Exception as exc:raise RuntimeError(f'Role initialization failed: {role}, {config[role]}') from exc
        config['camera']['recording_camera']=str(camera)
        graph=c.graph();nodes={n['id']:n for n in graph['nodes']}
        validate_program(program,graph,config)
        for name,data in [('config',config),('planning_graph',graph),('event_program',program),('plan',program['events'])]:
            (root/f'{name}.json').write_text(json.dumps(data,indent=2))
        def view(target=None,direction=None):
            data={'direction':direction} if direction is not None else {'position':nodes[target]['bounding_box']['center'],'region_object_id':target}
            (root/'logs/camera_view_target.json').write_text(json.dumps(data))
        first=program['events'][0];view(first['object_id'])
        (root/'logs/camera_start').touch()
        lines=[]
        active_actors=sorted({e['injector_character'] for e in program['events']})
        spawned_actors=set(active_actors)-delayed_actors
        for actor in active_actors:
            event_index,event=next((index,event) for index,event in enumerate(program['events'])
                                   if event['injector_character']==actor)
            if event_index>0:
                continue
            lines.append((action('Walk',nodes[event['operator_approach_id']],character=actor),
                          event.get('operator_find_solution',False)))
        setup=root/'setup';setup.mkdir()
        for index,(line,find_solution) in enumerate(lines):
            response=c.render([line],setup,f'setup_{index}',camera=str(camera),width=320,height=240,
                              find_solution=find_solution)
            if any(s.get('message')!='Success' for s in json.loads(response.get('message','{}')).values()):
                raise RuntimeError(response)
        clearance=config['motion_contract']['rules']['agent_clearance_m']
        navigation=None
        if config.get('native_navigation'):
            if not json.loads((root/'runtime_manifest.json').read_text()).get('native_navigation'):
                raise RuntimeError('The runtime does not support native patrol destinations')
            navigation=NativePatrol(c.graph(),config['room_id'],root/'logs',root/'native_navigation_plan.json',
                targets={event['object_id'] for event in program['events']},camera=config['camera'])
        setup_views=[]
        def future_spawn_positions():
            return [config[{1:'injector',2:'helper'}[actor]]['position']
                    for actor in active_actors if actor not in spawned_actors]
        first_target=first['object_id']
        def setup_target_visible():
            return visible_pixels(c.image(camera,mode='seg_inst'),c.colors()[str(first_target)])>=target_visibility_threshold(config,nodes[first_target])
        if not setup_target_visible():
            for waypoint in first['observer_approach_waypoints']:
                anchor=waypoint['anchor_id']
                if navigation is not None:
                    live=c.graph()['nodes']
                    origin=next(n for n in live if n['id']==1)['obj_transform']['position']
                    occupied=[n['obj_transform']['position'] for n in live if n['class_name']=='character' and n['id']!=1]+future_spawn_positions()
                    selected=navigation.choose(origin,occupied,position(nodes[first_target]),anchor,acquire=True,target_id=first_target)
                    if selected is None:continue
                    navigation.set_destination(origin,selected)
                response=c.render([action('Walk',nodes[anchor],character=0)],setup,
                                  f'observer_{anchor}',camera=str(camera),width=320,height=240,
                                  find_solution=False)
                succeeded=all(s.get('message')=='Success' for s in json.loads(response.get('message','{}')).values())
                seen=setup_target_visible() if succeeded else False
                setup_views.append(dict(anchor_id=anchor,native_success=succeeded,target_visible=seen))
                if seen:break
            if not setup_target_visible():
                (root/'setup_view_validation.json').write_text(json.dumps(setup_views,indent=2))
                raise RuntimeError('No frozen approach provides a verified initial target view')
        (root/'setup_view_validation.json').write_text(json.dumps(setup_views,indent=2))
        assert_agent_clearance(c.graph(),clearance)
        setup_frames=sorted(setup.rglob('*_normal.png'))
        if not setup_frames:
            # A pre-positioned operator can finish setup without animation.
            # Sample the live camera so renderer validation still checks pixels.
            live=setup/'setup_live_normal.png'
            Image.fromarray(c.image(camera,width=320,height=240)).save(live)
            setup_frames=[live]
        for frame in setup_frames:
            with Image.open(frame) as pixels:
                if not np.any(np.asarray(pixels)[...,:3]):
                    raise RuntimeError(f'Capture initialization failed: black setup frame {frame}')
        (root/'initial_graph.json').write_text(json.dumps(c.graph()))
        recorder=Recorder(c,root,camera,config)
        events=[];pending=[];stages={};last_start=None;motion=[];operator_reservations=[]
        observer_anchor=None;visible_anchor_by_target={}
        workspace_ids_by_actor={actor:set() for actor in active_actors}
        for event in program['events']:
            workspace_ids_by_actor[event['injector_character']].update(
                (event['object_id'],event['operator_approach_id']))
        minimum=config['evidence']['min_visible_pixels']
        def target_minimum(target):
            return target_visibility_threshold(config,nodes[target])
        def visible(target,threshold=None):
            threshold=target_minimum(target) if threshold is None else threshold
            colors=c.colors();mask=c.image(camera,mode='seg_inst')
            return visible_pixels(mask,colors[str(target)])>=threshold
        def snapshot(key,target,use_last_frame=False):
            result=recorder.snapshot(key,target,use_last_frame=use_last_frame)
            if result['color_collision_ids'] or result['visible_pixels']<target_minimum(target):raise RuntimeError(f'Unverified view: {key}')
            stages[key]={k:v for k,v in result.items() if k!='graph'}
            (root/'stages.json').write_text(json.dumps(stages,indent=2))
            return result
        def actor_position(actor):
            return next(n for n in c.graph()['nodes'] if n['id']==actor+1)['obj_transform']['position']
        def spawn_delayed_operator(actor):
            if actor in spawned_actors:
                return
            role={1:'injector',2:'helper'}[actor]
            c.character(**config[role])
            character_ids=sorted(node['id'] for node in c.graph()['nodes']
                                 if node['class_name']=='character')
            if actor+1 not in character_ids:
                raise RuntimeError(f'Delayed operator {actor} received an unexpected node ID')
            spawned_actors.add(actor)
            verify_clearance(f'delayed_operator_{actor}_spawn')
        def save_motion():
            (root/'motion_ledger.json').write_text(json.dumps(motion,indent=2))
        def verify_clearance(context):
            graph=c.graph();assert_agent_clearance(graph,clearance)
            characters=[node for node in graph['nodes'] if node['class_name']=='character']
            row=dict(context=context,time=recorder.time,
                     positions={str(node['id']-1):node['obj_transform']['position']
                                for node in characters},
                     minimum_clearance_m=clearance)
            motion.append(row);save_motion()
        def waypoint_score(waypoint,observer_position,occupied,preferred_clearance=None,
                           active_operator=None,operator_motion=None,
                           ordinary_clearance=None,minimum_step=None):
            rules=config['motion_contract']['rules']
            requested_ordinary_clearance=ordinary_clearance
            ordinary_clearance=(ordinary_clearance or
                                rules.get('dynamic_path_clearance_m',clearance))
            score=role_aware_waypoint_score(
                nodes,waypoint,observer_position,occupied,active_operator,
                rules.get('runtime_hard_clearance_m',clearance),
                ordinary_clearance,
                preferred_clearance,(rules['minimum_observer_step_m']
                                     if minimum_step is None else minimum_step),
                absolute_clearance=clearance)
            if score is None:
                return None
            if operator_motion is not None:
                goal=position(nodes[waypoint['anchor_id']])
                if segment_segment_distance(observer_position,goal,*operator_motion)<rules.get(
                        'runtime_hard_clearance_m',clearance):
                    return None
            goal=position(nodes[waypoint['anchor_id']])
            reservation_clearance=(preferred_clearance or requested_ordinary_clearance or
                                   rules['stationary_operator_clearance_m'])
            reservations_clear=all(
                planar_distance(point,goal)>=reservation_clearance
                for point in operator_reservations)
            return score if reservations_clear else None
        def take_patrol(route,context,excluded_anchor=None,preferred_clearance=None,
                        active_actor=None,moving_operator_goal=None,
                        ordinary_clearance=None,future_operator_goal=None,required=True):
            live={node['id']:node for node in c.graph()['nodes']}
            observer_position=live[1]['obj_transform']['position']
            occupied=[node['obj_transform']['position'] for key,node in live.items()
                      if node['class_name']=='character' and key!=1]+future_spawn_positions()
            active_node=live.get(active_actor+1) if active_actor is not None else None
            active_operator=(active_node['obj_transform']['position']
                             if active_node is not None else None)
            if navigation is not None:
                concurrent_seconds = e.get('measured_operation_seconds') if context.endswith('_operation') else None
                if active_operator is not None and moving_operator_goal is not None:
                    concurrent_seconds = planar_distance(active_operator, moving_operator_goal) / 1.1 + .75
                valid=[waypoint for waypoint in route if waypoint_is_navigable(live[waypoint['anchor_id']])]
                valid.sort(key=lambda waypoint:planar_distance(
                    observer_position,position(live[waypoint['anchor_id']])),reverse=True)
                selected=(navigation.choose(observer_position,occupied,
                    position(recorder.target),valid[0]['anchor_id'],
                    clearance=config['motion_contract']['rules'].get('runtime_hard_clearance_m',1.1),
                    operation_seconds=concurrent_seconds,
                    timing_seconds=(max(.1, ready_time-recorder.time) if 'timing' in context else None),
                    reserved_segments=(navigation.operator_segments(active_operator,moving_operator_goal)
                        if active_operator is not None and moving_operator_goal is not None else []),
                    future_goal=future_operator_goal,target_id=recorder.target['id'],
                    acquire=(any(word in context for word in ('approach','recovery','discovery'))
                             or ('timing' in context and e['requested_visibility']=='in_view')),
                    hidden=('concealment' in context or (e['requested_visibility']=='offscreen'
                        and any(word in context for word in ('timing','operation')))),
                    hidden_direction=hidden_heading if context.endswith('_operation') else None,
                    hidden_points=[position(recorder.target),active_operator] if active_operator is not None else [])
                    if valid else None)
                if selected is not None:return selected
                if not required:return None
                raise RuntimeError(f'No verified native path remains for {context}')
            operator_motion=((active_operator,moving_operator_goal)
                             if active_operator is not None and moving_operator_goal is not None
                             else None)
            inactive_workspace_ids=(set().union(*(
                ids for actor,ids in workspace_ids_by_actor.items()
                if actor!=active_actor and actor in spawned_actors)))
            def choose(minimum_step=None):
                preferred=[];fallback=[]
                for offset,waypoint in enumerate(route):
                    if waypoint['anchor_id']==excluded_anchor:
                        continue
                    if waypoint['anchor_id'] in inactive_workspace_ids:
                        continue
                    if not waypoint_is_navigable(live[waypoint['anchor_id']]):
                        continue
                    score=waypoint_score(
                        waypoint,observer_position,occupied,preferred_clearance,
                        active_operator,operator_motion,ordinary_clearance,minimum_step)
                    if score is None:
                        continue
                    future_clearance=float('inf')
                    if future_operator_goal is not None and active_operator is not None:
                        goal=position(nodes[waypoint['anchor_id']])
                        future_clearance=point_segment_distance(
                            goal,active_operator,future_operator_goal)
                        if future_clearance<config['motion_contract']['rules'].get(
                                'runtime_hard_clearance_m',clearance):
                            continue
                    candidate=(future_clearance,-offset,offset)
                    (preferred if score[0] else fallback).append(candidate)
                    if score[0] and future_operator_goal is None:
                        return offset
                # Every fallback satisfies the hard clearance rule and moves
                # away from an operator. Keep the frozen route order so the
                # observer does not repeatedly select one high-clearance anchor.
                choices=preferred or fallback
                return max(choices)[2] if future_operator_goal is not None and choices else (
                    choices[0][2] if choices else None)
            selected=choose()
            if selected is None:
                selected=choose(config['motion_contract']['rules'].get(
                    'short_relocation_minimum_step_m',.5))
            if selected is not None:
                # Native walking may need several calls to reach one object.
                # Keep that destination first until the measured motion stalls;
                # switching after every partial walk creates sharp reversals.
                return select_patrol_waypoint(route,selected)
            if not required:
                return None
            raise RuntimeError(f'No collision-safe patrol waypoint remains for {context}')
        def walk_waypoint(target,waypoint,hidden=False,extra=None,find_solution=False,
                          hide_operator=False,locomotion='WalkTowards'):
            nonlocal hidden_heading
            origin=actor_position(0)
            if navigation is not None:
                navigation.set_destination(origin,waypoint)
                locomotion='Walk'
            if hidden:
                goal=position(nodes[target])
                destination=waypoint.get('navigation_position',position(nodes[waypoint['anchor_id']]))
                if hidden_heading is None:
                    hidden_heading=offscreen_view_direction(origin,goal,destination)
                view(direction=hidden_heading)
            line=action(locomotion,nodes[waypoint['anchor_id']],character=0)
            if extra:line+=' | '+extra
            previous=recorder.time
            try:
                result=recorder.run(line,find_solution=find_solution)
            except RuntimeError as exc:
                if extra is None and 'Action execution failed:' in str(exc):
                    # Failed native actions are not added to the recorder ledger.
                    # Remove their scratch directory before the next frozen
                    # waypoint reuses the same action index.
                    shutil.rmtree(root/'actions'/f'{len(recorder.actions):04d}',
                                  ignore_errors=True)
                    raise ObserverNavigationRejected(str(exc)) from exc
                raise
            reached=actor_position(0)
            result['observer_step_m']=planar_distance(origin,reached)
            if navigation is not None:
                error=planar_distance(reached,waypoint['navigation_position'])
                with (root/'navigation_execution.jsonl').open('a') as trace:
                    trace.write(json.dumps(dict(start=result['start'],end=result['end'],
                        origin=origin,requested=waypoint['navigation_position'],reached=reached,
                        error_m=error,action=line))+'\n')
                if error>.40:raise RuntimeError(f'Native destination error: {error:.3f} m')
            context=('observer_near_' if recorder.time-previous<.1 else 'observer_to_')+str(waypoint['anchor_id'])
            verify_clearance(context)
            return result
        def turn_to_target(target):
            result=recorder.run(action('TurnTo',nodes[target],character=0))
            result['observer_step_m']=0.0
            verify_clearance(f'observer_turn_to_{target}')
            return result
        def turn_during_operation(target,waypoint,hidden,operation,hide_operator,actor):
            if hidden:
                origin=actor_position(0)
                goal=position(nodes[target])
                destination=position(nodes[waypoint['anchor_id']])
                direction=(offscreen_view_direction(origin,goal,destination) if hide_operator
                           else [origin[0]-goal[0],0,origin[2]-goal[2]])
                view(direction=direction)
            else:
                view(target)
            turn_verb='TurnRight' if waypoint['anchor_id']%2 else 'TurnLeft'
            line=action(turn_verb,character=0)
            result=recorder.run(line+' | '+operation,find_solution=False)
            result['observer_step_m']=0.0
            verify_clearance(f'observer_turn_during_operation_{target}')
            return result
        def update_stall(previous_anchor,elapsed,waypoint,result):
            if result['observer_step_m']>=.10:
                return None,0.0,None
            current=waypoint['anchor_id']
            duration=result['end']-result['start']
            elapsed=elapsed+duration if previous_anchor==current else duration
            budget=config['motion_contract']['rules'].get('low_motion_anchor_budget_seconds',.5)
            return current,elapsed,current if elapsed>=budget else None
        def operator_move_action(event):
            actor=event['injector_character'];target=event['operator_approach_id']
            if actor not in spawned_actors:return None
            previous=next((item for item in reversed(events) if item['injector_character']==actor),None)
            if event['verb']=='PutBack' or (previous and previous['object_id']==event['object_id']):
                return None
            destination=event.get('operator_position') or position(nodes[target])
            if planar_distance(actor_position(actor),destination)<=.5:return None
            return action('Walk',nodes[target],character=actor)
        def discover(observation=None):
            recorded=(observation or {}).get('last_visible_pixels',{})
            for event in pending[:]:
                key=str(event['object_id'])
                threshold=target_minimum(event['object_id'])
                seen=recorded[key]>=threshold if key in recorded else visible(event['object_id'])
                if seen:
                    recorder.set_target(nodes[event['object_id']])
                    after=snapshot(event['stage_keys']['after'],event['object_id'],key in recorded)
                    before=json.loads((root/'evidence'/f"{event['stage_keys']['before']}.json").read_text())
                    check_effect(before['graph'],after['graph'],event)
                    event['discovery_time']=after['time'];pending.remove(event)
            (root/'events.json').write_text(json.dumps(events,indent=2))
        try:
            for index,planned in enumerate(program['events']):
                e=dict(planned,observer_character=0)
                target=e['object_id'];actor=e['injector_character']
                operator_reservations[:]=[position(nodes[target]),
                                           position(nodes[e['operator_approach_id']])]
                if e.get('operator_position') is not None:
                    operator_reservations.append(e['operator_position'])
                desired=preferred_start(e,last_start,e.get('planned_gap_seconds',config['event_interval_seconds']),
                                        config.get('timing_policy','planned_start_windows_without_retiming'))
                deadline=event_window_deadline(e,desired,config.get(
                    'timing_policy','planned_start_windows_without_retiming'))
                view(target);recorder.set_target(nodes[target])
                if navigation is not None and not visible(target):
                    navigation.refresh_sight(nodes[target])
                approach_plan=list(e['observer_approach_waypoints'])
                hidden_plan=list(e['observer_hidden_waypoints'])
                known_visible=visible_anchor_by_target.get(target)
                if known_visible is not None:
                    approach_plan.sort(key=lambda waypoint:waypoint['anchor_id']!=known_visible)
                last_anchor=observer_anchor;low_motion_anchor=None;low_motion_seconds=0.0;stalled_anchor=None
                # The frozen approach order already starts with the best view of
                # a new target. Rotating it from the prior event can force a
                # room-scale detour before the target is first acquired.
                approach=patrol_cycle(approach_plan)
                pending_operator_move=operator_move_action(e)
                approach_turn_used=False
                approach_observation=None
                def approach_visible():
                    recorded=recorded_target_visible(approach_observation,target,target_minimum(target))
                    return visible(target) if recorded is None else recorded
                recovering_hidden=any(item['object_id']==target for item in pending)
                if recovering_hidden and pending_operator_move is None and not approach_visible():
                    try:
                        approach_observation=turn_to_target(target)
                    except RuntimeError as exc:
                        if 'Action execution failed:' not in str(exc):
                            raise
                        shutil.rmtree(root/'actions'/f'{len(recorder.actions):04d}',
                                      ignore_errors=True)
                    approach_turn_used=True
                while pending_operator_move is not None or not approach_visible():
                    if recorder.time-deadline>1e-6:
                        raise RuntimeError(f'Approach exceeded the frozen window for event {index}')
                    if (pending_operator_move is None and stalled_anchor is not None
                            and not approach_turn_used
                            and (known_visible is None
                                 or target_minimum(target)<minimum)):
                        try:
                            approach_observation=turn_to_target(target)
                        except RuntimeError as exc:
                            if 'Action execution failed:' not in str(exc):
                                raise
                            shutil.rmtree(root/'actions'/f'{len(recorder.actions):04d}',
                                          ignore_errors=True)
                        approach_turn_used=True
                        continue
                    operation_clearance=config['motion_contract']['rules'].get(
                        'operation_path_clearance_m',clearance)
                    start_clearance=config['motion_contract']['rules'].get(
                        'operation_start_clearance_m',clearance)
                    full_acquisition=(pending_operator_move is not None or
                                      known_visible is None)
                    complete_walk=full_acquisition or recovering_hidden
                    move_clearance=(operation_clearance if full_acquisition
                                    else start_clearance)
                    move_goal=(e.get('operator_position') or
                               position(nodes[e['operator_approach_id']]))
                    waypoint=take_patrol(approach,f'event_{index}_approach',stalled_anchor,
                                         move_clearance,actor,
                                         move_goal if pending_operator_move is not None else None,
                                         ordinary_clearance=(operation_clearance if full_acquisition
                                             else config['motion_contract']['rules'].get(
                                                 'dynamic_path_clearance_m',clearance)),
                                         required=False)
                    if waypoint is None and pending_operator_move is None:
                        hard_clearance=config['motion_contract']['rules'].get(
                            'runtime_hard_clearance_m',clearance)
                        waypoint=take_patrol(
                            approach,f'event_{index}_recovery_tight',stalled_anchor,
                            hard_clearance,actor,ordinary_clearance=hard_clearance,
                            required=False)
                    move_operator=pending_operator_move is not None and waypoint is not None
                    if waypoint is None:
                        if pending_operator_move is not None:
                            # Move the observer to one side of a crossing route,
                            # then retry the concurrent operator walk from there.
                            staging_route=patrol_cycle([
                                {'anchor_id':key} for key in config['room_patrol_anchors']],
                                last_anchor)
                            waypoint=take_patrol(
                                staging_route,f'event_{index}_staging',stalled_anchor,
                                active_actor=actor,future_operator_goal=move_goal,
                                required=False)
                            if waypoint is None:
                                hard_clearance=config['motion_contract']['rules'].get(
                                    'runtime_hard_clearance_m',clearance)
                                waypoint=take_patrol(
                                    staging_route,f'event_{index}_staging_tight',stalled_anchor,
                                    hard_clearance,actor,ordinary_clearance=hard_clearance,
                                    future_operator_goal=move_goal,
                                    required=False)
                            if waypoint is None:
                                raise RuntimeError(
                                    f'No collision-safe patrol waypoint remains for event_{index}_staging')
                        elif (known_visible is not None and last_anchor==known_visible
                              and not approach_turn_used):
                            approach_observation=turn_to_target(target)
                            approach_turn_used=True
                            continue
                        else:
                            raise RuntimeError(
                                f'No collision-safe patrol waypoint remains for event_{index}_approach')
                    try:
                        result=walk_waypoint(
                            target,waypoint,
                            extra=pending_operator_move if move_operator else None,
                            locomotion=('Walk' if complete_walk else 'WalkTowards'))
                    except ObserverNavigationRejected:
                        stalled_anchor=waypoint['anchor_id']
                        continue
                    approach_observation=result
                    if move_operator:
                        pending_operator_move=None
                        if navigation is not None and not approach_visible():
                            navigation.refresh_sight(nodes[target])
                    last_anchor=waypoint['anchor_id'];observer_anchor=last_anchor
                    low_motion_anchor,low_motion_seconds,stalled_anchor=update_stall(
                        low_motion_anchor,low_motion_seconds,waypoint,result)
                    should_turn=(recovering_hidden or known_visible is None
                                 or target_minimum(target)<minimum)
                    if (pending_operator_move is None and not approach_visible()
                            and not approach_turn_used and should_turn):
                        try:
                            approach_observation=turn_to_target(target)
                        except RuntimeError as exc:
                            if 'Action execution failed:' not in str(exc):
                                raise
                            shutil.rmtree(root/'actions'/f'{len(recorder.actions):04d}',
                                          ignore_errors=True)
                        approach_turn_used=True
                discover(approach_observation);recorder.set_target(nodes[target])
                before_key=f'event_{index:02d}_before';after_key=f'event_{index:02d}_after'
                before=snapshot(before_key,target,
                    recorded_target_visible(approach_observation,target,target_minimum(target)) is True)
                if navigation is not None:
                    navigation.note_visible(target,actor_position(0))
                if last_anchor is not None:visible_anchor_by_target[target]=last_anchor
                hidden=e['requested_visibility']=='offscreen'
                last_view_result=({'last_visible_pixels':{str(target):before['visible_pixels']}}
                                  if not hidden else None)
                hidden_heading=None
                if hidden:
                    hidden_route=patrol_cycle(hidden_plan,last_anchor)
                    concealment_turn_used=False
                    operation_clearance=config['motion_contract']['rules'].get(
                        'operation_path_clearance_m',clearance)
                    while True:
                        if recorder.time-deadline>1e-6:
                            raise RuntimeError(f'Concealment exceeded the frozen window for event {index}')
                        waypoint=take_patrol(
                            hidden_route,f'event_{index}_concealment',stalled_anchor,
                            operation_clearance,actor,
                            ordinary_clearance=operation_clearance,required=False)
                        if waypoint is None:
                            hard_clearance=config['motion_contract']['rules'].get(
                                'runtime_hard_clearance_m',clearance)
                            waypoint=take_patrol(
                                hidden_route,f'event_{index}_concealment_tight',stalled_anchor,
                                hard_clearance,actor,ordinary_clearance=hard_clearance,
                                required=False)
                        if waypoint is None:
                            raise RuntimeError(
                                f'No collision-safe patrol waypoint remains for event_{index}_concealment')
                        try:
                            last_view_result=walk_waypoint(
                                target,waypoint,True,locomotion='WalkTowards')
                        except ObserverNavigationRejected:
                            stalled_anchor=waypoint['anchor_id']
                            continue
                        last_anchor=waypoint['anchor_id'];observer_anchor=last_anchor
                        low_motion_anchor,low_motion_seconds,stalled_anchor=update_stall(
                            low_motion_anchor,low_motion_seconds,waypoint,last_view_result)
                        pixels=last_view_result.get('last_visible_pixels',{})
                        if pixels.get(str(target),minimum)==0:break
                        if not concealment_turn_used:
                            try:
                                # The camera keeps the frozen offscreen heading
                                # while the native turn supplies enough frames
                                # for the smoothed view to settle.
                                last_view_result=turn_to_target(target)
                            except RuntimeError as exc:
                                if 'Action execution failed:' not in str(exc):
                                    raise
                                shutil.rmtree(root/'actions'/f'{len(recorder.actions):04d}',
                                              ignore_errors=True)
                            concealment_turn_used=True
                            pixels=last_view_result.get('last_visible_pixels',{}) if last_view_result else {}
                            if pixels.get(str(target),minimum)==0:break
                delayed_operator_first_event=actor not in spawned_actors
                if delayed_operator_first_event:
                    spawn_delayed_operator(actor)
                    # The first delayed event is forced offscreen. A recorded
                    # post-spawn patrol frame must verify both the target and
                    # operator before the event can start.
                    last_view_result=None
                base_route=hidden_plan if hidden else approach_plan
                patrol=patrol_cycle(base_route,last_anchor)
                hidden_operator_mode=False
                # Full-animation probes include native approach latency. Start
                # the command early enough for the manipulation itself to land
                # in its immutable event window.
                ready_time, operation_lead = operation_command_time(e, desired, last_start, config)
                timing_turn_used=False
                turn_only=(config.get('observer_turns_during_manipulation',False)
                           or (delayed_operator_first_event and config.get(
                               'delayed_operator_first_event_turn_only',False)))
                operation_clearance=config['motion_contract']['rules'].get('operation_path_clearance_m')
                require_operation_walk=(navigation is not None and not turn_only and
                    config['motion_contract']['rules'].get('observer_moves_during_manipulation',True))
                operation_waypoint=None
                def view_ready():
                    observer=actor_position(0)
                    separation=config['motion_contract']['rules'].get(
                        'operation_start_clearance_m',
                        config['motion_contract']['rules']['stationary_operator_clearance_m'])
                    if navigation is not None:
                        separation=config['motion_contract']['rules'].get('runtime_hard_clearance_m',1.1)
                    if (planar_distance(observer,actor_position(actor))<separation
                            or (navigation is None and any(planar_distance(observer,point)<separation
                                   for point in operator_reservations))):
                        return False
                    if not hidden:
                        recorded=recorded_target_visible(last_view_result,target,target_minimum(target))
                        return visible(target) if recorded is None else recorded
                    if last_view_result is None:return False
                    pixels=last_view_result.get('last_visible_pixels',{})
                    return not any(pixels.get(str(key),minimum) for key in (target,actor+1))
                def manipulation_ready():
                    nonlocal operation_waypoint
                    if not view_ready():return False
                    if not require_operation_walk:return True
                    # A short turn cannot keep the observer active throughout
                    # a long manipulation. Stage a safe outgoing walk first.
                    operation_waypoint=take_patrol(
                        patrol,f'event_{index}_operation',last_anchor,
                        operation_clearance,actor,required=False)
                    return operation_waypoint is not None
                while recorder.time<ready_time or not manipulation_ready():
                    if (not hidden and not timing_turn_used
                            and 0<ready_time-recorder.time<=1.0 and view_ready()):
                        # Keep a verified view during a short timing gap. A new
                        # walk can put the operator between the camera and prop.
                        timing_turn_used=True
                        try:
                            last_view_result=turn_to_target(target)
                        except RuntimeError as exc:
                            if 'Action execution failed:' not in str(exc):raise
                            # Some native props have no valid watch point.
                            # An unrecorded optional turn can fall back to walking.
                            shutil.rmtree(root/'actions'/f'{len(recorder.actions):04d}',
                                          ignore_errors=True)
                        else:
                            if recorder.time-deadline>1e-6:
                                raise RuntimeError(f'Patrol exceeded the frozen window for event {index}')
                            continue
                    start_clearance=config['motion_contract']['rules'].get(
                        'operation_start_clearance_m',
                        config['motion_contract']['rules']['stationary_operator_clearance_m'])
                    pixels=(last_view_result or {}).get('last_visible_pixels',{})
                    hidden_operator_mode=(hidden and pixels.get(str(target),minimum)==0
                                          and pixels.get(str(actor+1),0)>0)
                    waypoint=take_patrol(patrol,f'event_{index}_timing',stalled_anchor,
                                         start_clearance,actor,required=False)
                    if waypoint is None and not hidden:
                        hard_clearance=config['motion_contract']['rules'].get(
                            'runtime_hard_clearance_m',clearance)
                        waypoint=take_patrol(
                            patrol,f'event_{index}_timing_tight',stalled_anchor,
                            hard_clearance,actor,ordinary_clearance=hard_clearance,
                            required=False)
                    if waypoint is None:
                        if not timing_turn_used:
                            last_view_result=turn_to_target(target)
                            timing_turn_used=True
                            if recorder.time-deadline>1e-6:
                                raise RuntimeError(
                                    f'Patrol exceeded the frozen window for event {index}')
                            continue
                        raise RuntimeError(
                            f'No collision-safe patrol waypoint remains for event_{index}_timing')
                    try:
                        last_view_result=walk_waypoint(
                            target,waypoint,hidden,hide_operator=hidden_operator_mode)
                    except ObserverNavigationRejected:
                        stalled_anchor=waypoint['anchor_id']
                        continue
                    last_anchor=waypoint['anchor_id'];observer_anchor=last_anchor
                    low_motion_anchor,low_motion_seconds,stalled_anchor=update_stall(
                        low_motion_anchor,low_motion_seconds,waypoint,last_view_result)
                    if recorder.time-deadline>1e-6:
                        raise RuntimeError(f'Patrol exceeded the frozen window for event {index}')
                operation=action(e['verb'],nodes[target],*([nodes[e['surface_id']]] if e['verb']=='PutBack' else []),character=actor)
                if config['motion_contract']['rules'].get('observer_moves_during_manipulation',True):
                    # Failed joint actions invalidate the attempt because the
                    # operator may already have changed the object state.
                    waypoint=(operation_waypoint if require_operation_walk else
                        None if turn_only else take_patrol(
                            patrol,f'event_{index}_operation',last_anchor,
                            operation_clearance,actor,required=False))
                    if waypoint is None:
                        turn_waypoint=next((item for item in patrol
                                            if item['anchor_id']!=last_anchor),next(iter(patrol)))
                        result=turn_during_operation(
                            target,turn_waypoint,hidden,operation,hidden_operator_mode,actor)
                    else:
                        result=walk_waypoint(target,waypoint,hidden,operation,False,
                                             hidden_operator_mode,locomotion='Walk')
                        last_anchor=waypoint['anchor_id'];observer_anchor=last_anchor
                else:
                    result=recorder.run(operation,find_solution=e.get('operator_find_solution',False))
                    verify_clearance(f"operation_{index}_{e['verb']}")
                operation_graph=c.graph()
                (root/'evidence'/f'event_{index:02d}_execution.json').write_text(json.dumps(dict(
                    graph=operation_graph,start=result['start'],end=result['end'],
                    scope='private_operation_verification'),indent=2))
                check_effect(before['graph'],operation_graph,e)
                updated_target=next(node for node in operation_graph['nodes'] if node['id']==target)
                if navigation is not None and planar_distance(position(nodes[target]),position(updated_target))>.2:
                    navigation.sight.pop(target,None)
                    navigation.observed_stations.pop(target,None)
                    visible_anchor_by_target.pop(target,None)
                nodes[target]=updated_target
                spans=[span for span in result.get('native_action_spans',{}).get(str(actor),[])
                       if span['verb']==e['verb'].upper()]
                if not spans:
                    raise RuntimeError(f'Missing native operation timing for event {index}')
                actual=spans[-1]
                check_injection_start(e,actual['start'],last_start,config)
                last_start=actual['start']
                e.update(start=actual['start'],end=actual['end'],joint_action_start=result['start'],
                         joint_action_end=result['end'],before_time=before['time'],
                         stage_keys={'before':before_key,'after':after_key})
                events.append(e);pending.append(e)
                if not hidden:discover(result)
                (root/'events.json').write_text(json.dumps(events,indent=2))
                print(json.dumps(dict(event=index,time=recorder.time,pending=len(pending))),flush=True)
            discovery_task=None;discovery_route=None;discovery_stalled=None
            discovery_low_anchor=None;discovery_low_seconds=0.0
            while pending:
                event=pending[0];view(event['object_id'])
                recorder.set_target(nodes[event['object_id']])
                if event['task']!=discovery_task:
                    discovery_task=event['task']
                    discovery_route=patrol_cycle(event['observer_approach_waypoints'])
                    discovery_stalled=None;discovery_low_anchor=None;discovery_low_seconds=0.0
                waypoint=take_patrol(
                    discovery_route,'final_discovery',discovery_stalled,
                    active_actor=event['injector_character'])
                try:
                    result=walk_waypoint(event['object_id'],waypoint)
                except ObserverNavigationRejected:
                    discovery_stalled=waypoint['anchor_id']
                    continue
                discover(result)
                discovery_low_anchor,discovery_low_seconds,discovery_stalled=update_stall(
                    discovery_low_anchor,discovery_low_seconds,waypoint,result)
                if recorder.time>config['duration_range_seconds'][1]:raise RuntimeError('Final inspection exceeded duration limit')
            tail_plan=list(events[-1]['observer_approach_waypoints'])+list(events[-1]['observer_hidden_waypoints'])
            route=patrol_cycle(tail_plan);tail_stalled=None;tail_low_anchor=None;tail_low_seconds=0.0
            while recorder.time<config['target_duration_seconds']:
                waypoint=take_patrol(
                    route,'duration_fill',tail_stalled,
                    active_actor=events[-1]['injector_character'])
                try:
                    result=walk_waypoint(events[-1]['object_id'],waypoint)
                except ObserverNavigationRejected:
                    tail_stalled=waypoint['anchor_id']
                    continue
                tail_low_anchor,tail_low_seconds,tail_stalled=update_stall(
                    tail_low_anchor,tail_low_seconds,waypoint,result)
        finally:recorder.close()
    finalize(root,defer_qa=args.defer_qa)


if __name__=='__main__':main()
