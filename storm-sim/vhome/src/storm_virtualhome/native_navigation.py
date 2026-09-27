"""Compile patrol destinations from Unity NavMesh paths before capture."""
import hashlib
import heapq
import json
import math
from pathlib import Path
import time

from .motion import planar_distance, point_segment_distance, route_turn_degrees, segment_segment_distance
from .room_preflight import spawn_points


def request_paths(log_dir, points, edges, timeout=30, sight_targets=None, camera=None, fresh=False):
    log_dir=Path(log_dir)
    request=dict(points=points,edges=edges,sample_radius=.35,sight_targets=sight_targets)
    if camera:
        side,height,depth=camera['position']
        request.update(camera_arm=-depth,camera_height=height,camera_side=side)
    if fresh:request['world_revision']=time.monotonic_ns()
    request['id']=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    temporary=log_dir/'navigation_request.tmp'
    temporary.write_text(json.dumps(request))
    temporary.replace(log_dir/'navigation_request.json')
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        path=log_dir/'navigation_response.json'
        if path.exists():
            result=json.loads(path.read_text())
            if result['id']==request['id']:return result
        time.sleep(.05)
    raise TimeoutError('Unity did not return the native navigation graph')


def camera_route_penalties(log_dir, routes, timeout=10):
    """Probe whether an observer route would pass behind nearby camera occluders."""
    log_dir=Path(log_dir)
    request=dict(id=str(time.monotonic_ns()), routes=routes)
    temporary=log_dir/'camera_routes_request.tmp'
    temporary.write_text(json.dumps(request))
    temporary.replace(log_dir/'camera_routes_request.json')
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        response=log_dir/'camera_routes_response.json'
        if response.exists():
            result=json.loads(response.read_text())
            if result['id']==request['id']:
                if len(result['blocked'])!=len(routes):raise ValueError('Incomplete camera route probe')
                return result['blocked']
        time.sleep(.02)
    raise TimeoutError('Unity did not return the camera route probe')


def path_clearance(corners, occupied):
    return min((point_segment_distance(person,start,end) for person in occupied
                for start,end in zip(corners,corners[1:])),default=float('inf'))


def observation_distance(target):
    """Prefer a closer view for props that occupy few image pixels."""
    if target['class_name']=='lightswitch':return 1.4
    size=sorted(target['bounding_box']['size'])
    return 1.0 if size[-1]*size[-2]<.012 else 1.6


def hidden_path(corners, direction, hidden_points, camera):
    """Require a horizontal view margin along the whole planned manipulation route."""
    norm=math.hypot(direction[0],direction[2])
    if norm<1e-6:return False
    forward=(direction[0]/norm,direction[2]/norm)
    side,_,depth=camera.get('position',[.35,2.2,-1.7])
    half_fov=math.atan(math.tan(math.radians(75/2))*4/3)+math.radians(12)
    for start,end in zip(corners,corners[1:]):
        steps=max(1,math.ceil(planar_distance(start,end)/.2))
        for index in range(steps+1):
            point=[a+(b-a)*index/steps for a,b in zip(start,end)]
            cx=point[0]+side*forward[1]+depth*forward[0]
            cz=point[2]-side*forward[0]+depth*forward[1]
            for target in hidden_points:
                dx,dz=target[0]-cx,target[2]-cz
                distance=math.hypot(dx,dz)
                if distance<.8:return False
                angle=math.acos(max(-1,min(1,(dx*forward[0]+dz*forward[1])/distance)))
                if angle<=half_fov+math.asin(min(1,.5/distance)):return False
    return True


class NativePatrol:
    def __init__(self, graph, room_id, log_dir, output, targets=(), camera=None):
        self.log_dir=Path(log_dir)
        self.previous=None
        self.camera=camera or {}
        self.observed_stations={}
        self.observation_distances={node['id']:observation_distance(node)
                                    for node in graph['nodes'] if node['id'] in targets}
        origin=next(n for n in graph['nodes'] if n['id']==1)['obj_transform']['position']
        proposals=[origin,*spawn_points(graph,room_id,spacing=.5)]
        if len(proposals)>160:
            step=(len(proposals)-1+158)//159
            proposals=[origin,*proposals[1::step]]
        # Local edges are sufficient for continuous patrol and avoid room-scale jumps.
        edges=[[i,j] for i,a in enumerate(proposals) for j,b in enumerate(proposals)
               if i!=j and .6<=planar_distance(a,b)<=4.5]
        sight_targets=[dict(id=node['id'],center=node['bounding_box']['center'],size=node['bounding_box']['size'])
                       for node in graph['nodes'] if node['id'] in targets]
        result=request_paths(self.log_dir,proposals,edges,sight_targets=sight_targets,camera=camera)
        self.camera_route_probe=result.get('camera_route_probe',False)
        self.sight={}
        self.camera_stations={}
        for row in result.get('sight',[]):
            if row['clear_rays']>0:self.sight.setdefault(row['target_id'],set()).add(row['point'])
            if 'observer_camera_clear' in row:
                stations=self.camera_stations.setdefault(row['target_id'],{'visible':set(),'hidden':set()})
                if row['observer_camera_clear']:stations['visible'].add(row['point'])
                if row['hidden_camera_clear']:stations['hidden'].add(row['point'])
        self.points={row['index']:row['position'] for row in result['points'] if row['valid']}
        self.paths={}
        for row in result['paths']:
            if row['complete'] and .6<=row['length']<=5.0 and len(row['corners'])>=2:
                self.paths.setdefault(row['from'],[]).append(row)
        result['scope']='native_static_patrol_graph'
        result['plan_id']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
        Path(output).write_text(json.dumps(result,indent=2))
        if len(self.paths)<2:raise ValueError('The room has no connected native patrol graph')

    def note_visible(self, target_id, observer_position):
        """Retain a navigation station backed by an actual visible target mask."""
        nearest=min(self.points,key=lambda key:planar_distance(observer_position,self.points[key]))
        if planar_distance(observer_position,self.points[nearest])<=.4:
            self.observed_stations[target_id]=nearest

    def operator_segments(self, origin, destination):
        """Reserve the NPC's NavMesh route, including bends around furniture."""
        key=tuple(round(value,3) for point in (origin,destination) for value in point)
        if not hasattr(self,'operator_paths'):
            self.operator_paths={}
        if key not in self.operator_paths:
            report=request_paths(self.log_dir,[origin,destination],[[0,1]])
            paths=[row for row in report['paths'] if row['complete']]
            if len(paths)!=1 or len(paths[0]['corners'])<2:
                raise ValueError('The planned operator destination has no complete native route')
            corners=paths[0]['corners']
            self.operator_paths[key]=list(zip(corners,corners[1:]))
            with (self.log_dir/'operator_routes.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(origin=origin,destination=destination,corners=corners))+'\n')
        return self.operator_paths[key]

    def refresh_sight(self, target):
        """Recheck frozen observation stations after objects or operators move."""
        indices=sorted(self.points)
        report=request_paths(self.log_dir,[self.points[index] for index in indices],[],
            sight_targets=[dict(id=target['id'],center=target['bounding_box']['center'],
                                size=target['bounding_box']['size'])],camera=self.camera,fresh=True)
        visible=set();stations={'visible':set(),'hidden':set()}
        for row in report.get('sight',[]):
            station=indices[row['point']]
            if row['clear_rays']>0:visible.add(station)
            if row.get('observer_camera_clear'):stations['visible'].add(station)
            if row.get('hidden_camera_clear'):stations['hidden'].add(station)
        self.sight[target['id']]=visible
        if not hasattr(self,'observation_distances'):self.observation_distances={}
        # Rotation can inflate a held prop's axis-aligned box without making
        # its thin silhouette easier to see.
        self.observation_distances[target['id']]=min(
            self.observation_distances.get(target['id'],float('inf')),
            observation_distance(target))
        self.camera_stations[target['id']]=stations
        if self.observed_stations.get(target['id']) not in visible:
            self.observed_stations.pop(target['id'],None)
        with (self.log_dir/'visibility_refresh.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(target_id=target['id'],
                center=target['bounding_box']['center'],visible_stations=sorted(visible),
                request_id=report['id']))+'\n')

    def choose(self, origin, occupied, target, anchor_id, clearance=1.1, operation_seconds=None,
               reserved_segments=(), future_goal=None, acquire=False, hidden=False, target_id=None,
               hidden_direction=None, hidden_points=(), timing_seconds=None):
        nearest=min(self.points,key=lambda i:planar_distance(origin,self.points[i]))
        if planar_distance(origin,self.points[nearest])>.65:
            raise ValueError('The observer left the frozen navigation graph')
        outgoing=self.paths.get(nearest,[])
        camera_penalties={}
        if getattr(self,'camera_route_probe',False) and outgoing:
            values=camera_route_penalties(self.log_dir,[[origin,*row['corners'][1:]] for row in outgoing])
            camera_penalties={row['to']:value for row,value in zip(outgoing,values)}
        viable=getattr(self,'sight',{}).get(target_id,set())
        station_report=getattr(self,'camera_stations',{}).get(target_id)
        camera_safe=station_report['hidden' if hidden else 'visible'] if station_report is not None else None
        if camera_safe is not None:viable=viable & camera_safe
        observed=getattr(self,'observed_stations',{}).get(target_id)
        view_distance=getattr(self,'observation_distances',{}).get(target_id,1.6)
        if observed is not None:viable=set(viable)|{observed}
        if acquire and viable and operation_seconds is None and timing_seconds is None:
            distances={nearest:0.0};first_edges={};queue=[(0.0,nearest)]
            while queue:
                cost,index=heapq.heappop(queue)
                if cost>distances[index]:continue
                for edge in self.paths.get(index,[]):
                    if edge['length']>1.7 or path_clearance(edge['corners'],occupied)<clearance:continue
                    if camera_safe is not None and edge['to'] not in camera_safe:continue
                    if any(segment_segment_distance(start,end,*segment)<clearance
                           for start,end in zip(edge['corners'],edge['corners'][1:])
                           for segment in reserved_segments):continue
                    end=edge['to'];candidate=cost+edge['length']
                    if index==nearest:candidate += 1000 * bool(camera_penalties.get(end))
                    if candidate>=distances.get(end,float('inf')):continue
                    distances[end]=candidate
                    first_edges[end]=edge if index==nearest else first_edges[index]
                    heapq.heappush(queue,(candidate,end))
            goals=[index for index in viable if index in first_edges]
            if goals:
                goal=(observed if observed in goals else min(goals,key=lambda index:
                      distances[index]+3.0*abs(planar_distance(self.points[index],target)-view_distance)))
                row=first_edges[goal]
                return dict(anchor_id=anchor_id,navigation_position=self.points[row['to']],
                    navigation_path=[origin,*row['corners'][1:]],planned_turn_degrees=0.0,
                    observation_station=goal)
        choices=[]
        for row in self.paths.get(nearest,[]):
            goal=self.points[row['to']]
            corners=[origin,*row['corners'][1:]]
            if hidden_direction is not None and not hidden_path(corners,hidden_direction,hidden_points,self.camera):continue
            initial=min((planar_distance(origin,person) for person in occupied),default=clearance)
            if path_clearance(corners,occupied)<max(.9,min(clearance,initial)-.01):continue
            if any(planar_distance(goal,person)<clearance for person in occupied):continue
            if any(segment_segment_distance(start,end,*segment)<clearance
                   for start,end in zip(corners,corners[1:]) for segment in reserved_segments):continue
            if future_goal is not None and planar_distance(goal,future_goal)<clearance:continue
            length=row['length']
            if operation_seconds is None and length>1.7:continue
            heading=route_turn_degrees(self.previous,origin,corners[1]) if self.previous is not None else 0.0
            distance=planar_distance(goal,target)
            desired=min(4.0,max(1.2,(operation_seconds or 1.2)*1.1))
            if timing_seconds is not None:
                desired=max(.6,min(1.7,(timing_seconds-.35)*1.1))
            if operation_seconds is not None:
                # During a joint action, finishing a short walk leaves the observer
                # idle until the NPC finishes. Visibility and clearance were checked
                # above; prefer a route whose travel time matches the action.
                score=abs(length-desired)*3.0+heading/180.0+abs(distance-2.3)*.3
            elif hidden:
                score=abs(length-desired)+heading/180.0+max(0,3.0-distance)*4.0
            elif acquire:
                score=abs(length-desired)+heading/300.0+abs(distance-view_distance)*3.0
            else:
                score=abs(length-desired)+heading/90.0+abs(distance-2.3)*.35
            if camera_safe is not None and row['to'] not in camera_safe:
                score += 1000.0
            score += 1000.0 * bool(camera_penalties.get(row['to']))
            if timing_seconds is not None:
                # A final short walk should not consume the NPC's preparation window.
                estimated = length / 1.1 + .35 + heading / 180.0
                score += max(0.0, estimated - timing_seconds) * 8.0
                if acquire and viable and row['to'] not in viable:
                    score += 4.0
            choices.append((score,goal,corners,heading))
        if not choices:return None
        _,goal,corners,heading=min(choices,key=lambda item:item[0])
        return dict(anchor_id=anchor_id,navigation_position=goal,navigation_path=corners,
                    planned_turn_degrees=heading,concurrent_seconds=operation_seconds)

    def set_destination(self, origin, waypoint):
        goal=waypoint['navigation_position']
        path=waypoint['navigation_path']
        last=path[-2]
        length=max(planar_distance(goal,last),1e-6)
        direction=[(goal[0]-last[0])/length,0,(goal[2]-last[2])/length]
        data=dict(position=goal,look_at=[goal[i]+direction[i] for i in range(3)])
        if waypoint.get('concurrent_seconds'):
            travel=sum(planar_distance(a,b) for a,b in zip(path,path[1:]))
            # Slow the observer's native root-motion animation on short joint
            # walks instead of arriving early and waiting for the operator.
            duration=max(.6,waypoint['concurrent_seconds']-.3)
            data['animation_speed_scale']=max(.45,min(1.0,travel/(1.7*duration)))
        temporary=self.log_dir/'observer_destination.tmp'
        temporary.write_text(json.dumps(data))
        temporary.replace(self.log_dir/'observer_destination.json')
        self.previous=list(origin)
