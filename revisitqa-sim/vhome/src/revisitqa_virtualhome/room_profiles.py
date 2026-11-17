"""Derive room-local candidate recipes for native validation.

Graph geometry supplies proposals, not a navigation or visibility certificate.
Every generated recipe must pass the full renderer audits before publication.
"""
import copy
import math
import random
from collections import Counter

from .scene import position, room_id_of, support_of
from .qa import label


def distance(a, b):
    return math.hypot(position(a)[0] - position(b)[0], position(a)[2] - position(b)[2])


def build_profile(graph, template, scene, room_id, seed):
    nodes = {n["id"]: n for n in graph["nodes"]}
    room = nodes[room_id]
    if room.get("category", "").lower() != "rooms":
        raise ValueError("The selected node is not a room")
    members = [n for n in graph["nodes"] if room_id_of(graph, n["id"]) == room_id]
    rng = random.Random(seed)
    rng.shuffle(members)
    counts = Counter(n["class_name"] for n in members)
    # Unique visible names avoid ambiguous QA references to identical furniture.
    openables = [n for n in members if "CAN_OPEN" in n.get("properties", [])
                 and "CLOSED" in n.get("states", []) and "GRABBABLE" not in n["properties"]
                 and n["class_name"] not in ("window", "curtains", "door")
                 and counts[n["class_name"]] == 1]
    switches = [n for n in members if "HAS_SWITCH" in n.get("properties", [])
                and "OFF" in n.get("states", [])
                and n["class_name"] in ("tv", "computer", "radio", "tablelamp", "floorlamp")
                and counts[n["class_name"]] == 1]
    surfaces = [n for n in members if "SURFACES" in n.get("properties", [])
                and "GRABBABLE" not in n["properties"]
                and n["class_name"] not in ("floor", "ceiling", "wall", "bookshelf", "bed", "sofa")
                and position(n)[1] + n["bounding_box"]["size"][1] / 2 <= 1.3]
    portable = [n for n in members if "GRABBABLE" in n.get("properties", [])
                and counts[n["class_name"]] == 1 and support_of(graph, n["id"]) is not None
                and room_id_of(graph, support_of(graph, n["id"])["id"]) == room_id
                and max(n["bounding_box"]["size"]) < 0.6
                and 0.2 < position(n)[1] < 1.4
                and n["class_name"] not in ("mouse", "keyboard", "paper")]
    if len(openables) < 2 or not switches or len(portable) < 2 or len(surfaces) < 2:
        raise ValueError(f"Insufficient distinct targets: open={len(openables)}, "
                         f"switch={len(switches)}, portable={len(portable)}, surfaces={len(surfaces)}")
    switch = min(switches, key=lambda n: (n["class_name"] != "tv", n["class_name"] != "computer"))
    openables = [n for n in openables if n["id"] != switch["id"]]
    if len(openables) < 2:
        raise ValueError("The switch and opening targets must be different objects")
    # Keep the manipulation cluster away from the hidden viewing direction.
    primary = max(openables, key=lambda n: distance(n, switch))
    openables.sort(key=lambda n: distance(n, primary))
    openables = openables[:3]
    portable = [n for n in portable if n["id"] != switch["id"]]
    portable.sort(key=lambda n: distance(n, primary))
    if len(portable) < 2:
        raise ValueError("The switch must not also be picked up")
    portable = portable[:2]
    surfaces.sort(key=lambda n: distance(n, primary))
    primary_surface = surfaces[0]
    secondary = min(surfaces[1:], key=lambda n: abs(distance(n, primary_surface) - 2.5))
    anchors = [n for n in members if "GRABBABLE" not in n.get("properties", [])
               and n["class_name"] in ("desk", "cabinet", "nightstand", "closet", "bookshelf",
                                       "coffeetable", "kitchentable", "kitchencounter", "sofa", "tvstand", "bench")]
    hidden = switch
    midpoint = [(position(primary)[a] + position(hidden)[a]) / 2 for a in (0, 2)]
    away_choices = [n for n in anchors if n["id"] != primary["id"] and distance(n, primary) >= 1.2]
    if not away_choices:
        raise ValueError("The observation route overlaps the operator's work position")
    away = min(away_choices, key=lambda n: math.hypot(position(n)[0] - midpoint[0], position(n)[2] - midpoint[1]))
    operating_ids = {n['id'] for n in openables} | {switch['id']}
    observation_choices = [n for n in anchors if n['id'] not in operating_ids and n['id'] != away['id']]
    if not observation_choices:
        raise ValueError("No separate observation anchor is available")
    observation = min(observation_choices, key=lambda n: distance(n, primary))
    far = [n for n in anchors if n["id"] != hidden["id"] and distance(n, hidden) >= 1.2]
    if not far:
        raise ValueError("The room has no separated patrol anchors")
    patrol = min(far, key=lambda n: abs(distance(n, hidden) - 2.0))
    heading = math.degrees(math.atan2(position(hidden)[0] - position(primary)[0],
                                     position(hidden)[2] - position(primary)[2]))
    config = copy.deepcopy(template)
    config.update(scene=scene, seed=seed, room_id=room_id, room_name=room["class_name"],
                  profile_status="requires_native_validation", scene_signature_mode="room_recipe", surface_id=primary_surface["id"],
                  offscreen_camera_heading_degrees=heading, room_adaptive_view=True)
    config["mixed_candidates"] = dict(portable=[n["id"] for n in portable],
                                      cabinets=[n["id"] for n in openables[1:]],
                                      appliance=primary["id"], switch=switch["id"],
                                      secondary_surface_id=secondary["id"])
    config["navigation"] = dict(kitchen_anchor=observation["id"], away_anchor=away["id"],
                                tv_side_anchor=patrol["id"], tv_walk_anchor=patrol["id"],
                                worker_start=primary["id"], hidden_anchor=hidden["id"],
                                placement_anchor=secondary["id"], offscreen_patrol_anchor=patrol["id"])
    config["object_descriptions"] = {n["id"]: label(n["class_name"]) for n in [*openables, *portable, switch]}
    center, size = room["bounding_box"]["center"], room["bounding_box"]["size"]
    floors = [n for n in members if n["class_name"] == "floor"]
    if not floors:
        raise ValueError("No floor elevation is available for this room")
    floor_y = min(position(n)[1] for n in floors)
    config["camera_continuity"]["position_min"] = [center[0] - size[0] / 2 + 0.35, floor_y + 1.8, center[2] - size[2] / 2 + 0.35]
    config["camera_continuity"]["position_max"] = [center[0] + size[0] / 2 - 0.35, floor_y + 2.28, center[2] + size[2] / 2 - 0.35]
    # Reserve separated spawn points clear of furniture AABBs. Native navigation
    # still verifies the proposals; this test does not replace the NavMesh.
    # Structural tile bounds enclose empty floor space as well as their walls.
    # Room insets constrain spawning; native colliders validate the actual shell.
    obstacles = [n for n in members if n["class_name"] not in ("floor", "wall", "ceiling")
                 and n.get("category", "").lower() != "rooms"
                 and "GRABBABLE" not in n.get("properties", [])
                 and position(n)[1] + n["bounding_box"]["size"][1] / 2 > floor_y + 0.1
                 and position(n)[1] - n["bounding_box"]["size"][1] / 2 < floor_y + 1.7]
    points = []
    for ix in range(1, int(size[0] / 0.5)):
        for iz in range(1, int(size[2] / 0.5)):
            p = [center[0] - size[0] / 2 + ix * 0.5, floor_y, center[2] - size[2] / 2 + iz * 0.5]
            if any(all(abs(p[a] - position(n)[a]) < n["bounding_box"]["size"][a] / 2 + 0.35 for a in (0, 2)) for n in obstacles):
                continue
            points.append(p)
    for role, goal in (("observer", away), ("injector", primary), ("helper", switch)):
        if not points:
            raise ValueError("The room has insufficient free spawn space")
        chosen = min(points, key=lambda p: math.dist(p, position(goal)))
        config[role]["room"] = room["class_name"]
        config[role]["position"] = chosen
        points = [p for p in points if math.dist(p, chosen) >= 0.9]
    return config
