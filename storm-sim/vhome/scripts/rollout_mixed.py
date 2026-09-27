#!/usr/bin/env python3
"""Record a planned household routine performed by two operating characters."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import imageio.v2 as imageio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storm_virtualhome.client import UnityProcess
from storm_virtualhome.planning import create_program, validate_program, audit_execution, preferred_start, check_injection_start
from storm_virtualhome.mixed import (
    check_effect,
    audit_mixed,
    build_mixed_questions,
)
from storm_virtualhome.recording import Recorder, visible_pixels
from storm_virtualhome.scene import action
from storm_virtualhome.raw_qa import write_raw_questions


def finalize(root, defer_qa=False):
    audit_execution(root)
    report = audit_mixed(root)
    runtime = json.loads((root / 'runtime_manifest.json').read_text())
    if runtime.get('per_frame_actor_positions'):
        from storm_virtualhome.separation import audit_separation
        separation = audit_separation(root)
        (root / 'actor_separation.json').write_text(json.dumps(separation, indent=2))
        if separation['status'] != 'separation_verified' or not separation['exact_root_evidence']:
            raise ValueError(f"Frame-level actor separation failed: {separation['status']}")
    config = json.loads((root / "config.json").read_text())
    events = json.loads((root / "events.json").read_text())
    with imageio.get_reader(str(root / "video.mp4")) as v:
        if (
            v.count_frames() != report["frames"]
            or v.get_meta_data()["fps"] != report["fps"]
        ):
            raise ValueError("Video metadata does not match recorded evidence")
    print(json.dumps(report, indent=2), flush=True)
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("render_debug.py")), str(root)],
        check=True,
    )
    # Draft annotations belong to the capture; only batch curation is deferred.
    write_raw_questions(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable", required=True)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/randomized.yaml",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--xorg-root", type=Path)
    p.add_argument("--gpu-index", type=int, default=3)
    p.add_argument("--display")
    p.add_argument("--seed", type=int)
    p.add_argument("--plan", type=Path, help="Execute a frozen event program without resampling")
    p.add_argument("--plan-only", action="store_true", help="Save the scene and event program without action rendering")
    args = p.parse_args()
    config = yaml.safe_load(args.config.read_text())
    supplied_program = json.loads(args.plan.read_text()) if args.plan else None
    if supplied_program is not None and args.seed is None:
        config["seed"] = supplied_program["seed"]
    if args.seed is not None:
        config["seed"] = args.seed
    anchors = config["navigation"]
    kitchen_anchor = anchors["kitchen_anchor"]
    away_anchor = anchors["away_anchor"]
    tv_side_anchor = anchors["tv_side_anchor"]
    tv_walk_anchor = anchors.get("tv_walk_anchor", tv_side_anchor)
    worker_start = anchors["worker_start"]
    hidden_anchor = anchors["hidden_anchor"]
    placement_anchor = anchors["placement_anchor"]
    offscreen_patrol_anchor = anchors.get("offscreen_patrol_anchor", away_anchor)
    television_id = config["mixed_candidates"]["switch"]
    appliance_id = config["mixed_candidates"]["appliance"]
    root = args.output.resolve()
    runtime = Path(args.executable).resolve().parent / "camera_guard.json"
    if config.get("camera_continuity"):
        if not runtime.is_file() or json.loads(runtime.read_text()).get("controller") != "continuous":
            p.error("Build the continuous-camera runtime before using this preset")
    if config.get("camera_continuity", {}).get("position_smooth_seconds"):
        if json.loads(runtime.read_text()).get("smoothing") != "damped_position_and_angles":
            p.error("Build the damped-camera runtime before using the smoothing preset")
    if config.get("offscreen_camera_heading_degrees") is not None and not json.loads(runtime.read_text()).get("fixed_view_direction"):
        p.error("Build a camera runtime with fixed-direction tracking for this preset")
    root.mkdir(parents=True, exist_ok=False)
    if runtime.exists():
        shutil.copy2(runtime, root / "runtime_manifest.json")
    with UnityProcess(
        args.executable,
        root / "logs",
        gpu_index=args.gpu_index,
        display=args.display,
        xorg_root=args.xorg_root,
        camera_rules=config.get("camera_continuity"),
    ) as c:
        c.reset(config["scene"])
        if config.get("camera_continuity"):
            scene_graph = c.graph()
            keepouts = [dict(n["bounding_box"], object_id=n["id"]) for n in scene_graph["nodes"] if n["class_name"] == "ceilinglamp"]
            (root / "logs/camera_keepouts.json").write_text(json.dumps({"obstacles": keepouts}, indent=2))
        c.follow_camera(**config["camera"])
        count = c.command("camera_count")["value"]
        names = c.command("character_cameras")["message"]
        if isinstance(names, str):
            names = json.loads(names)
        camera = count + names.index(config["camera"]["name"])
        c.character(**config["observer"])
        c.character(**config["injector"])
        c.character(**config["helper"])
        config["camera"]["recording_camera"] = str(camera)
        (root / "config.json").write_text(json.dumps(config, indent=2))
        graph = c.graph()
        nodes = {n["id"]: n for n in graph["nodes"]}
        (root / "planning_graph.json").write_text(json.dumps(graph, indent=2))
        program = supplied_program if supplied_program is not None else create_program(graph, config)
        validate_program(program, graph, config)
        (root / "event_program.json").write_text(json.dumps(program, indent=2))
        plan = program["events"]
        (root / "plan.json").write_text(json.dumps(plan, indent=2))
        print("PLAN", [(e["task"], e["object_id"]) for e in plan], flush=True)
        if args.plan_only:
            print("Saved event program:", root / "event_program.json", flush=True)
            return
        setup = [
            action("Walk", nodes[kitchen_anchor], character=0),
            action("Walk", nodes[television_id], character=0)
            + " | "
            + action("Walk", nodes[worker_start], character=1),
            action("Walk", nodes[away_anchor], character=0)
            + " | "
            + action("Walk", nodes[television_id], character=2),
        ]
        if config.get("room_adaptive_view"):
            setup = [
                action("Walk", nodes[tv_walk_anchor], character=0)
                + " | " + action("Walk", nodes[worker_start], character=1)
                + " | " + action("Walk", nodes[television_id], character=2),
                action("WalkTowards", nodes[television_id], character=0),
            ]
        elif config.get("camera_continuity"):
            setup.extend([
                action("WalkTowards", nodes[television_id], character=0),
                action("TurnTo", nodes[3], character=0) + " | " + action("TurnTo", nodes[away_anchor], character=1),
                action("TurnTo", nodes[3], character=0) + " | " + action("TurnTo", nodes[worker_start], character=1),
            ])
        for i, line in enumerate(setup):
            if config.get("camera_continuity") and (i >= 3 or config.get("room_adaptive_view")):
                (root / "logs/camera_view_target.json").write_text(json.dumps({"position": nodes[television_id]["bounding_box"]["center"]}))
            print("SETUP", line, flush=True)
            folder = root / "setup" / f"{i:04d}"
            folder.mkdir(parents=True)
            response = c.render(
                [line], folder, "setup", camera=str(camera), width=320, height=240
            )
            statuses = json.loads(response.get("message", "{}"))
            if any(s.get("message") != "Success" for s in statuses.values()):
                raise RuntimeError(statuses)
        (root / "initial_graph.json").write_text(json.dumps(c.graph(), indent=2))
        if config.get("camera_continuity"):
            (root / "logs/camera_start").touch()
        recorder = Recorder(c, root, camera, config)
        stages = {}
        events = []
        deferred_closes = []
        pending_switch = None
        helper_departed = False
        at_anchor = False
        view_region = television_id
        hide_kitchen = False
        last_injection_start = None

        def run(verb, target, inject=None, helper=None, resident=None):
            nonlocal last_injection_start
            planned_injection = inject if inject is not None else pending_switch if helper and helper[0] == "SwitchOff" else None
            if planned_injection is not None:
                check_injection_start(planned_injection, recorder.time, last_injection_start, config)
            if verb == "TurnTo":
                current = {n["id"]: n for n in c.graph()["nodes"]}
                origin = current[1]["obj_transform"]["position"]
                goal = current[target]["obj_transform"]["position"]
                distance = sum((origin[axis] - goal[axis]) ** 2 for axis in (0, 2)) ** 0.5
                # Native watching requires a nearby target. Approach distant targets on foot.
                if distance > 1.5:
                    verb = "WalkTowards"
            line = action(verb, nodes[target], character=0)
            if inject is not None:
                dest = (
                    [nodes[inject["surface_id"]]] if inject["verb"] == "PutBack" else []
                )
                line += " | " + action(
                    inject["verb"],
                    nodes[inject["object_id"]],
                    *dest,
                    character=inject["injector_character"],
                )
            if resident is not None:
                if verb == "Walk" and resident[0] == "WalkTowards" and "CAN_OPEN" in nodes[resident[1]].get("properties", []):
                    # Use the observer's longer walk to finish the operator's approach.
                    resident = ("Walk", resident[1])
                line += " | " + action(resident[0], nodes[resident[1]], character=1)
            if helper is not None:
                line += " | " + action(helper[0], nodes[helper[1]], character=2)
            if config.get("camera_continuity"):
                view_nodes = {n["id"]: n for n in c.graph()["nodes"]} if config.get("room_adaptive_view") else nodes
                target = {"position": view_nodes[view_region]["bounding_box"]["center"], "region_object_id": view_region}
                if hide_kitchen and config.get("offscreen_camera_heading_degrees") is not None:
                    angle = math.radians(config["offscreen_camera_heading_degrees"])
                    target = {"direction": [math.sin(angle), 0, math.cos(angle)]}
                (root / "logs/camera_view_target.json").write_text(json.dumps(target))
            print("ACTION", line, flush=True)
            result = recorder.run(line)
            if planned_injection is not None:
                last_injection_start = result["start"]
            return result

        def hidden_destination():
            current = {n["id"]: n for n in c.graph()["nodes"]}
            origin = current[1]["obj_transform"]["position"]
            goal = current[hidden_anchor]["obj_transform"]["position"]
            distance = sum((origin[axis] - goal[axis]) ** 2 for axis in (0, 2)) ** 0.5
            return tv_walk_anchor if distance < 1.5 else hidden_anchor

        def visible_now(target):
            colors = c.colors()
            return visible_pixels(c.image(camera, mode="seg_inst"), colors[str(target)]) >= config["evidence"]["min_visible_pixels"]

        def snap(key, target):
            s = recorder.snapshot(key, target)
            stages[key] = {k: v for k, v in s.items() if k != "graph"}
            (root / "stages.json").write_text(json.dumps(stages, indent=2))
            print("SNAPSHOT", key, s["time"], s["visible_pixels"], flush=True)
            return s

        try:
            for i, event in enumerate(plan):
                e = dict(event, event_index=i, observer_character=0)
                if e["verb"] == "SwitchOff":
                    continue
                starts = [item["start"] for item in events]
                if pending_switch is not None and "start" in pending_switch:
                    starts.append(pending_switch["start"])
                target_start = preferred_start(e, max(starts) if starts else None, config["event_interval_seconds"])
                # Recover small delays instead of carrying them into every later event.
                target_start = min(target_start, e["planned_start_seconds"] - 0.75) if starts else target_start
                preposition = ("WalkTowards", e["object_id"]) if e["verb"] in ("Open", "Close", "Grab") else None
                television = e["injector_character"] == 2
                view_region = television_id if television else appliance_id
                if config.get("room_adaptive_view") and not television:
                    view_region = e["surface_id"] if e["verb"] == "PutBack" else e["object_id"]
                placed_elsewhere = (
                    e["verb"] == "PutBack" and e["surface_id"] != config["surface_id"]
                )
                after_look = (
                    e["object_id"] if placed_elsewhere else e["injector_node_id"]
                )
                recorder.set_target(nodes[e["object_id"]])
                if not television and events and events[-1]["injector_character"] == 2:
                    run("Walk", 2, helper=("Walk", tv_side_anchor))
                    at_anchor = False
                elif (
                    e["requested_visibility"] != "offscreen"
                    and events
                    and recorder.time - events[-1]["start"]
                    < config["minimum_start_gap_seconds"]
                ):
                    if at_anchor:
                        run("WalkTowards", away_anchor, resident=preposition)
                        at_anchor = False
                    else:
                        run("Walk", kitchen_anchor, resident=preposition)
                        at_anchor = True
                if television:
                    if at_anchor:
                        run("Walk", away_anchor)
                        at_anchor = False
                    if not config.get("room_adaptive_view") or not visible_now(television_id):
                        run("WalkTowards", television_id)
                lead = 2.0 if e["requested_visibility"] == "offscreen" else 0.1
                while e["requested_visibility"] != "offscreen" and recorder.time < target_start - lead - 0.6:
                    destination = away_anchor if at_anchor else kitchen_anchor
                    progress = recorder.time
                    remaining = target_start - lead - recorder.time
                    run("WalkTowards" if remaining < 1.5 else "Walk", destination, resident=preposition)
                    at_anchor = destination == kitchen_anchor
                    if recorder.time - progress < 0.2:
                        raise RuntimeError("The planned timing walk made no progress")
                before_key = f"event_{i:02d}_before"
                after_key = f"event_{i:02d}_after"
                if e["requested_visibility"] == "offscreen" and e["verb"] == "Close":
                    # The saved open-state inspection remains valid until this object's close.
                    prior = next(item for item in reversed(events)
                                 if item["object_id"] == e["object_id"] and item["verb"] == "Open")
                    before_key = prior["stage_keys"]["after"]
                    before = json.loads((root / "evidence" / f"{before_key}.json").read_text())
                    before["time"] = prior["discovery_time"]
                else:
                    colors = c.colors()
                    pixels = visible_pixels(c.image(camera, mode="seg_inst"), colors[str(e["object_id"])])
                    if pixels < config["evidence"]["min_visible_pixels"]:
                        run("TurnTo", e["injector_node_id"])
                    if e["verb"] in ("Open", "Close") and not visible_now(e["object_id"]):
                        run("WalkTowards", away_anchor, resident=preposition)
                        at_anchor = False
                    before = snap(before_key, e["object_id"])
                if before["color_collision_ids"]:
                    raise RuntimeError(f"Ambiguous instance mask for {e['task']}: shared with {before['color_collision_ids']}")
                if before["visible_pixels"] < config["evidence"]["min_visible_pixels"]:
                    raise RuntimeError(f"Target not visible before event: {e['task']}")
                if e["requested_visibility"] == "offscreen":
                    hide_kitchen = True
                    view_region = television_id
                    lower_start = e["planned_start_seconds"] - e["start_tolerance_seconds"]
                    warmup_seconds = 3.0 if e["verb"] == "Close" else 0.0
                    ready = target_start - warmup_seconds if warmup_seconds else lower_start if placed_elsewhere else max(lower_start, target_start - 1.4)
                    if events:
                        ready = max(ready, events[-1]["start"] + config["minimum_start_gap_seconds"] - warmup_seconds)
                    prepared_actor = False
                    for attempt in range(6):
                        colors = c.colors()
                        mask = c.image(camera, mode="seg_inst")
                        hidden_now = not visible_pixels(mask, colors[str(e["object_id"])]) and not visible_pixels(mask, colors[str(e["injector_node_id"])])
                        if hidden_now and recorder.time >= ready:
                            break
                        run("Walk" if placed_elsewhere and attempt == 0 else "WalkTowards", hidden_destination(),
                            resident=(("Walk", e["surface_id"]) if placed_elsewhere else preposition) if attempt == 0 else None,
                            helper=("WalkTowards", television_id) if placed_elsewhere and attempt == 0 else ("WalkTowards", tv_side_anchor) if e["verb"] == "Close" and not helper_departed else None)
                        prepared_actor = True
                        if e["verb"] == "Close":
                            helper_departed = True
                        if recorder.time > e["planned_start_seconds"] + e["start_tolerance_seconds"]:
                            raise RuntimeError("The offscreen route missed its planned start window")
                    else:
                        raise RuntimeError("The observer could not leave the injection area out of view")
                    injection_destination = tv_walk_anchor if placed_elsewhere else hidden_destination()
                    if not placed_elsewhere:
                        # Complete the corner turn before the hidden manipulation starts.
                        injection_destination = tv_walk_anchor
                        for turn in range(3):
                            turn_destination = hidden_anchor if e["verb"] == "Close" and turn % 2 else injection_destination
                            run("Walk" if e["verb"] == "Close" else "WalkTowards", turn_destination,
                                resident=preposition if not prepared_actor or e["verb"] == "Close" else None,
                                helper=("WalkTowards", tv_side_anchor) if e["verb"] == "Close" and not helper_departed else None)
                            prepared_actor = True
                            if e["verb"] == "Close":
                                helper_departed = True
                            colors = c.colors()
                            mask = c.image(camera, mode="seg_inst")
                            if not any(visible_pixels(mask, colors[str(key)]) for key in (e["object_id"], e["injector_node_id"])):
                                break
                            if recorder.time > e["planned_start_seconds"] + e["start_tolerance_seconds"]:
                                raise RuntimeError("The camera turn missed the offscreen event window")
                        else:
                            raise RuntimeError("The corner turn did not keep the injection area out of view")
                    start = recorder.time
                    if abs(start - e["planned_start_seconds"]) > e["start_tolerance_seconds"] + 1e-6:
                        raise RuntimeError(f"Offscreen start {start:.2f}s is outside the planned window for {e['task']}")
                    operation = run(
                        "Walk",
                        offscreen_patrol_anchor if e["verb"] in ("Grab", "Close") else injection_destination,
                        inject=e,
                    )
                    hidden = snap(f"event_{i:02d}_hidden", e["object_id"])
                    e["hidden_time"] = hidden["time"]
                    defer_inspection = e["verb"] == "Close" and any(item["verb"] == "Close" for item in plan[i + 1:])
                    hide_kitchen = defer_inspection
                    view_region = television_id if defer_inspection else e["surface_id"] if placed_elsewhere else appliance_id
                    if config.get("room_adaptive_view") and not defer_inspection:
                        view_region = e["surface_id"] if placed_elsewhere else e["object_id"]
                    if placed_elsewhere:
                        pending_switch = dict(
                            next(item for item in plan if item["verb"] == "SwitchOff"),
                            event_index=next(j for j, item in enumerate(plan) if item["verb"] == "SwitchOff"),
                            observer_character=0,
                        )
                        recorder.set_target(nodes[television_id])
                        lower = max(pending_switch["planned_start_seconds"] - 1.2,
                                    start + config["minimum_start_gap_seconds"])
                        timing_destination = hidden_anchor
                        while recorder.time < lower:
                            progress = recorder.time
                            run("Walk", timing_destination)
                            timing_destination = hidden_anchor if timing_destination == tv_walk_anchor else tv_walk_anchor
                            if recorder.time - progress < 0.2:
                                raise RuntimeError("The switch-off timing walk made no progress")
                        returning = run(
                            "Walk", away_anchor,
                            helper=("SwitchOff", television_id),
                            resident=("Walk", next((item["object_id"] for item in plan[i + 1:] if item["verb"] == "Close"), appliance_id)),
                        )
                        pending_switch.update(
                            start=returning["start"], end=returning["end"],
                            before_time=events[0]["discovery_time"],
                            stage_keys={"before": events[0]["stage_keys"]["after"],
                                        "after": "television_final_inspection"},
                        )
                        recorder.set_target(nodes[e["object_id"]])
                    else:
                        resident = ("WalkTowards", placement_anchor) if e["verb"] == "Grab" else None
                        if not defer_inspection:
                            run("Walk", away_anchor, resident=resident)
                    at_anchor = False
                    if defer_inspection:
                        # Discover related closures together when the observer returns to the kitchen.
                        e.update(start=start, end=operation["end"], before_time=before["time"],
                                 stage_keys={"before": before_key, "after": after_key})
                        events.append(e)
                        deferred_closes.append(e)
                        (root / "events.json").write_text(json.dumps(events, indent=2))
                        continue
                else:
                    start = recorder.time
                    if abs(start - e["planned_start_seconds"]) > e["start_tolerance_seconds"] + 1e-6:
                        raise RuntimeError("The observation route missed its planned start window")
                    if television:
                        destination = away_anchor
                    elif at_anchor:
                        destination = away_anchor
                    else:
                        destination = kitchen_anchor
                    if i + 1 < len(plan) and plan[i + 1]["verb"] == "Grab":
                        position = next(node for node in c.graph()["nodes"] if node["id"] == 1)["obj_transform"]["position"]
                        destination = max((away_anchor, kitchen_anchor), key=lambda key:
                                          sum((position[axis] - nodes[key]["obj_transform"]["position"][axis]) ** 2 for axis in (0, 2)))
                    operation = run(
                        "Walk",
                        destination,
                        inject=e,
                        helper=("TurnTo", television_id) if i == 1 else ("Walk", tv_side_anchor) if e["verb"] == "Close" and not helper_departed else None,
                    )
                    if e["verb"] == "Close":
                        helper_departed = True
                    at_anchor = destination in (kitchen_anchor, 2)
                if e["verb"] == "Grab":
                    if e["requested_visibility"] != "offscreen":
                        run("WalkTowards", television_id,
                            resident=("WalkTowards", placement_anchor))
                        at_anchor = False
                    if not visible_now(e["object_id"]):
                        run("WalkTowards", kitchen_anchor)
                elif e["verb"] == "PutBack" and not placed_elsewhere:
                    run(
                        "WalkTowards",
                        appliance_id,
                        resident=("WalkTowards", appliance_id),
                    )
                    if not visible_now(e["object_id"]):
                        run("TurnTo", after_look)
                elif not visible_now(e["object_id"]):
                    run(
                        "TurnTo",
                        after_look,
                        resident=("TurnTo", appliance_id)
                        if e["verb"] == "Grab"
                        else None,
                    )
                colors = c.colors()
                pixels = visible_pixels(
                    c.image(camera, mode="seg_inst"), colors[str(e["object_id"])]
                )
                if pixels < config["evidence"]["min_visible_pixels"]:
                    if e["verb"] in ("Open", "Close"):
                        run("WalkTowards", away_anchor)
                    run("TurnTo", after_look)
                after = snap(after_key, e["object_id"])
                if after["visible_pixels"] < config["evidence"]["min_visible_pixels"]:
                    raise RuntimeError(f"Target not visible after event: {e['task']}")
                check_effect(before["graph"], after["graph"], e)
                e.update(
                    start=start,
                    end=operation["end"],
                    before_time=before["time"],
                    discovery_time=after["time"],
                    stage_keys={"before": before_key, "after": after_key},
                )
                events.append(e)
                if e["verb"] == "Close" and deferred_closes:
                    for deferred in deferred_closes:
                        recorder.set_target(nodes[deferred["object_id"]])
                        if not visible_now(deferred["object_id"]):
                            run("WalkTowards", away_anchor)
                        inspection = snap(deferred["stage_keys"]["after"], deferred["object_id"])
                        if inspection["visible_pixels"] < config["evidence"]["min_visible_pixels"]:
                            raise RuntimeError("A deferred closure was not visible on return")
                        previous = json.loads((root / "evidence" / f"{deferred['stage_keys']['before']}.json").read_text())
                        check_effect(previous["graph"], inspection["graph"], deferred)
                        deferred["discovery_time"] = inspection["time"]
                    deferred_closes.clear()
                (root / "events.json").write_text(json.dumps(events, indent=2))
            if pending_switch is None:
                raise RuntimeError("The television operator did not finish the routine")
            recorder.set_target(nodes[television_id])
            view_region = television_id
            run("Walk", television_id)
            run("TurnTo", television_id)
            final_tv = snap(pending_switch["stage_keys"]["after"], television_id)
            pending_switch["discovery_time"] = final_tv["time"]
            events.append(pending_switch)
            events.sort(key=lambda item: item["event_index"])
            (root / "events.json").write_text(json.dumps(events, indent=2))
            tail = 0
            while recorder.time < config.get(
                "target_duration_seconds", config["duration_range_seconds"][0]
            ):
                t = recorder.time
                view_region = appliance_id
                run("Walk", (away_anchor, kitchen_anchor)[tail % 2])
                tail += 1
                if recorder.time - t < 0.2:
                    raise RuntimeError("The closing walk made no progress")
        finally:
            recorder.close()
    finalize(root)


if __name__ == "__main__":
    main()
