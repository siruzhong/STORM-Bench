"""Plan and verify household routines with different object changes."""

import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .debug_view import camera_audit, target_mask
from .motion import audit_motion
from .continuity import audit_continuity
from .observer_validation import check_roles
from .qa import label
from .recording import frame_interval, target_visibility_threshold, visible_pixels
from .scene import held_by, on_surface, support_of
from .schedule import check_duration

STATE_EFFECTS = {
    "Open": ("CLOSED", "OPEN", "open"),
    "Close": ("OPEN", "CLOSED", "close"),
    "SwitchOn": ("OFF", "ON", "toggle_on"),
    "SwitchOff": ("ON", "OFF", "toggle_off"),
}


def plan_routine(graph, config):
    if config.get("randomize_event_mix", False):
        return plan_random_mix(graph, config)
    rng = random.Random(config["seed"])
    nodes = {n["id"]: n for n in graph["nodes"]}
    pool = config["mixed_candidates"]
    portable = rng.sample(pool["portable"], 2)
    cabinet = rng.choice(pool["cabinets"])
    microwave, switch = pool["appliance"], pool["switch"]
    for key in portable:
        if (
            "GRABBABLE" not in nodes[key]["properties"]
            or support_of(graph, key) is None
        ):
            raise ValueError("Portable candidates must be supported, grabbable objects")
    for key in (cabinet, microwave):
        if (
            "CAN_OPEN" not in nodes[key]["properties"]
            or "CLOSED" not in nodes[key]["states"]
        ):
            raise ValueError("Openable candidates must begin closed")
    if (
        "HAS_SWITCH" not in nodes[switch]["properties"]
        or "OFF" not in nodes[switch]["states"]
    ):
        raise ValueError("The switch target must support switching and begin off")
    openables = [("cabinet", cabinet), ("appliance", microwave)]
    rng.shuffle(openables)
    first, second = (name for name, _ in openables)
    tasks = {
        "switch_on": ("SwitchOn", switch, []),
        f"{first}_open": ("Open", openables[0][1], ["switch_on"]),
        "pickup_a": ("Grab", portable[0], [f"{first}_open"]),
        "place_a": ("PutBack", portable[0], ["pickup_a"]),
        f"{second}_open": ("Open", openables[1][1], ["place_a"]),
        "pickup_b": ("Grab", portable[1], [f"{second}_open"]),
        "place_b": ("PutBack", portable[1], ["pickup_b"]),
        "cabinet_close": ("Close", cabinet, ["switch_off", "cabinet_open"]),
        "appliance_close": ("Close", microwave, ["switch_off", "appliance_open"]),
        "switch_off": ("SwitchOff", switch, ["place_b", "switch_on"]),
    }
    if config.get("randomize_event_order", False):
        for name, target in openables:
            tasks[f"{name}_open"] = ("Open", target, ["switch_on"])
        tasks["pickup_a"] = ("Grab", portable[0], ["switch_on"])
        tasks["pickup_b"] = ("Grab", portable[1], ["place_a"])
        tasks["place_b"] = ("PutBack", portable[1], ["pickup_b", "cabinet_open", "appliance_open"])
    done, result = {}, []
    while len(result) < len(tasks):
        eligible = []
        for key, (verb, target, deps) in tasks.items():
            if key in done or any(d not in done for d in deps):
                continue
            if config.get("randomize_event_order", False) and verb == "Open":
                if any(f"pickup_{suffix}" in done and f"place_{suffix}" not in done for suffix in ("a", "b")):
                    continue
            if config.get("randomize_event_order", False) and key == "pickup_b":
                if any(name not in done for name in ("cabinet_open", "appliance_open")):
                    continue
            if key == "appliance_close" and len(result) - done["appliance_open"] < 3:
                continue
            eligible.append(key)
        if not eligible:
            raise ValueError("The routine has no executable next task")
        key = rng.choice(eligible)
        verb, target, deps = tasks[key]
        destination = (
            pool.get("secondary_surface_id", config["surface_id"])
            if key in ("pickup_b", "place_b")
            else config["surface_id"]
        )
        if "SURFACES" not in nodes[destination]["properties"]:
            raise ValueError("Placement destinations must be support surfaces")
        done[key] = len(result)
        result.append(
            {
                "task": key,
                "verb": verb,
                "object_id": target,
                "object": nodes[target]["class_name"],
                "event_family": "state_change" if verb in STATE_EFFECTS else "movement",
                "event_action": STATE_EFFECTS[verb][2]
                if verb in STATE_EFFECTS
                else "move",
                "operation": verb,
                "injector_character": 2 if target == switch else 1,
                "injector_node_id": 3 if target == switch else 2,
                "surface_id": destination,
                "surface": nodes[destination]["class_name"],
                "requested_visibility": "offscreen"
                if key in ("pickup_a", "place_b")
                else "in_view",
                "depends_on": deps,
            }
        )
    return result


def plan_random_mix(graph, config):
    """Sample event proportions and a legal ordering before any action is rendered."""
    rng = random.Random(config["seed"])
    nodes = {n["id"]: n for n in graph["nodes"]}
    pool = config["mixed_candidates"]
    openables = sorted(set(pool["cabinets"] + [pool["appliance"]]))
    portable = sorted(set(pool["portable"]))
    openables = [key for key in openables if "CAN_OPEN" in nodes[key]["properties"] and "CLOSED" in nodes[key]["states"]]
    portable = [key for key in portable if "GRABBABLE" in nodes[key]["properties"] and support_of(graph, key) is not None]
    counts = [n for n in (1, 2) if n <= len(portable) and 4 - n <= len(openables)]
    if not counts:
        raise ValueError("The candidate pool cannot supply five valid event pairs")
    picked = rng.sample(portable, rng.choice(counts))
    opened = rng.sample(openables, 4 - len(picked))
    switch = pool["switch"]
    tasks = {}

    def add(key, verb, target, deps, surface=None, offscreen=False):
        destination = config["surface_id"] if surface is None else surface
        actor = 2 if target == switch else 1
        tasks[key] = dict(task=key, verb=verb, object_id=target,
                          object=nodes[target]["class_name"],
                          event_family="state_change" if verb in STATE_EFFECTS else "movement",
                          event_action=STATE_EFFECTS[verb][2] if verb in STATE_EFFECTS else "move",
                          operation=verb, injector_character=actor, injector_node_id=actor + 1,
                          surface_id=destination, surface=nodes[destination]["class_name"],
                          requested_visibility="offscreen" if offscreen else "in_view",
                          depends_on=deps)

    add("switch_on", "SwitchOn", switch, [])
    opening_tasks = [f"open_{key}" for key in opened]
    for key in opened:
        add(f"open_{key}", "Open", key, ["switch_on"])
    for index, key in enumerate(picked):
        deps = ["switch_on"] if index == 0 else [f"place_{index - 1}"]
        if index == len(picked) - 1:
            deps += opening_tasks
        destination = pool["secondary_surface_id"] if index == len(picked) - 1 else config["surface_id"]
        add(f"pickup_{index}", "Grab", key, deps, destination, offscreen=index == 0)
        add(f"place_{index}", "PutBack", key, [f"pickup_{index}"], destination,
            offscreen=index == len(picked) - 1)
    add("switch_off", "SwitchOff", switch, ["switch_on", f"place_{len(picked) - 1}"] + opening_tasks)
    for key in opened:
        add(f"close_{key}", "Close", key, ["switch_off", f"open_{key}"])
    done, result, holding = set(), [], False
    while len(result) < len(tasks):
        eligible = [event for key, event in tasks.items() if key not in done
                    and set(event["depends_on"]) <= done
                    and not (holding and event["verb"] in ("Open", "Grab"))]
        if not eligible:
            raise ValueError("No feasible event remains in the sampled program")
        event = rng.choice(eligible)
        result.append(event)
        done.add(event["task"])
        if event["verb"] == "Grab":
            holding = True
        elif event["verb"] == "PutBack":
            holding = False
    requested = config.get("offscreen_event_count", 2)
    if not isinstance(requested, int) or not 2 <= requested <= 7:
        raise ValueError("The offscreen event count must be between two and seven")
    remaining = requested - sum(e["requested_visibility"] == "offscreen" for e in result)
    # Closing tasks already follow visible open-state inspections.
    candidates = [e for e in result if e["verb"] == "Close"]
    rng.shuffle(candidates)
    candidates += [e for e in reversed(result) if e["verb"] == "Open"]
    if remaining > len(candidates):
        raise ValueError("The routine cannot supply the requested offscreen count")
    for event in candidates[:remaining]:
        event["requested_visibility"] = "offscreen"
    return result


def state_description(graph, event):
    node = next(n for n in graph["nodes"] if n["id"] == event["object_id"])
    verb = event["verb"]
    if verb in ("Open", "Close"):
        return "open" if "OPEN" in node["states"] else "closed"
    if verb in ("SwitchOn", "SwitchOff"):
        return "on" if "ON" in node["states"] else "off"
    if held_by(graph, event["object_id"], event.get("injector_node_id", 2)):
        return "in the woman’s hand"
    support = support_of(graph, event["object_id"])
    return (
        "on the " + label(support["class_name"])
        if support
        else "at an unverified location"
    )


def check_effect(before, after, event):
    key, verb = event["object_id"], event["verb"]
    actor = event.get("injector_node_id", 2)
    if verb in STATE_EFFECTS:
        if (verb in ('SwitchOn','SwitchOff')
                and event.get('state_verification')=='native_recorded_animation'):
            return
        old, new, _ = STATE_EFFECTS[verb]
        a = next(n for n in before["nodes"] if n["id"] == key)
        b = next(n for n in after["nodes"] if n["id"] == key)
        if old not in a["states"] or new not in b["states"] or old in b["states"]:
            raise ValueError(
                f"The requested state transition did not occur: {verb} {key}"
            )
    elif verb == "Grab":
        if held_by(before, key, actor) or not held_by(after, key, actor):
            raise ValueError("The injector did not pick up the target")
    elif verb == "PutBack":
        if (
            not held_by(before, key, actor)
            or held_by(after, key, actor)
            or not on_surface(after, key, event["surface_id"])
        ):
            raise ValueError(
                "The injector did not put the target on the requested surface"
            )
    else:
        raise ValueError("Unsupported manipulation")


def check_offscreen_count(count, config):
    required = config.get("offscreen_event_count", 2)
    if count < required:
        raise ValueError(f"Expected at least {required} fully offscreen events, got {count}")
    if "offscreen_event_count" in config and count > required + 1:
        raise ValueError(f"Too many unintended offscreen events: {count}")


def check_event_spacing(events, config, program=None):
    gaps = [b["start"] - a["start"] for a, b in zip(events, events[1:])]
    tolerance = config["event_interval_tolerance_seconds"]
    policy = config.get("timing_policy")
    if policy == "frozen_relative_intervals" or (policy == "planned_start_windows_without_retiming" and program):
        if not program or len(program.get("events", [])) != len(events):
            raise ValueError("The frozen timing program is missing or incomplete")
        expected = [
            event["planned_gap_seconds"] for event in program["events"][1:]
        ]
        tolerances = [
            event.get("start_tolerance_seconds", tolerance)
            for event in program["events"][1:]
        ]
    else:
        expected = [config["event_interval_seconds"]] * len(gaps)
        tolerances = [tolerance] * len(gaps)
    if any(
        abs(gap - center) > allowed + 1e-6
        for gap, center, allowed in zip(gaps, expected, tolerances)
    ):
        raise ValueError(f"Irregular event spacing: {gaps}")
    return gaps


def check_recorded_event_timing(event, row, actor, verb):
    """Match native manipulations within their containing joint render call."""
    def same(left, right):
        return math.isclose(left, right, rel_tol=0, abs_tol=1e-6)

    if 'joint_action_start' in event or 'joint_action_end' in event:
        if not all(key in event for key in ('joint_action_start', 'joint_action_end')):
            raise ValueError('Incomplete joint action timing')
        if not same(row['start'], event['joint_action_start']) or not same(row['end'], event['joint_action_end']):
            raise ValueError('Joint action timing does not match its recorded action')
        spans = [span for span in row.get('native_action_spans', {}).get(str(actor), [])
                 if span['verb'] == verb.upper()]
        if not spans or not same(spans[-1]['start'], event['start']) or not same(spans[-1]['end'], event['end']):
            raise ValueError('Event timing does not match its native manipulation')
        if not row['start'] - 1e-6 <= event['start'] < event['end'] <= row['end'] + 1e-6:
            raise ValueError('Native manipulation lies outside its recorded action')
    elif not same(row['start'], event['start']) or not same(row['end'], event['end']):
        raise ValueError('Event timing does not match its recorded action')


def audit_mixed(root):
    from .recording import validate_first_person_pose
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    events = json.loads((root / "events.json").read_text())
    program = None
    if (root / "event_program.json").exists():
        from .planning import audit_execution
        program = json.loads((root / "event_program.json").read_text())
        audit_execution(root)
    manifest = [json.loads(l) for l in (root / "frame_manifest.jsonl").open()]
    fps = config["render"]["fps"]
    check_duration(len(manifest) / fps, config)
    if len(events) != config["event_count"]:
        raise ValueError("The episode must contain ten completed events")
    if (
        len({e["object_id"] for e in events}) < config.get("minimum_distinct_objects", 5)
        or len({e["verb"] for e in events}) < config.get("minimum_distinct_operations", 6)
    ):
        raise ValueError(
            "The episode does not meet the object/action diversity requirement"
        )
    actions = json.loads((root / "actions.json").read_text())
    expected_actors=set(config.get('active_operator_characters',(1,2)))
    check_roles(actions, allowed_characters=(0,*sorted(expected_actors)))
    manipulations = []
    for row in actions:
        for command in row["action"].split("|"):
            match = re.fullmatch(r"<char(\d+)> \[(\w+)\].*", command.strip())
            if match[2] in (*STATE_EFFECTS, "Grab", "PutBack"):
                manipulations.append((row, int(match[1]), match[2]))
    if len(manipulations) != len(events):
        raise ValueError("Every recorded manipulation must have exactly one event")
    if {actor for _, actor, _ in manipulations} != expected_actors:
        raise ValueError("Recorded event actors differ from the room operator plan")
    for event, (row, actor, verb) in zip(events, manipulations):
        if (actor, verb) != (event["injector_character"], event["verb"]):
            raise ValueError("Event actor/action does not match the recorded script")
        if row["target_id"] != event["object_id"]:
            raise ValueError("Recorded target does not match the event object")
        check_recorded_event_timing(event, row, actor, verb)
        if (
            not 0
            <= event["before_time"]
            <= event["start"]
            < event["end"]
            <= event["discovery_time"]
            <= len(manifest) / fps
        ):
            raise ValueError("Event observations have invalid time bounds")
    rows = []
    margins = []
    camera_offsets = []
    first_person = config.get('viewpoint') == 'first_person'
    if first_person and config.get('camera_continuity', {}).get('viewpoint') != 'first_person':
        raise ValueError('First-person configuration does not match its camera controller')
    for index, row in enumerate(manifest):
        if row["frame"] != index or row.get("color_collision_ids"):
            raise ValueError("Invalid frame order or ambiguous target mask")
        monitored_ids=config.get('monitor_ids',[1,2,3])
        if any(row["monitored_collisions"].get(str(actor)) for actor in monitored_ids):
            raise ValueError("Character masks must have unique instance colors")
        seg = np.asarray(Image.open(root / row["mask"]).convert("RGB"))
        ys = np.nonzero(target_mask(seg, row["monitored_colors"]["1"]))[0]
        if first_person:
            camera_offsets.append(validate_first_person_pose(row['camera_pose']))
        else:
            if not len(ys):
                raise ValueError("The observer is missing from the third-person view")
            margins.append(int(ys.min()))
        rows.append(
            {
                "frame": index,
                "time": index / fps,
                "target_id": row["target_id"],
                "target_pixels": visible_pixels(seg, row["color"]),
                "actor_pixels": {
                    key: (visible_pixels(seg, row["monitored_colors"][str(key)])
                          if str(key) in row["monitored_colors"] else 0)
                    for key in sorted(actor+1 for actor in expected_actors)
                },
            }
        )
    reports = []
    for e in events:
        stages = {
            phase: json.loads(
                (root / "evidence" / f"{e['stage_keys'][phase]}.json").read_text()
            )
            for phase in ("before", "after")
        }
        target_node=next(node for node in stages['before']['graph']['nodes']
                         if node['id']==e['object_id'])
        target_minimum=target_visibility_threshold(config,target_node)
        check_effect(stages["before"]["graph"], stages["after"]["graph"], e)
        for phase, data in stages.items():
            if (
                data["color_collision_ids"]
                or data["visible_pixels"] < target_minimum
            ):
                raise ValueError(
                    f"The {phase} target is not clearly visible: {e['task']}"
                )
        first_frame, stop_frame = frame_interval(e['start'], e['end'], fps)
        active = [r for r in rows if first_frame <= r['frame'] < stop_frame]
        if not active or any(r["target_id"] != e["object_id"] for r in active):
            raise ValueError(
                "Event frames must reference the actual manipulated object"
            )
        for row in active:
            row["injector_pixels"] = row["actor_pixels"][e["injector_node_id"]]
        offscreen = all(
            r["target_pixels"] == 0 and r["injector_pixels"] == 0 for r in active
        )
        if e["requested_visibility"] == "offscreen" and not offscreen:
            raise ValueError(f"The offscreen action became visible: {e['task']}")
        seen = sum(
            r["target_pixels"] >= target_minimum and r["injector_pixels"] >= 40 for r in active
        )
        e["visibility"] = (
            "offscreen"
            if offscreen
            else "fully_observed"
            if seen == len(active)
            else "partly_observed"
        )
        if e['verb'] in ('SwitchOn','SwitchOff') and e.get('state_verification')=='native_recorded_animation':
            old,new,_=STATE_EFFECTS[e['verb']]
            e['before_state'],e['after_state']=old.lower(),new.lower()
        else:
            e["before_state"] = state_description(stages["before"]["graph"], e)
            e["after_state"] = state_description(stages["after"]["graph"], e)
        if e["before_state"] == e["after_state"]:
            raise ValueError("A counted event must change the target state or support")
        reports.append(
            {
                "event_index": e["event_index"],
                "object_id": e["object_id"],
                "visibility": e["visibility"],
                "injection_frames": len(active),
                "observed_action_frames": seen,
            }
        )
    gaps = check_event_spacing(events, config, program)
    offscreen_count = sum(e["visibility"] == "offscreen" for e in events)
    required_offscreen = config.get("offscreen_event_count", 2)
    check_offscreen_count(offscreen_count, config)
    if margins and min(margins) == 0:
        raise ValueError("The observer silhouette touches the upper image border")
    report = {
        "viewpoint": 'first_person' if first_person else 'third_person',
        "minimum_headroom_pixels": min(margins) if margins else None,
        "maximum_camera_observer_horizontal_offset_m": max(camera_offsets) if camera_offsets else None,
        "frames": len(rows),
        "fps": fps,
        "duration_sec": len(rows) / fps,
        "event_count": len(events),
        "distinct_objects": len({e["object_id"] for e in events}),
        "offscreen_event_count": offscreen_count,
        "required_offscreen_event_count": required_offscreen,
        "operations": dict(Counter(e["verb"] for e in events)),
        "start_gaps_seconds": gaps,
        "observer_manipulations": 0,
        "operating_characters": sorted(expected_actors),
        "events": reports,
        **camera_audit(root),
        "motion": audit_motion(root),
        "camera_continuity": audit_continuity(root),
    }
    (root / "events.json").write_text(json.dumps(events, indent=2))
    (root / "visibility.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "observer_validation.json").write_text(json.dumps(report, indent=2))
    (root / "event_schedule.json").write_text(
        json.dumps(
            {
                "event_count": len(events),
                "complete": True,
                "counting_unit": "one successful state or support transition",
                "start_gaps_seconds": gaps,
                "events": events,
            },
            indent=2,
        )
    )
    return report


def build_mixed_questions(episode, events, seed=0):
    rng = random.Random(seed)
    questions = []
    for i, e in enumerate(events):
        name = e.get("object_description", label(e["object"]))
        start, end = e["before_time"], e["discovery_time"]
        before, after = e["before_state"], e["after_state"]
        prefix = events[: i + 1]

        def add(
            kind,
            subtype,
            question,
            answer,
            distractors,
            spans,
            unknown=False,
            query=None,
        ):
            options = list(dict.fromkeys([answer, *distractors]))[:4]
            if len(options) != 4:
                raise ValueError("Four distinct choices are required")
            rng.shuffle(options)
            questions.append(
                {
                    "id": f"{episode}_q{len(questions):03d}",
                    "episode_id": episode,
                    "event_index": i,
                    "query_time": end if query is None else query,
                    "question_type": kind,
                    "question_subtype": subtype,
                    "question": question,
                    "options": options,
                    "answer_index": options.index(answer),
                    "evidence_spans": spans,
                    "diagnostics": {
                        "epistemic_status": "uncertain" if unknown else "known",
                        "uncertainty_sources": ["offscreen_change"] if unknown else [],
                    },
                    "video_evidence": "Use only the observer camera up to the query time.",
                    "change_intensity": "high",
                }
            )

        alternatives = (
            (
                ["open", "closed", "broken", "missing"]
                if e["verb"] in ("Open", "Close")
                else ["on", "off", "unplugged", "missing"]
            )
            if e["verb"] in STATE_EFFECTS
            else [
                before,
                after,
                "in the woman’s hand",
                "on the floor",
                "inside a closed container",
                "outside the room",
            ]
        )
        add(
            "factual_retrieval",
            "previous_observation",
            f"At the preceding inspection, what was the state or location of the {name}?",
            before,
            [x for x in alternatives if x != before],
            [[max(0, start - 0.5), start]],
        )
        if e["visibility"] == "offscreen":
            blind = e.get("hidden_time", e["end"])
            if e["verb"] in STATE_EFFECTS:
                options = (
                    ["It is open.", "It is closed.", "It has been removed."]
                    if e["verb"] in ("Open", "Close")
                    else ["It is on.", "It is off.", "It has been removed."]
                )
                add(
                    "current_state",
                    "unobserved_state",
                    f"At the end of this clip, what state is the {name} in?",
                    "Its current state cannot be determined from this view.",
                    options,
                    [[max(0, start - 0.5), start], [max(0, blind - 0.5), blind]],
                    True,
                    blind,
                )
            else:
                add(
                    "current_state",
                    "unobserved_state",
                    f"At the end of this clip, where is the {name} now?",
                    "Its current location cannot be determined from this view.",
                    [f"It is {value}." for value in dict.fromkeys(
                        [before, after, "in the woman’s hand", "on the floor", "outside the room"]
                    )][:3],
                    [[max(0, start - 0.5), start], [max(0, blind - 0.5), blind]],
                    True,
                    blind,
                )
        else:
            add(
                "current_state",
                "current_observation",
                f"At the end of this clip, what is the state or location of the {name}?",
                after,
                [x for x in alternatives if x != after],
                [[max(0, end - 0.5), end]],
            )
        add(
            "state_change",
            "observed_difference",
            f"What changed between the two most recent inspections of the {name}?",
            f"It changed from {before} to {after}.",
            [
                f"It changed from {after} to {before}.",
                "It stayed unchanged.",
                "Its color changed.",
            ],
            [[max(0, start - 0.5), start], [max(0, end - 0.5), end]],
        )
        other_names = [
            label(x["object"]) for x in events if x["object_id"] != e["object_id"]
        ]
        add(
            "object_tracking",
            "changed_object",
            f"Which object changed from {before} to {after} in the latest comparison?",
            name,
            list(dict.fromkeys(other_names)),
            [[max(0, start - 0.5), start], [max(0, end - 0.5), end]],
        )
        add(
            "temporal_reasoning",
            "state_order",
            f"Which state or location of the {name} was seen first in the latest comparison?",
            before,
            [after, "Both were first seen at the same time.", "Neither was visible."],
            [[max(0, start - 0.5), start], [max(0, end - 0.5), end]],
        )
        same = [x for x in prefix if x["object_id"] == e["object_id"]]
        n = len(same)
        spans = [[max(0, x["before_time"] - 0.5), x["discovery_time"]] for x in same]
        add(
            "history_aggregation",
            "confirmed_changes",
            f"How many changes to the {name} have been confirmed by successive inspections so far?",
            str(n),
            [str(k) for k in range(5) if k != n],
            spans,
        )
    return questions
