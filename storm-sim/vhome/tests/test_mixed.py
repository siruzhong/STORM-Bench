"""Check seeded routine planning, real state transitions, and QA prefix bounds."""

import copy

import pytest

from storm_virtualhome.mixed import (
    plan_routine,
    check_effect,
    build_mixed_questions,
    STATE_EFFECTS,
)
from storm_virtualhome.qa import TYPES


def scene():
    return {
        "nodes": [
            {
                "id": 266,
                "class_name": "washingsponge",
                "properties": ["GRABBABLE"],
                "states": [],
            },
            {
                "id": 267,
                "class_name": "dishwashingliquid",
                "properties": ["GRABBABLE"],
                "states": [],
            },
            {
                "id": 235,
                "class_name": "kitchencabinet",
                "properties": ["CAN_OPEN"],
                "states": ["CLOSED"],
            },
            {
                "id": 236,
                "class_name": "kitchencabinet",
                "properties": ["CAN_OPEN"],
                "states": ["CLOSED"],
            },
            {
                "id": 313,
                "class_name": "microwave",
                "properties": ["CAN_OPEN"],
                "states": ["CLOSED", "OFF"],
            },
            {
                "id": 264,
                "class_name": "tv",
                "properties": ["HAS_SWITCH"],
                "states": ["OFF"],
            },
            {
                "id": 238,
                "class_name": "kitchencounter",
                "properties": ["SURFACES"],
                "states": [],
            },
            {
                "id": 231,
                "class_name": "kitchentable",
                "properties": ["SURFACES"],
                "states": [],
            },
        ],
        "edges": [
            {"from_id": k, "to_id": 238, "relation_type": "ON"} for k in (266, 267)
        ],
    }


def config(seed=42):
    return {
        "seed": seed,
        "surface_id": 238,
        "mixed_candidates": {
            "portable": [266, 267],
            "cabinets": [235, 236],
            "appliance": 313,
            "switch": 264,
            "secondary_surface_id": 231,
        },
    }


def test_random_routines_preserve_dependencies_and_diversity():
    sequences = set()
    for seed in range(100):
        plan = plan_routine(scene(), config(seed))
        assert plan == plan_routine(scene(), config(seed))
        assert plan[0]["task"] == "switch_on" and plan[7]["task"] == "switch_off"
        assert {e["injector_character"] for e in plan} == {1, 2}
        assert all(e["injector_node_id"] == e["injector_character"] + 1 for e in plan)
        assert len(plan) == 10 and len({e["object_id"] for e in plan}) == 5
        assert len({e["verb"] for e in plan}) == 6
        positions = {e["task"]: i for i, e in enumerate(plan)}
        for i, e in enumerate(plan):
            assert all(positions[d] < i for d in e["depends_on"])
        assert positions["place_a"] < positions["pickup_b"]
        assert {e["surface_id"] for e in plan if e["verb"] == "PutBack"} == {238, 231}
        assert positions["appliance_close"] - positions["appliance_open"] >= 3
        assert sum(e["requested_visibility"] == "offscreen" for e in plan) == 2
        sequences.add(tuple((e["task"], e["object_id"]) for e in plan))
    assert len(sequences) > 10


def test_state_graph_changes_cannot_be_counted_as_noops():
    before = scene()
    after = copy.deepcopy(before)
    event = {"object_id": 264, "verb": "SwitchOn"}
    with pytest.raises(ValueError, match="did not occur"):
        check_effect(before, after, event)
    next(n for n in after["nodes"] if n["id"] == 264)["states"] = ["ON"]
    check_effect(before, after, event)
    with pytest.raises(ValueError):
        check_effect(after, after, event)
    pickup = {"object_id": 266, "verb": "Grab"}
    with pytest.raises(ValueError):
        check_effect(before, after, pickup)
    after["edges"].append({"from_id": 2, "to_id": 266, "relation_type": "HOLDS_RH"})
    check_effect(before, after, pickup)


def test_recorded_switch_animation_can_cover_stale_unity_graph_only():
    before = scene()
    after = copy.deepcopy(before)
    switch = {
        "object_id": 264,
        "verb": "SwitchOn",
        "state_verification": "native_recorded_animation",
    }
    check_effect(before, after, switch)
    with pytest.raises(ValueError, match="did not occur"):
        check_effect(before, after, dict(switch, state_verification="graph_state"))
    with pytest.raises(ValueError):
        check_effect(
            before,
            after,
            {
                "object_id": 266,
                "verb": "Grab",
                "state_verification": "native_recorded_animation",
            },
        )


@pytest.mark.parametrize("offscreen_count", [2, 5])
@pytest.mark.parametrize("locations", [("kitchen counter", "kitchen table"), ("desk", "nightstand")])
def test_questions_cover_six_types_and_keep_blind_queries_before_discovery(offscreen_count, locations):
    options = config()
    if offscreen_count == 5:
        options.update(randomize_event_mix=True, offscreen_event_count=5)
    events = plan_routine(scene(), options)
    for i, e in enumerate(events):
        e.update(
            event_index=i,
            before_time=i * 6 + 0.5,
            start=i * 6 + 1,
            end=i * 6 + 2,
            discovery_time=i * 6 + 3,
            hidden_time=i * 6 + 2,
            visibility="offscreen"
            if e["requested_visibility"] == "offscreen"
            else "partly_observed",
        )
        if e["verb"] in STATE_EFFECTS:
            a, b, _ = STATE_EFFECTS[e["verb"]]
            e.update(before_state=a.lower(), after_state=b.lower())
        else:
            a, b = f"on the {locations[0]}", "in the woman’s hand"
            if e["verb"] == "PutBack" and e["surface_id"] == 231:
                a = f"on the {locations[1]}"
            e.update(
                before_state=a if e["verb"] == "Grab" else b,
                after_state=b if e["verb"] == "Grab" else a,
            )
    # A concurrent operator can finish before the observer returns to inspect it.
    switch_off = next(e for e in events if e["verb"] == "SwitchOff")
    switch_off["before_time"] = events[0]["discovery_time"]
    switch_off["discovery_time"] = events[-1]["discovery_time"] + 2
    if offscreen_count == 5:
        for event in events:
            if event["verb"] == "Close":
                opened = next(item for item in events if item["verb"] == "Open" and item["object_id"] == event["object_id"])
                event["before_time"] = opened["discovery_time"]
                event["discovery_time"] = 60.0
    questions = build_mixed_questions("mixed", events)
    if locations[0] == "desk":
        assert all("kitchen counter" not in option and "kitchen table" not in option
                   for q in questions for option in q["options"])
    assert len(questions) == 60 and {q["question_type"] for q in questions} == set(
        TYPES
    )
    assert len({q["id"] for q in questions}) == 60
    assert sum(q["diagnostics"]["epistemic_status"] == "uncertain" for q in questions) == offscreen_count
    for q in questions:
        assert len(q["options"]) == len(set(q["options"])) == 4
        assert all(0 <= a <= b <= q["query_time"] for a, b in q["evidence_spans"])
        if q["diagnostics"]["epistemic_status"] == "uncertain":
            e = events[q["event_index"]]
            assert q["query_time"] == e["hidden_time"] < e["discovery_time"]


def test_placement_requires_the_planned_surface():
    before = scene()
    before["edges"] = [e for e in before["edges"] if e["from_id"] != 266]
    before["edges"].append({"from_id": 2, "to_id": 266, "relation_type": "HOLDS_RH"})
    after = copy.deepcopy(before)
    after["edges"] = [e for e in after["edges"] if e["relation_type"] != "HOLDS_RH"]
    after["edges"].append({"from_id": 266, "to_id": 231, "relation_type": "ON"})
    event = {"object_id": 266, "verb": "PutBack", "surface_id": 231}
    check_effect(before, after, event)
    with pytest.raises(ValueError, match="requested surface"):
        check_effect(before, after, dict(event, surface_id=238))


def test_multiple_actors_do_not_allow_observer_manipulation():
    from storm_virtualhome.observer_validation import check_roles

    check_roles(
        [{"action": "<char0> [Walk] <book> (268) | <char2> [SwitchOn] <tv> (264)"}],
        (0, 1, 2),
    )
    with pytest.raises(ValueError, match="observer"):
        check_roles([{"action": "<char0> [SwitchOn] <tv> (264)"}], (0, 1, 2))
    before = scene()
    after = copy.deepcopy(before)
    after["edges"].append({"from_id": 3, "to_id": 266, "relation_type": "HOLDS_RH"})
    with pytest.raises(ValueError):
        check_effect(
            before, after, {"object_id": 266, "verb": "Grab", "injector_node_id": 2}
        )


def test_native_event_excludes_implicit_approach_inside_joint_action():
    from storm_virtualhome.mixed import check_recorded_event_timing
    event = dict(start=6.9, end=8.5, joint_action_start=6.15, joint_action_end=8.5)
    row = dict(start=6.15, end=8.5, native_action_spans={
        '1': [dict(verb='SWITCHON', start=6.9, end=8.5)]})
    check_recorded_event_timing(event, row, 1, 'SwitchOn')
    with pytest.raises(ValueError, match='native manipulation'):
        check_recorded_event_timing(dict(event, start=6.15), row, 1, 'SwitchOn')
    with pytest.raises(ValueError, match='native manipulation'):
        check_recorded_event_timing(event, row, 2, 'SwitchOn')
    with pytest.raises(ValueError, match='Joint action'):
        check_recorded_event_timing(dict(event, joint_action_end=9), row, 1, 'SwitchOn')


def test_legacy_event_still_matches_entire_action():
    from storm_virtualhome.mixed import check_recorded_event_timing
    check_recorded_event_timing(dict(start=1, end=2), dict(start=1, end=2), 1, 'Open')
    with pytest.raises(ValueError):
        check_recorded_event_timing(dict(start=1.5, end=2), dict(start=1, end=2), 1, 'Open')
