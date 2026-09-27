"""Verify room identity, deterministic planning, and unsupported-room rejection."""
import copy

import pytest

from storm_virtualhome.scene import room_id_of
from storm_virtualhome.room_profiles import build_profile
from storm_virtualhome.planning import create_program


def room_graph():
    nodes, edges = [], []
    def add(key, name, props, xyz, size=(0.3, 0.6, 0.3), states=(), room=10, category="Objects"):
        nodes.append(dict(id=key, class_name=name, category=category, properties=props,
                          states=list(states), bounding_box=dict(center=list(xyz), size=list(size))))
        if key != room:
            edges.append(dict(from_id=key, to_id=room, relation_type="INSIDE"))
    for key, x in ((10, 0), (20, 20)):
        add(key, "bedroom", [], (x, 1.25, 0), (8, 3, 8), room=key, category="Rooms")
        add(key + 1, "floor", ["SURFACES"], (x, 0, 0), (8, 0, 8), room=key)
    add(30, "cabinet", ["CAN_OPEN", "SURFACES"], (-2, .5, -2), states=["CLOSED"])
    add(31, "closet", ["CAN_OPEN"], (-2, 1, 0), states=["CLOSED"])
    add(32, "desk", ["CAN_OPEN", "SURFACES"], (-2, .5, 2), states=["CLOSED"])
    add(33, "tv", ["HAS_SWITCH"], (2, 1, 0), states=["OFF"])
    add(34, "coffeetable", ["SURFACES"], (0, .4, 0))
    add(37, "bookshelf", ["SURFACES"], (0, 1, 2))
    for key, name in ((35, "mug"), (36, "book")):
        add(key, name, ["GRABBABLE"], (-2, 1, -2), (.1, .1, .1))
        edges.append(dict(from_id=key, to_id=30, relation_type="ON"))
    return dict(nodes=nodes, edges=edges)


def template():
    return dict(seed=42, randomize_event_mix=True, event_count=10, offscreen_event_count=5,
                event_interval_seconds=6, duration_range_seconds=[60, 70], camera_continuity={},
                observer={}, injector={}, helper={})


def test_same_name_rooms_are_separate():
    graph = room_graph()
    assert room_id_of(graph, 30) == 10
    assert room_id_of(graph, 21) == 20
    with pytest.raises(ValueError, match="Insufficient"):
        build_profile(graph, template(), 0, 20, 42)


def test_profile_keeps_objects_and_spawn_bounds_local():
    graph = room_graph()
    original = copy.deepcopy(graph)
    for seed in range(10):
        config = build_profile(graph, template(), 2, 10, seed)
        assert config == build_profile(graph, template(), 2, 10, seed)
        program = create_program(graph, config)
        assert len(program["events"]) == 10
        assert sum(e["requested_visibility"] == "offscreen" for e in program["events"]) == 5
        assert all(room_id_of(graph, e["object_id"]) == 10 for e in program["events"])
        assert all(e["injector_character"] in (1, 2) for e in program["events"])
        assert config["camera_continuity"]["position_min"][0] == -3.65
        for role in ("observer", "injector", "helper"):
            assert all(abs(config[role]["position"][axis]) < 4 for axis in (0, 2))
    assert graph == original


def test_missing_capability_is_not_replaced_by_another_room():
    graph = room_graph()
    for edge in graph["edges"]:
        if edge["from_id"] == 33:
            edge["to_id"] = 20
    with pytest.raises(ValueError, match="switch=0"):
        build_profile(graph, template(), 0, 10, 42)


def test_wall_tile_bounds_do_not_fill_the_room():
    graph = room_graph()
    graph["nodes"].append(dict(id=99, class_name="wall", category="Walls", properties=[],
                               states=[], bounding_box=dict(center=[0, 1.25, 0], size=[8, 2.5, 8])))
    graph["edges"].append(dict(from_id=99, to_id=10, relation_type="INSIDE"))
    assert build_profile(graph, template(), 0, 10, 42)["room_id"] == 10


def test_recipe_signature_preserves_relevant_support_and_legacy_guards():
    from storm_virtualhome.planning import scene_signature, validate_program
    graph = room_graph()
    config = build_profile(graph, template(), 0, 10, 42)
    program = create_program(graph, config)
    changed = copy.deepcopy(graph)
    changed['edges'].append(dict(from_id=11, to_id=30, relation_type='ON'))
    assert scene_signature(graph) != scene_signature(changed)
    validate_program(program, changed, config)
    changed['edges'] = [e for e in changed['edges'] if not (e['from_id'] == 35 and e['relation_type'] == 'ON')]
    with pytest.raises(ValueError, match='scene'):
        validate_program(program, changed, config)


def test_snapshot_excludes_captures_and_runtime(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rollout_rooms", Path(__file__).parents[1] / "scripts/rollout_rooms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "project"
    project.mkdir()
    for name in ("src", "scripts", "configs", "docs", "unity_debug", "tests", "outputs", ".runtime"):
        (project / name).mkdir()
    for name in (".gitignore", "pyproject.toml", "README.md", "requirements-tested.txt", "FILES.sha256"):
        (project / name).write_text("test")
    (project / "outputs" / "large_capture").write_text("not source")
    target = project / "outputs" / "snapshot"
    module.snapshot_source(project, target)
    assert (target / "src").is_dir()
    assert not (target / "outputs").exists()
    assert not (target / ".runtime").exists()


def test_failed_snapshot_does_not_publish_partial_source(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rollout_rooms", Path(__file__).parents[1] / "scripts/rollout_rooms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    project = tmp_path / "project"
    project.mkdir()
    for name in ("src", "scripts", "configs", "docs", "tests"):
        (project / name).mkdir()
    for name in (".gitignore", "pyproject.toml", "README.md", "requirements-tested.txt", "FILES.sha256"):
        (project / name).write_text("test")
    target = tmp_path / "snapshot"
    with pytest.raises(FileNotFoundError):
        module.snapshot_source(project, target)
    assert not target.exists()
    assert not target.with_name('.snapshot.staging').exists()
