from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from tools.storm.benchgen.compiler.pipeline import compile_episode_plan
from tools.storm.benchgen.domain.config import load_config
from tools.storm.benchgen.domain.contracts import (
    ObservationStation,
    ExecutionTrace,
    SceneObject,
    SceneProfile,
    StationPath,
    ValidationReport,
    Vec3,
    stable_digest,
)
from tools.storm.benchgen.scene.profiler import scene_profile_from_dict
from tools.storm.benchgen.domain.failures import RetryScope, StageFailure
from tools.storm.benchgen.orchestrator.batch import BatchGenerator
from tools.storm.benchgen.validation.static import validate_batch, validate_plan


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchgen_v1.yaml"


def synthetic_profile(scene="FloorPlanSynthetic"):
    targets = []
    stations = []
    for index in range(5):
        angle = 2.0 * math.pi * index / 5.0
        target = Vec3(
            x=0.65 * math.cos(angle), y=0.9,
            z=0.65 * math.sin(angle),
        )
        targets.append(SceneObject(
            object_id=f"target-{index}",
            object_type="Mug",
            position=target,
            parent_receptacles=("counter-0",),
            pickupable=True,
            capabilities=("pickup", "fill"),
        ))
        for side in range(2):
            station_angle = angle + (0.24 if side else -0.24)
            position = Vec3(
                x=1.55 * math.cos(station_angle), y=0.9,
                z=1.55 * math.sin(station_angle),
            )
            yaw = math.degrees(math.atan2(
                target.x - position.x, target.z - position.z)) % 360.0
            stations.append(ObservationStation(
                station_id=f"station-{index}-{side}",
                position=position,
                target_yaw=yaw,
                hidden_yaw=(yaw + (58.0 if side else -58.0)) % 360.0,
                target_horizon=12.0,
                hidden_horizon=12.0,
                target_id=f"target-{index}",
                target_type="Mug",
                surface_id="counter-0",
                surface_type="CounterTop",
                distractor_ids=(f"distractor-{index}",),
                target_pixels=4000,
                target_bbox_width=64,
                target_bbox_height=64,
                target_center_distance=0.1,
                score=1000.0 - index * 10.0 - side,
            ))
    paths = []
    for left_index, left in enumerate(stations):
        for right in stations[left_index + 1:]:
            distance = math.hypot(
                right.position.x - left.position.x,
                right.position.z - left.position.z,
            )
            if 0.35 <= distance <= 3.5:
                paths.append(StationPath(
                    start_station_id=left.station_id,
                    end_station_id=right.station_id,
                    points=(left.position, right.position),
                    distance_m=distance,
                ))
    reachable = tuple(station.position for station in stations)
    return SceneProfile.create(
        scene=scene,
        simulator_build="synthetic-1",
        reachable=reachable,
        objects=tuple(targets),
        stations=tuple(stations),
        station_paths=tuple(paths),
    )


class BenchgenTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_config_and_digest_are_stable(self):
        other = load_config(CONFIG)
        self.assertEqual((640, 480), (
            self.config.render.width, self.config.render.height))
        self.assertEqual(10, self.config.timing.event_count)
        self.assertEqual(self.config.digest, other.digest)
        self.assertEqual(
            stable_digest(self.config), stable_digest(other))

    def test_compile_is_deterministic_and_passes_static_gates(self):
        profile = synthetic_profile()
        first = compile_episode_plan(profile, 17, self.config)
        second = compile_episode_plan(profile, 17, self.config)
        self.assertEqual(first.plan_id, second.plan_id)
        self.assertEqual(1801, len(first.trajectory.frames))
        self.assertEqual(10, len(first.event_program.events))
        self.assertTrue(validate_plan(first, self.config).accepted)

    def test_presence_events_are_state_safe_pairs(self):
        plan = compile_episode_plan(
            synthetic_profile(), 23, self.config)
        state = {}
        for event in plan.event_program.events:
            self.assertEqual("presence_change", event.event_family)
            self.assertEqual(event.event_type, event.event_action)
            self.assertIn(event.event_action, {"appear", "disappear"})
            previous = state.get(event.target_id, "present")
            if event.event_type == "disappear":
                self.assertEqual("present", previous)
                state[event.target_id] = "absent"
            else:
                self.assertEqual("absent", previous)
                state[event.target_id] = "present"
        self.assertTrue(all(value == "present" for value in state.values()))

    def test_scene_profile_round_trip_preserves_digest(self):
        profile = synthetic_profile()
        from tools.storm.benchgen.domain.contracts import primitive
        loaded = scene_profile_from_dict(primitive(profile))
        self.assertEqual(profile, loaded)

    def test_batch_gate_requires_distinct_scenes_and_programs(self):
        first = compile_episode_plan(
            synthetic_profile("FloorPlan1"), 101, self.config)
        second = compile_episode_plan(
            synthetic_profile("FloorPlan2"), 202, self.config)
        accepted = ValidationReport(
            stage="episode", accepted=True, checks=())
        report = validate_batch(
            (first, second), (accepted, accepted), expected_count=2)
        self.assertTrue(report.accepted)
        rejected = validate_batch(
            (first, first), (accepted, accepted), expected_count=2)
        self.assertFalse(rejected.accepted)

    def test_pure_planners_do_not_import_simulator_or_runtime(self):
        pure_roots = (
            ROOT / "src/tools/storm/benchgen/motion",
            ROOT / "src/tools/storm/benchgen/events",
            ROOT / "src/tools/storm/benchgen/camera",
            ROOT / "src/tools/storm/benchgen/compiler",
            ROOT / "src/tools/storm/benchgen/validation",
        )
        forbidden = ("ai2thor", "runtime.replay", "run_ego", "run_dataset")
        for directory in pure_roots:
            for path in directory.glob("*.py"):
                content = path.read_text(encoding="utf-8")
                for name in forbidden:
                    self.assertNotIn(name, content, f"{path} imports {name}")

    def test_batch_orchestrator_rotates_rejected_scenes(self):
        config = replace(
            self.config,
            batch=replace(
                self.config.batch,
                scenes=("BadScene", "GoodScene1", "GoodScene2"),
                max_attempts_per_scene=1,
            ),
        )

        class Profiler:
            def profile(self, scene, unused_config):
                if scene == "BadScene":
                    raise StageFailure(
                        stage="scene_profile", code="bad_scene",
                        retry_scope=RetryScope.SCENE,
                        detail="synthetic rejection",
                    )
                return synthetic_profile(scene)

        class Replay:
            def preflight(self, plan, unused_config):
                trace = ExecutionTrace(
                    plan_id=plan.plan_id, accepted=True, events=())
                return ValidationReport(
                    stage="rendered_preflight", accepted=True, checks=()), trace

            def capture(self, plan, unused_config, output_dir):
                return ExecutionTrace(
                    plan_id=plan.plan_id, accepted=True, events=())

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            result = BatchGenerator(
                config=config,
                profiler=Profiler(),
                replay=Replay(),
                output_root=temporary,
            ).run(seed=11)
            self.assertTrue(result.batch_report.accepted)
            self.assertEqual(2, len(result.episode_dirs))
            self.assertEqual(
                {"GoodScene1", "GoodScene2"},
                {path.name.split("_")[1] for path in result.episode_dirs},
            )


if __name__ == "__main__":
    unittest.main()
