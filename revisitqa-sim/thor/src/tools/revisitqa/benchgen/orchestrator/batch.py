"""Generate and jointly accept a configurable distinct-scene batch."""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from ..artifacts.store import ArtifactStore
from ..compiler.pipeline import compile_episode_plan
from ..domain.config import BenchmarkConfig
from ..domain.contracts import (
    EpisodePlan,
    ValidationCheck,
    ValidationReport,
    stable_digest,
)
from ..domain.failures import RetryScope, StageFailure
from ..domain.ports import ReplayPort, SceneProfilerPort
from ..scene.profiler import scene_profile_from_dict
from ..validation.static import validate_batch, validate_plan


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchRunResult:
    run_dir: Path
    episode_dirs: tuple[Path, ...]
    batch_report: ValidationReport


class BatchGenerator:
    def __init__(
        self,
        config: BenchmarkConfig,
        profiler: SceneProfilerPort,
        replay: ReplayPort,
        output_root: str | Path,
    ) -> None:
        self.config = config
        self.profiler = profiler
        self.replay = replay
        self.store = ArtifactStore(output_root)

    def _profile_cache_key(self, scene: str) -> str:
        return stable_digest({
            "profile_schema": 2,
            "scene": scene,
            "render": self.config.render,
            "station": self.config.station,
            "station_quality": {
                "min_target_pixels": self.config.quality.min_target_pixels,
                "min_target_dimension": self.config.quality.min_target_dimension,
                "min_edge_margin": self.config.quality.min_edge_margin,
                "max_center_distance": self.config.quality.max_center_distance,
                "distractor_min_pixels": self.config.quality.distractor_min_pixels,
                "distractor_min_objects": self.config.quality.distractor_min_objects,
            },
            "motion_edges": {
                "min": self.config.motion.min_station_step_m,
                "max": self.config.motion.max_station_step_m,
            },
        }, 20)

    def _profile(self, scene: str):
        cache_dir = self.store.root / "_profile_cache"
        cache_path = cache_dir / f"{scene}_{self._profile_cache_key(scene)}.json"
        if self.config.runtime.profile_cache and cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                profile = scene_profile_from_dict(json.load(handle))
            LOGGER.info(
                "loaded scene profile %s (%d stations, %d edges)",
                profile.profile_id, len(profile.stations),
                len(profile.station_paths),
            )
            return profile
        profile = self.profiler.profile(scene, self.config)
        if self.config.runtime.profile_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.store.write_json(cache_path, profile)
        LOGGER.info(
            "built scene profile %s (%d stations, %d edges)",
            profile.profile_id, len(profile.stations),
            len(profile.station_paths),
        )
        return profile

    def _episode_report(
        self,
        static: ValidationReport,
        rendered: ValidationReport,
        capture_accepted: bool,
    ) -> ValidationReport:
        capture = ValidationCheck(
            name="capture_replay", passed=capture_accepted,
            value=capture_accepted, limit=True,
        )
        checks = (*static.checks, *rendered.checks, capture)
        accepted = all(item.passed for item in checks)
        return ValidationReport(
            stage="episode",
            accepted=accepted,
            checks=checks,
            failure_code=None if accepted else "episode_acceptance_failed",
        )

    def _generate_episode(
        self,
        run_dir: Path,
        index: int,
        scene: str,
        base_seed: int,
    ) -> tuple[EpisodePlan, ValidationReport, Path]:
        profile = self._profile(scene)
        attempts = []
        attempts_dir = run_dir / "_attempts"
        attempts_dir.mkdir(exist_ok=True)
        for attempt in range(self.config.batch.max_attempts_per_scene):
            seed = int(base_seed + index * 100003 + attempt * 7919)
            LOGGER.info(
                "planning scene=%s seed=%d attempt=%d", scene, seed,
                attempt + 1,
            )
            try:
                plan = compile_episode_plan(profile, seed, self.config)
            except StageFailure as failure:
                attempts.append({
                    "seed": seed, "stage": failure.stage,
                    "code": failure.code, "detail": failure.detail,
                    "retry_scope": failure.retry_scope.value,
                })
                continue
            static_report = validate_plan(plan, self.config)
            if not static_report.accepted:
                attempts.append({
                    "seed": seed, "stage": "static",
                    "code": static_report.failure_code,
                    "failed_checks": [
                        item.name for item in static_report.checks
                        if not item.passed
                    ],
                })
                continue
            rendered_report, preflight_trace = self.replay.preflight(
                plan, self.config)
            if not rendered_report.accepted:
                attempts.append({
                    "seed": seed, "stage": "rendered_preflight",
                    "code": rendered_report.failure_code,
                    "failed_checks": [
                        item.name for item in rendered_report.checks
                        if not item.passed
                    ],
                })
                continue
            attempt_dir = attempts_dir / f"{scene}_seed{seed}"
            attempt_dir.mkdir(exist_ok=False)
            self.store.write_json(attempt_dir / "scene_profile.json", profile)
            self.store.write_json(attempt_dir / "episode_plan.json", plan)
            self.store.write_json(
                attempt_dir / "static_validation.json", static_report)
            self.store.write_json(
                attempt_dir / "preflight_validation.json", rendered_report)
            self.store.write_json(
                attempt_dir / "preflight_trace.json", preflight_trace)
            capture_trace = self.replay.capture(
                plan, self.config, attempt_dir)
            episode_report = self._episode_report(
                static_report, rendered_report, capture_trace.accepted)
            self.store.write_json(
                attempt_dir / "episode_validation.json", episode_report)
            if not episode_report.accepted:
                attempts.append({
                    "seed": seed, "stage": "capture",
                    "code": episode_report.failure_code,
                })
                continue
            final_dir = run_dir / f"{index:05d}_{scene}_seed{seed}"
            attempt_dir.rename(final_dir)
            self.store.write_json(
                run_dir / f"attempts_{index:05d}_{scene}.json", attempts)
            return plan, episode_report, final_dir
        self.store.write_json(
            run_dir / f"attempts_{index:05d}_{scene}.json", attempts)
        raise StageFailure(
            stage="orchestrator",
            code="scene_attempts_exhausted",
            retry_scope=RetryScope.SCENE,
            detail=(f"{scene} exhausted "
                    f"{self.config.batch.max_attempts_per_scene} attempts"),
            context={"attempts": attempts},
        )

    def run(self, seed: int) -> BatchRunResult:
        rng = random.Random(int(seed) ^ 0x2A1F00)
        scene_candidates = list(dict.fromkeys(self.config.batch.scenes))
        rng.shuffle(scene_candidates)
        run_dir = self.store.create_run()
        self.store.write_json(run_dir / "config.json", self.config)
        self.store.write_json(run_dir / "selection.json", {
            "seed": int(seed),
            "candidate_order": scene_candidates,
            "selected_scenes": [],
            "config_digest": self.config.digest,
        })
        plans = []
        reports = []
        episode_dirs = []
        scene_failures = []
        try:
            for candidate_index, scene in enumerate(scene_candidates):
                if len(plans) >= self.config.batch.episode_count:
                    break
                episode_index = len(plans)
                try:
                    plan, report, episode_dir = self._generate_episode(
                        run_dir,
                        episode_index,
                        scene,
                        int(seed) + candidate_index * 1000003,
                    )
                except StageFailure as failure:
                    scene_failures.append({
                        "scene": scene,
                        "stage": failure.stage,
                        "code": failure.code,
                        "detail": failure.detail,
                        "retry_scope": failure.retry_scope.value,
                    })
                    LOGGER.warning(
                        "rejecting scene=%s code=%s: %s",
                        scene, failure.code, failure.detail,
                    )
                    continue
                plans.append(plan)
                reports.append(report)
                episode_dirs.append(episode_dir)
            self.store.write_json(
                run_dir / "scene_failures.json", scene_failures)
            if len(plans) < self.config.batch.episode_count:
                raise StageFailure(
                    stage="orchestrator",
                    code="scene_pool_exhausted",
                    retry_scope=RetryScope.SCENE,
                    detail=(f"accepted {len(plans)} of "
                            f"{self.config.batch.episode_count} episodes"),
                    context={"scene_failures": scene_failures},
                )
            self.store.write_json(run_dir / "selection.json", {
                "seed": int(seed),
                "candidate_order": scene_candidates,
                "selected_scenes": [plan.recipe.scene for plan in plans],
                "config_digest": self.config.digest,
            })
            batch_report = validate_batch(
                tuple(plans), tuple(reports), self.config.batch.episode_count)
            self.store.write_json(
                run_dir / "batch_validation.json", batch_report)
            self.store.write_json(run_dir / "manifest.json", {
                "accepted": batch_report.accepted,
                "config_digest": self.config.digest,
                "episodes": [
                    {
                        "scene": plan.recipe.scene,
                        "seed": plan.recipe.seed,
                        "plan_id": plan.plan_id,
                        "event_program_id": plan.event_program.program_id,
                        "directory": path.name,
                    }
                    for plan, path in zip(plans, episode_dirs)
                ],
            })
            if not batch_report.accepted:
                raise StageFailure(
                    stage="orchestrator", code="batch_acceptance_failed",
                    retry_scope=RetryScope.EPISODE_SEED,
                    detail="generated episodes did not pass the batch gate",
                )
            return BatchRunResult(
                run_dir=run_dir,
                episode_dirs=tuple(episode_dirs),
                batch_report=batch_report,
            )
        except Exception as error:
            self.store.write_json(run_dir / "run_failure.json", {
                "type": type(error).__name__, "message": str(error),
            })
            raise
