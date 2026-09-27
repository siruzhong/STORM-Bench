"""Pure validation gates for plans and two-scene generation batches."""
from __future__ import annotations

from ..domain.config import BenchmarkConfig
from ..domain.contracts import (
    EpisodePlan,
    ValidationCheck,
    ValidationReport,
)


def _check(name, passed, value, limit, detail="") -> ValidationCheck:
    return ValidationCheck(
        name=name, passed=bool(passed), value=value, limit=limit,
        detail=detail,
    )


def _shortest_delta(source: float, target: float) -> float:
    return (target - source + 180.0) % 360.0 - 180.0


def validate_plan(
    plan: EpisodePlan, config: BenchmarkConfig,
) -> ValidationReport:
    recipe = plan.recipe
    events = plan.event_program.events
    frames = plan.trajectory.frames
    event_times = [item.trigger_time_s for item in events]
    gaps = [right - left for left, right in zip(event_times, event_times[1:])]
    yaw_rates = [
        _shortest_delta(left.yaw, right.yaw) * recipe.fps
        for left, right in zip(frames, frames[1:])
    ]
    yaw_accels = [
        (right - left) * recipe.fps
        for left, right in zip(yaw_rates, yaw_rates[1:])
    ]
    expected_frames = int(round(recipe.duration_s * recipe.fps)) + 1
    checks = (
        _check(
            "config_digest", plan.config_digest == config.digest,
            plan.config_digest, config.digest),
        _check(
            "duration", config.timing.duration_range_s[0]
            <= recipe.duration_s <= config.timing.duration_range_s[1],
            recipe.duration_s, config.timing.duration_range_s),
        _check(
            "event_count", len(events) == config.timing.event_count,
            len(events), config.timing.event_count),
        _check(
            "first_event", bool(event_times)
            and event_times[0] <= config.timing.first_event_max_s,
            event_times[0] if event_times else None,
            config.timing.first_event_max_s),
        _check(
            "last_event", bool(event_times)
            and event_times[-1] >= config.timing.last_event_min_s,
            event_times[-1] if event_times else None,
            config.timing.last_event_min_s),
        _check(
            "event_gap", not gaps
            or max(gaps) <= config.timing.max_event_gap_s,
            max(gaps, default=0.0), config.timing.max_event_gap_s),
        _check(
            "travel_distance",
            plan.mobility.metrics.distance_m
            >= config.quality.min_travel_distance_m,
            plan.mobility.metrics.distance_m,
            config.quality.min_travel_distance_m),
        _check(
            "segment_speed",
            plan.mobility.metrics.max_segment_speed_mps
            <= config.motion.max_segment_speed_mps,
            plan.mobility.metrics.max_segment_speed_mps,
            config.motion.max_segment_speed_mps),
        _check(
            "stationary_fraction",
            plan.mobility.metrics.stationary_fraction
            <= config.quality.max_stationary_fraction,
            plan.mobility.metrics.stationary_fraction,
            config.quality.max_stationary_fraction),
        _check(
            "frame_count", len(frames) == expected_frames,
            len(frames), expected_frames),
        _check(
            "camera_yaw_rate", max((abs(v) for v in yaw_rates), default=0.0)
            <= config.camera.max_yaw_rate_dps,
            max((abs(v) for v in yaw_rates), default=0.0),
            config.camera.max_yaw_rate_dps),
        _check(
            "camera_yaw_acceleration",
            max((abs(v) for v in yaw_accels), default=0.0)
            <= config.camera.max_yaw_accel_dps2,
            max((abs(v) for v in yaw_accels), default=0.0),
            config.camera.max_yaw_accel_dps2),
    )
    accepted = all(item.passed for item in checks)
    return ValidationReport(
        stage="static",
        accepted=accepted,
        checks=checks,
        failure_code=None if accepted else "static_quality_gate_failed",
    )


def validate_batch(
    plans: tuple[EpisodePlan, ...],
    reports: tuple[ValidationReport, ...],
    expected_count: int,
) -> ValidationReport:
    scenes = [plan.recipe.scene for plan in plans]
    seeds = [plan.recipe.seed for plan in plans]
    config_digests = {plan.config_digest for plan in plans}
    programs = {plan.event_program.program_id for plan in plans}
    checks = (
        _check(
            "episode_count", len(plans) == expected_count,
            len(plans), expected_count),
        _check(
            "distinct_scenes", len(set(scenes)) == len(plans),
            scenes, f"{len(plans)} unique"),
        _check(
            "distinct_seeds", len(set(seeds)) == len(plans),
            seeds, f"{len(plans)} unique"),
        _check(
            "single_config", len(config_digests) == 1,
            sorted(config_digests), "1 unique"),
        _check(
            "distinct_event_programs", len(programs) == len(plans),
            sorted(programs), f"{len(plans)} unique"),
        _check(
            "all_episodes_accepted", len(reports) == len(plans)
            and all(report.accepted for report in reports),
            [report.accepted for report in reports], "all true"),
    )
    accepted = all(item.passed for item in checks)
    return ValidationReport(
        stage="batch",
        accepted=accepted,
        checks=checks,
        failure_code=None if accepted else "batch_acceptance_failed",
    )
