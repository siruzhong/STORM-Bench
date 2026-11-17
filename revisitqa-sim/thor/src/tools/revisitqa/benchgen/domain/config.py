"""Typed loading and validation for the single benchgen configuration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import stable_digest


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain [min, max]")
    result = (float(value[0]), float(value[1]))
    if result[0] > result[1]:
        raise ValueError(f"{name} minimum exceeds maximum")
    return result


@dataclass(frozen=True)
class RenderConfig:
    width: int = 640
    height: int = 480
    field_of_view: float = 55.0
    fps: int = 30


@dataclass(frozen=True)
class TimingConfig:
    duration_s: float = 60.0
    duration_range_s: tuple[float, float] = (55.0, 65.0)
    event_count: int = 10
    first_event_max_s: float = 4.0
    last_event_min_s: float = 50.0
    max_event_gap_s: float = 8.0
    before_lead_s: float = 1.05
    hidden_lead_s: float = 0.10
    after_lag_s: float = 1.20
    event_jitter_s: float = 0.45


@dataclass(frozen=True)
class StationConfig:
    surface_types: tuple[str, ...]
    target_types: tuple[str, ...]
    min_targets: int = 2
    preferred_distance: float = 0.78
    min_distance: float = 0.5
    max_distance: float = 1.2
    horizons: tuple[float, ...] = (8.0, 12.0, 16.0, 20.0)
    distractor_offsets: tuple[float, ...] = (58.0, -58.0, 72.0, -72.0)
    max_targets: int = 18
    max_position_candidates: int = 16
    max_stations_per_target: int = 3
    max_total_stations: int = 18


@dataclass(frozen=True)
class MotionConfig:
    preferred_speed_mps: float = 0.42
    max_segment_speed_mps: float = 0.85
    max_station_step_m: float = 3.5
    min_station_step_m: float = 0.35
    max_station_reuse: int = 3
    route_candidate_count: int = 20000
    functional_radius_m: float = 1.8


@dataclass(frozen=True)
class CameraConfig:
    max_yaw_rate_dps: float = 210.0
    max_yaw_accel_dps2: float = 700.0
    turn_window_s: float = 0.62
    hidden_hold_s: float = 0.16
    focus_hold_s: float = 0.5


@dataclass(frozen=True)
class EventConfig:
    enabled_types: tuple[str, ...] = ("disappear", "appear")
    allow_simultaneous: bool = True
    max_simultaneous: int = 2


@dataclass(frozen=True)
class QualityConfig:
    min_target_pixels: int = 1400
    min_target_dimension: int = 28
    min_edge_margin: int = 4
    max_center_distance: float = 0.9
    distractor_min_pixels: int = 160
    distractor_min_objects: int = 1
    max_stationary_fraction: float = 0.18
    min_travel_distance_m: float = 8.0


@dataclass(frozen=True)
class BatchConfig:
    scenes: tuple[str, ...]
    episode_count: int = 2
    max_attempts_per_scene: int = 4


@dataclass(frozen=True)
class RuntimeConfig:
    force_action: bool = True
    highlight_alpha: float = 0.9
    video_codec: str = "libx264"
    profile_cache: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: int
    render: RenderConfig
    timing: TimingConfig
    station: StationConfig
    motion: MotionConfig
    camera: CameraConfig
    events: EventConfig
    quality: QualityConfig
    batch: BatchConfig
    runtime: RuntimeConfig

    @property
    def digest(self) -> str:
        return stable_digest(self)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported schema_version {self.schema_version}")
        if self.render.width != 640 or self.render.height != 480:
            raise ValueError("benchgen v1 render size must be 640x480")
        low, high = self.timing.duration_range_s
        if not low <= self.timing.duration_s <= high:
            raise ValueError("timing.duration_s is outside duration_range_s")
        if self.timing.event_count < 1:
            raise ValueError("timing.event_count must be positive")
        if not 2 <= self.station.min_targets <= self.station.max_targets:
            raise ValueError("station.min_targets must be between 2 and max_targets")
        if self.batch.episode_count < 1:
            raise ValueError("batch.episode_count must be positive")
        if len(set(self.batch.scenes)) < self.batch.episode_count:
            raise ValueError(
                "batch.scenes must contain at least episode_count distinct scenes")
        if not self.events.enabled_types:
            raise ValueError("events.enabled_types cannot be empty")


def config_from_mapping(raw: Mapping[str, Any]) -> BenchmarkConfig:
    render = _mapping(raw.get("render", {}), "render")
    timing = _mapping(raw.get("timing", {}), "timing")
    station = _mapping(raw.get("station", {}), "station")
    motion = _mapping(raw.get("motion", {}), "motion")
    camera = _mapping(raw.get("camera", {}), "camera")
    events = _mapping(raw.get("events", {}), "events")
    quality = _mapping(raw.get("quality", {}), "quality")
    batch = _mapping(raw.get("batch", {}), "batch")
    runtime = _mapping(raw.get("runtime", {}), "runtime")
    cfg = BenchmarkConfig(
        schema_version=int(raw.get("schema_version", 1)),
        render=RenderConfig(
            width=int(render.get("width", 640)),
            height=int(render.get("height", 480)),
            field_of_view=float(render.get("field_of_view", 55.0)),
            fps=int(render.get("fps", 30)),
        ),
        timing=TimingConfig(
            duration_s=float(timing.get("duration_s", 60.0)),
            duration_range_s=_pair(
                timing.get("duration_range_s", (55.0, 65.0)),
                "timing.duration_range_s",
            ),
            event_count=int(timing.get("event_count", 10)),
            first_event_max_s=float(timing.get("first_event_max_s", 4.0)),
            last_event_min_s=float(timing.get("last_event_min_s", 50.0)),
            max_event_gap_s=float(timing.get("max_event_gap_s", 8.0)),
            before_lead_s=float(timing.get("before_lead_s", 1.05)),
            hidden_lead_s=float(timing.get("hidden_lead_s", 0.10)),
            after_lag_s=float(timing.get("after_lag_s", 1.20)),
            event_jitter_s=float(timing.get("event_jitter_s", 0.45)),
        ),
        station=StationConfig(
            surface_types=tuple(station.get("surface_types", ("CounterTop",))),
            target_types=tuple(station.get("target_types", ())),
            min_targets=int(station.get("min_targets", 2)),
            preferred_distance=float(station.get("preferred_distance", 0.78)),
            min_distance=float(station.get("min_distance", 0.5)),
            max_distance=float(station.get("max_distance", 1.2)),
            horizons=tuple(float(v) for v in station.get(
                "horizons", (8.0, 12.0, 16.0, 20.0))),
            distractor_offsets=tuple(float(v) for v in station.get(
                "distractor_offsets", (58.0, -58.0, 72.0, -72.0))),
            max_targets=int(station.get("max_targets", 18)),
            max_position_candidates=int(station.get("max_position_candidates", 16)),
            max_stations_per_target=int(station.get("max_stations_per_target", 3)),
            max_total_stations=int(station.get("max_total_stations", 18)),
        ),
        motion=MotionConfig(
            preferred_speed_mps=float(motion.get("preferred_speed_mps", 0.42)),
            max_segment_speed_mps=float(motion.get("max_segment_speed_mps", 0.85)),
            max_station_step_m=float(motion.get("max_station_step_m", 3.5)),
            min_station_step_m=float(motion.get("min_station_step_m", 0.35)),
            max_station_reuse=int(motion.get("max_station_reuse", 3)),
            route_candidate_count=int(motion.get("route_candidate_count", 20000)),
            functional_radius_m=float(motion.get("functional_radius_m", 1.8)),
        ),
        camera=CameraConfig(
            max_yaw_rate_dps=float(camera.get("max_yaw_rate_dps", 210.0)),
            max_yaw_accel_dps2=float(camera.get("max_yaw_accel_dps2", 700.0)),
            turn_window_s=float(camera.get("turn_window_s", 0.62)),
            hidden_hold_s=float(camera.get("hidden_hold_s", 0.16)),
            focus_hold_s=float(camera.get("focus_hold_s", 0.5)),
        ),
        events=EventConfig(
            enabled_types=tuple(events.get("enabled_types", ("disappear", "appear"))),
            allow_simultaneous=bool(events.get("allow_simultaneous", True)),
            max_simultaneous=int(events.get("max_simultaneous", 2)),
        ),
        quality=QualityConfig(
            min_target_pixels=int(quality.get("min_target_pixels", 1400)),
            min_target_dimension=int(quality.get("min_target_dimension", 28)),
            min_edge_margin=int(quality.get("min_edge_margin", 4)),
            max_center_distance=float(quality.get("max_center_distance", 0.9)),
            distractor_min_pixels=int(quality.get("distractor_min_pixels", 160)),
            distractor_min_objects=int(quality.get("distractor_min_objects", 1)),
            max_stationary_fraction=float(quality.get("max_stationary_fraction", 0.18)),
            min_travel_distance_m=float(quality.get("min_travel_distance_m", 8.0)),
        ),
        batch=BatchConfig(
            scenes=tuple(str(v) for v in batch.get("scenes", ())),
            episode_count=int(batch.get("episode_count", 2)),
            max_attempts_per_scene=int(batch.get("max_attempts_per_scene", 4)),
        ),
        runtime=RuntimeConfig(
            force_action=bool(runtime.get("force_action", True)),
            highlight_alpha=float(runtime.get("highlight_alpha", 0.9)),
            video_codec=str(runtime.get("video_codec", "libx264")),
            profile_cache=bool(runtime.get("profile_cache", True)),
        ),
    )
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_mapping(_mapping(raw, "config"))
