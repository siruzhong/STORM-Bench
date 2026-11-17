"""Immutable, serializable contracts for the plan-first pipeline."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = 1


def primitive(value: Any) -> Any:
    """Convert domain values into deterministic JSON-compatible values."""
    if dataclasses.is_dataclass(value):
        return {
            item.name: primitive(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    return value


def stable_digest(value: Any, length: int = 16) -> str:
    encoded = json.dumps(
        primitive(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    object_type: str
    position: Vec3
    parent_receptacles: tuple[str, ...] = ()
    pickupable: bool = False
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationStation:
    station_id: str
    position: Vec3
    target_yaw: float
    hidden_yaw: float
    target_horizon: float
    hidden_horizon: float
    target_id: str
    target_type: str
    surface_id: str
    surface_type: str
    distractor_ids: tuple[str, ...]
    target_pixels: int
    target_bbox_width: int
    target_bbox_height: int
    target_center_distance: float
    score: float


@dataclass(frozen=True)
class StationPath:
    start_station_id: str
    end_station_id: str
    points: tuple[Vec3, ...]
    distance_m: float


@dataclass(frozen=True)
class SceneProfile:
    schema_version: int
    scene: str
    simulator_build: str
    reachable: tuple[Vec3, ...]
    objects: tuple[SceneObject, ...]
    stations: tuple[ObservationStation, ...]
    station_paths: tuple[StationPath, ...]
    profile_id: str

    @classmethod
    def create(
        cls,
        scene: str,
        simulator_build: str,
        reachable: tuple[Vec3, ...],
        objects: tuple[SceneObject, ...],
        stations: tuple[ObservationStation, ...],
        station_paths: tuple[StationPath, ...] = (),
    ) -> "SceneProfile":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scene": scene,
            "simulator_build": simulator_build,
            "reachable": reachable,
            "objects": objects,
            "stations": stations,
            "station_paths": station_paths,
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            scene=scene,
            simulator_build=simulator_build,
            reachable=reachable,
            objects=objects,
            stations=stations,
            station_paths=station_paths,
            profile_id=stable_digest(payload),
        )


@dataclass(frozen=True)
class EpisodeRecipe:
    schema_version: int
    scene: str
    seed: int
    duration_s: float
    fps: int
    event_count: int
    beat_times_s: tuple[float, ...]
    recipe_id: str

    @classmethod
    def create(
        cls,
        scene: str,
        seed: int,
        duration_s: float,
        fps: int,
        beat_times_s: tuple[float, ...],
    ) -> "EpisodeRecipe":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "scene": scene,
            "seed": seed,
            "duration_s": duration_s,
            "fps": fps,
            "beat_times_s": beat_times_s,
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            scene=scene,
            seed=seed,
            duration_s=duration_s,
            fps=fps,
            event_count=len(beat_times_s),
            beat_times_s=beat_times_s,
            recipe_id=stable_digest(payload),
        )


@dataclass(frozen=True)
class BodyKeyframe:
    time_s: float
    position: Vec3
    station_id: str


@dataclass(frozen=True)
class ViewOpportunity:
    index: int
    target_id: str
    before_station_id: str
    after_station_id: str
    before_time_s: float
    hidden_time_s: float
    trigger_time_s: float
    after_time_s: float


@dataclass(frozen=True)
class MobilityMetrics:
    distance_m: float
    mean_speed_mps: float
    max_segment_speed_mps: float
    stationary_fraction: float


@dataclass(frozen=True)
class MobilityPlan:
    keyframes: tuple[BodyKeyframe, ...]
    opportunities: tuple[ViewOpportunity, ...]
    metrics: MobilityMetrics


@dataclass(frozen=True)
class EventIntent:
    index: int
    event_type: str
    event_family: str
    event_action: str
    target_id: str
    target_type: str
    station_id: str
    trigger_time_s: float
    parameters: tuple[tuple[str, Any], ...] = ()

    def spec(self) -> dict[str, Any]:
        result = {
            "type": self.event_type,
            "object_id": self.target_id,
            "object": self.target_type,
        }
        result.update(dict(self.parameters))
        return result


@dataclass(frozen=True)
class EventProgram:
    events: tuple[EventIntent, ...]
    program_id: str

    @classmethod
    def create(cls, events: tuple[EventIntent, ...]) -> "EventProgram":
        return cls(events=events, program_id=stable_digest(events))


@dataclass(frozen=True)
class ObservationRequirement:
    event_index: int
    phase: str
    time_s: float
    station_id: str
    focus_ids: tuple[str, ...]
    must_be_visible: bool


@dataclass(frozen=True)
class CameraKeyframe:
    time_s: float
    yaw: float
    horizon: float
    phase: str
    focus_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class CameraPlan:
    keyframes: tuple[CameraKeyframe, ...]
    requirements: tuple[ObservationRequirement, ...]


@dataclass(frozen=True)
class DenseFrame:
    index: int
    time_s: float
    position: Vec3
    yaw: float
    horizon: float
    phase: str
    focus_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class DenseTrajectory:
    fps: int
    frames: tuple[DenseFrame, ...]


@dataclass(frozen=True)
class EpisodePlan:
    schema_version: int
    config_digest: str
    profile_id: str
    recipe: EpisodeRecipe
    mobility: MobilityPlan
    event_program: EventProgram
    camera: CameraPlan
    trajectory: DenseTrajectory
    plan_id: str

    @classmethod
    def create(
        cls,
        config_digest: str,
        profile_id: str,
        recipe: EpisodeRecipe,
        mobility: MobilityPlan,
        event_program: EventProgram,
        camera: CameraPlan,
        trajectory: DenseTrajectory,
    ) -> "EpisodePlan":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_digest": config_digest,
            "profile_id": profile_id,
            "recipe": recipe,
            "mobility": mobility,
            "event_program": event_program,
            "camera": camera,
            "trajectory": trajectory,
        }
        return cls(
            schema_version=SCHEMA_VERSION,
            config_digest=config_digest,
            profile_id=profile_id,
            recipe=recipe,
            mobility=mobility,
            event_program=event_program,
            camera=camera,
            trajectory=trajectory,
            plan_id=stable_digest(payload),
        )


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    value: Any
    limit: Any
    detail: str = ""


@dataclass(frozen=True)
class ValidationReport:
    stage: str
    accepted: bool
    checks: tuple[ValidationCheck, ...]
    failure_code: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EventExecution:
    index: int
    event_type: str
    target_id: str
    applied: bool
    observed: bool
    note: str
    before_evidence: Mapping[str, Any]
    after_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionTrace:
    plan_id: str
    accepted: bool
    events: tuple[EventExecution, ...]
    failed_frames: tuple[int, ...] = ()


def to_dict(value: Any) -> dict[str, Any]:
    result = primitive(value)
    if not isinstance(result, dict):
        raise TypeError("domain value did not serialize to an object")
    return result
