"""AI2-THOR adapter for scene inspection and verified navigation edges."""
from __future__ import annotations

import itertools
import logging
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from ai2thor.controller import Controller

from ...ego_station import find_ego_stations, supporting_surface
from ..domain.config import BenchmarkConfig
from ..domain.contracts import DenseFrame, SceneProfile, stable_digest
from ..domain.failures import RetryScope, StageFailure
from ..scene.profiler import build_scene_profile


LOGGER = logging.getLogger(__name__)


class Ai2ThorSession:
    """Own one controller and expose only low-level simulator operations."""

    def __init__(self) -> None:
        self.controller: Controller | None = None
        self.scene: str | None = None

    @property
    def event(self):
        if self.controller is None:
            raise RuntimeError("AI2-THOR session is not initialized")
        return self.controller.last_event

    @property
    def simulator_build(self) -> str:
        try:
            return f"ai2thor-{version('ai2thor')}"
        except PackageNotFoundError:
            return "ai2thor-unknown"

    def reset(self, scene: str, config: BenchmarkConfig) -> Any:
        if self.controller is None:
            kwargs = {
                "scene": scene,
                "width": config.render.width,
                "height": config.render.height,
                "fieldOfView": config.render.field_of_view,
                "gridSize": 0.25,
                "renderInstanceSegmentation": True,
            }
            local_executable = os.environ.get("STORM_THOR_EXECUTABLE")
            kwargs["platform"] = os.environ.get(
                "STORM_THOR_PLATFORM", "CloudRendering" if sys.platform == "linux" else "OSXIntel64"
            )
            if local_executable:
                if not os.path.isfile(local_executable):
                    raise FileNotFoundError(local_executable)
                kwargs["local_executable_path"] = local_executable
                LOGGER.info("using local THOR executable: %s", local_executable)
            else:
                kwargs["commit_id"] = os.environ.get(
                    "STORM_THOR_COMMIT", "4d2e1f1d04051fafcd9794b810f227551121253a"
                )
            self.controller = Controller(**kwargs)
        else:
            self.controller.reset(scene)
        self.scene = scene
        return self.controller.step(action="Pass")

    def step(self, **kwargs):
        if self.controller is None:
            raise RuntimeError("AI2-THOR session is not initialized")
        return self.controller.step(**kwargs)

    def teleport(self, frame: DenseFrame, force_action: bool = True):
        return self.step(
            action="TeleportFull",
            position={
                "x": float(frame.position.x),
                "y": float(frame.position.y),
                "z": float(frame.position.z),
            },
            rotation={"x": 0.0, "y": float(frame.yaw), "z": 0.0},
            horizon=float(frame.horizon),
            standing=True,
            forceAction=bool(force_action),
        )

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None
            self.scene = None


def _path_distance(points: list[dict[str, float]]) -> float:
    return sum(math.hypot(
        float(right["x"]) - float(left["x"]),
        float(right["z"]) - float(left["z"]),
    ) for left, right in zip(points, points[1:]))


def _same_position(left: dict[str, float], right: dict[str, float]) -> bool:
    return math.hypot(
        float(right["x"]) - float(left["x"]),
        float(right["z"]) - float(left["z"]),
    ) < 1e-4


class Ai2ThorSceneProfiler:
    """Render candidate work-area stations and build a verified route graph."""

    def __init__(self, session: Ai2ThorSession):
        self.session = session

    def _candidate_targets(
        self, objects: list[dict[str, Any]], config: BenchmarkConfig,
    ) -> list[dict[str, Any]]:
        by_id = {item["objectId"]: item for item in objects}
        target_types = set(config.station.target_types)
        candidates = []
        for item in objects:
            if not item.get("pickupable"):
                continue
            if target_types and item.get("objectType") not in target_types:
                continue
            surface = supporting_surface(
                item, by_id, config.station.surface_types)
            if surface is None:
                continue
            candidates.append(item)
        candidates.sort(key=lambda item: (
            item.get("objectType", ""), item.get("objectId", "")))
        return candidates[:config.station.max_targets]

    def _probe_stations(
        self,
        reachable: list[dict[str, float]],
        objects: list[dict[str, Any]],
        config: BenchmarkConfig,
    ) -> list[dict[str, Any]]:
        rows = []
        for target in self._candidate_targets(objects, config):
            stations = find_ego_stations(
                self.session.controller,
                reachable,
                target,
                objects,
                surface_types=config.station.surface_types,
                preferred_distance=config.station.preferred_distance,
                min_distance=config.station.min_distance,
                max_distance=config.station.max_distance,
                horizons=config.station.horizons,
                distractor_offsets=config.station.distractor_offsets,
                min_target_pixels=config.quality.min_target_pixels,
                min_target_dimension=config.quality.min_target_dimension,
                min_edge_margin=config.quality.min_edge_margin,
                max_center_distance=config.quality.max_center_distance,
                distractor_min_pixels=config.quality.distractor_min_pixels,
                distractor_min_objects=config.quality.distractor_min_objects,
                max_position_candidates=config.station.max_position_candidates,
                max_results=config.station.max_stations_per_target,
            )
            for station in stations:
                evidence = station.target_evidence
                station_key = {
                    "target": station.target_id,
                    "x": round(float(station.position["x"]), 4),
                    "z": round(float(station.position["z"]), 4),
                }
                rows.append({
                    "station_id": f"station-{stable_digest(station_key, 12)}",
                    "position": station.position,
                    "target_yaw": station.yaw_a,
                    "hidden_yaw": station.yaw_b,
                    "target_horizon": station.horizon_a,
                    "hidden_horizon": station.horizon_b,
                    "target_id": station.target_id,
                    "target_type": station.target_type,
                    "surface_id": station.surface_id,
                    "surface_type": station.surface_type,
                    "distractor_ids": station.distractor_visible_ids,
                    "target_pixels": evidence.get("mask_pixels", 0),
                    "target_bbox_width": evidence.get("bbox_width", 0),
                    "target_bbox_height": evidence.get("bbox_height", 0),
                    "target_center_distance": evidence.get(
                        "center_distance", 1.0),
                    "score": station.score,
                })
        rows.sort(key=lambda row: -float(row["score"]))
        selected = []
        target_counts: dict[str, int] = {}
        for row in rows:
            target_id = str(row["target_id"])
            count = target_counts.get(target_id, 0)
            if count >= config.station.max_stations_per_target:
                continue
            selected.append(row)
            target_counts[target_id] = count + 1
            if len(selected) >= config.station.max_total_stations:
                break
        return selected

    def _verified_paths(
        self,
        stations: list[dict[str, Any]],
        config: BenchmarkConfig,
    ) -> list[dict[str, Any]]:
        paths = []
        for left, right in itertools.combinations(stations, 2):
            start = left["position"]
            goal = right["position"]
            direct = math.hypot(
                float(goal["x"]) - float(start["x"]),
                float(goal["z"]) - float(start["z"]),
            )
            if direct < config.motion.min_station_step_m:
                continue
            if direct > config.motion.max_station_step_m:
                continue
            event = self.session.step(
                action="GetShortestPathToPoint",
                position={
                    "x": float(start["x"]), "y": float(start["y"]),
                    "z": float(start["z"]),
                },
                target={
                    "x": float(goal["x"]), "y": float(goal["y"]),
                    "z": float(goal["z"]),
                },
                allowedError=0.15,
            )
            if not event.metadata.get("lastActionSuccess"):
                continue
            returned = event.metadata.get("actionReturn") or {}
            points = list(returned.get("corners") or ())
            if not points:
                continue
            if not _same_position(start, points[0]):
                points.insert(0, dict(start))
            else:
                points[0] = dict(start)
            if not _same_position(points[-1], goal):
                points.append(dict(goal))
            else:
                points[-1] = dict(goal)
            # Navmesh corners carry floor height, while TeleportFull expects the
            # reachable Agent height. Keep XZ corners but normalize their Y.
            agent_y = float(start["y"])
            points = [{
                "x": float(point["x"]),
                "y": agent_y,
                "z": float(point["z"]),
            } for point in points]
            distance = _path_distance(points)
            if distance > config.motion.max_station_step_m * 1.35:
                continue
            paths.append({
                "start_station_id": left["station_id"],
                "end_station_id": right["station_id"],
                "points": points,
                "distance_m": distance,
            })
        return paths

    def profile(self, scene: str, config: BenchmarkConfig) -> SceneProfile:
        initial = self.session.reset(scene, config)
        objects = list(initial.metadata.get("objects") or ())
        reachable_event = self.session.step(action="GetReachablePositions")
        if not reachable_event.metadata.get("lastActionSuccess"):
            raise StageFailure(
                stage="scene_profile", code="reachable_positions_failed",
                retry_scope=RetryScope.SCENE,
                detail=reachable_event.metadata.get("errorMessage", "unknown"),
            )
        reachable = list(reachable_event.metadata.get("actionReturn") or ())
        stations = self._probe_stations(reachable, objects, config)
        target_count = len({row["target_id"] for row in stations})
        if target_count < config.station.min_targets:
            raise StageFailure(
                stage="scene_profile", code="insufficient_observable_targets",
                retry_scope=RetryScope.SCENE,
                detail=(f"found {target_count} observable work-area targets; "
                        f"requires {config.station.min_targets}"),
            )
        paths = self._verified_paths(stations, config)
        if not paths:
            raise StageFailure(
                stage="scene_profile", code="no_verified_station_paths",
                retry_scope=RetryScope.SCENE,
                detail="no station pair has a simulator-confirmed path",
            )
        return build_scene_profile(
            scene=scene,
            simulator_build=self.session.simulator_build,
            reachable=reachable,
            objects=objects,
            stations=stations,
            station_paths=paths,
        )
