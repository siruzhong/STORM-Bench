"""Rendered worktop station selection for egocentric memory episodes."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .evidence import evidence_passes, frame_evidence
from .path import heading_to, shortest_delta
from .visibility import in_frame_ids


@dataclass
class EgoStation:
    position: dict
    yaw_a: float
    yaw_b: float
    horizon_a: float
    horizon_b: float
    target_id: str
    target_type: str
    surface_id: str
    surface_type: str
    target_evidence: dict
    distractor_object_count: int
    distractor_visible_ids: list[str]
    score: float

    def to_dict(self):
        return asdict(self)


def supporting_surface(obj, objects_by_id, surface_types=("CounterTop",)):
    for receptacle_id in reversed(obj.get("parentReceptacles") or []):
        receptacle = objects_by_id.get(receptacle_id)
        if receptacle and receptacle.get("objectType") in surface_types:
            return receptacle
    return None


def _teleport(controller, position, yaw, horizon):
    return controller.step(
        action="TeleportFull",
        position={
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        },
        rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
        horizon=float(horizon),
        standing=True,
        forceAction=True,
    )


def semantic_visible_ids(event, objects_by_id, min_pixels, excluded=()):
    """Return visible objects that provide a meaningful non-structural view."""
    visible = in_frame_ids(event, min_pixels=min_pixels)
    structural_types = {"Floor", "Wall", "Ceiling"}
    return sorted(
        object_id for object_id in visible
        if object_id not in excluded
        and objects_by_id.get(object_id)
        and objects_by_id[object_id].get("objectType") not in structural_types
    )


def find_ego_stations(controller, reachable, target, objects,
                      surface_types=("CounterTop",),
                      preferred_distance=0.8,
                      min_distance=0.55,
                      max_distance=1.15,
                      horizons=(10.0, 15.0, 20.0, 25.0),
                      distractor_offsets=(60.0, -60.0),
                      min_target_pixels=3000,
                      min_target_dimension=40,
                      min_edge_margin=8,
                      max_center_distance=0.9,
                      distractor_min_pixels=300,
                      distractor_min_objects=1,
                      max_position_candidates=12,
                      additional_hidden_ids=(),
                      max_results=4):
    """Probe renders and retain several spatially distinct A/B stations."""
    objects_by_id = {obj["objectId"]: obj for obj in objects}
    surface = supporting_surface(target, objects_by_id, surface_types)
    if surface is None:
        return []

    target_pos = target["position"]
    position_candidates = []
    for position in reachable:
        distance = math.hypot(
            float(position["x"]) - float(target_pos["x"]),
            float(position["z"]) - float(target_pos["z"]),
        )
        if min_distance <= distance <= max_distance:
            position_candidates.append((abs(distance - preferred_distance), position))
    position_candidates.sort(key=lambda item: (
        item[0], float(item[1]["x"]), float(item[1]["z"])))

    best_by_position = {}
    for _, position in position_candidates[:max_position_candidates]:
        position_best = None
        yaw_a = heading_to(
            float(target_pos["x"]) - float(position["x"]),
            float(target_pos["z"]) - float(position["z"]),
        )
        for horizon in horizons:
            event_a = _teleport(controller, position, yaw_a, horizon)
            if not event_a.metadata["lastActionSuccess"]:
                continue
            target_evidence = frame_evidence(event_a, [target["objectId"]])
            if not evidence_passes(
                    target_evidence,
                    min_pixels=min_target_pixels,
                    min_dimension=min_target_dimension,
                    min_edge_margin=min_edge_margin,
                    max_center_distance=max_center_distance):
                continue

            for offset in distractor_offsets:
                yaw_b = (yaw_a + offset) % 360.0
                event_b = _teleport(controller, position, yaw_b, horizon)
                if not event_b.metadata["lastActionSuccess"]:
                    continue
                hidden_ids = {target["objectId"], *additional_hidden_ids}
                if hidden_ids & in_frame_ids(event_b, min_pixels=1):
                    continue
                distractor_ids = semantic_visible_ids(
                    event_b,
                    objects_by_id,
                    distractor_min_pixels,
                    excluded={target["objectId"]},
                )
                if len(distractor_ids) < int(distractor_min_objects):
                    continue
                angular_separation = abs(shortest_delta(yaw_a, yaw_b))
                score = (
                    target_evidence["mask_pixels"]
                    + len(distractor_ids) * 500.0
                    + target_evidence["edge_margin"] * 20.0
                    - target_evidence["center_distance"] * 1000.0
                    - abs(angular_separation - 65.0) * 10.0
                )
                candidate = EgoStation(
                    position={
                        "x": float(position["x"]),
                        "y": float(position["y"]),
                        "z": float(position["z"]),
                    },
                    yaw_a=float(yaw_a),
                    yaw_b=float(yaw_b),
                    horizon_a=float(horizon),
                    horizon_b=float(horizon),
                    target_id=target["objectId"],
                    target_type=target["objectType"],
                    surface_id=surface["objectId"],
                    surface_type=surface["objectType"],
                    target_evidence=target_evidence,
                    distractor_object_count=len(distractor_ids),
                    distractor_visible_ids=distractor_ids,
                    score=float(score),
                )
                if (position_best is None
                        or candidate.score > position_best.score):
                    position_best = candidate
            if position_best is not None:
                break
        if position_best is not None:
            position_key = (
                round(position_best.position["x"], 4),
                round(position_best.position["z"], 4),
            )
            best_by_position[position_key] = position_best
    ranked = sorted(
        best_by_position.values(), key=lambda candidate: -candidate.score)
    return ranked[:max(1, int(max_results))]


def find_ego_station(controller, reachable, target, objects, **kwargs):
    """Return the highest-scoring station for the single-event runner."""
    stations = find_ego_stations(
        controller, reachable, target, objects, max_results=1, **kwargs)
    return stations[0] if stations else None
