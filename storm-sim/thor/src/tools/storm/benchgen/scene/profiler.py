"""Pure conversion of simulator observations into a SceneProfile."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..domain.contracts import (
    ObservationStation,
    SceneObject,
    SceneProfile,
    StationPath,
    Vec3,
)


CAPABILITY_FIELDS = {
    "pickupable": "pickup",
    "sliceable": "slice",
    "breakable": "break",
    "toggleable": "toggle",
    "dirtyable": "dirty",
    "cookable": "cook",
    "openable": "open",
    "canFillWithLiquid": "fill",
}


def _vec(position: Mapping[str, Any]) -> Vec3:
    return Vec3(
        x=float(position["x"]),
        y=float(position.get("y", 0.0)),
        z=float(position["z"]),
    )


def object_capabilities(raw: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(
        capability for field, capability in CAPABILITY_FIELDS.items()
        if raw.get(field)
    ))


def build_scene_profile(
    scene: str,
    simulator_build: str,
    reachable: Iterable[Mapping[str, Any]],
    objects: Iterable[Mapping[str, Any]],
    stations: Iterable[Mapping[str, Any]],
    station_paths: Iterable[Mapping[str, Any]] = (),
) -> SceneProfile:
    scene_objects = tuple(sorted((
        SceneObject(
            object_id=str(raw["objectId"]),
            object_type=str(raw["objectType"]),
            position=_vec(raw["position"]),
            parent_receptacles=tuple(raw.get("parentReceptacles") or ()),
            pickupable=bool(raw.get("pickupable")),
            capabilities=object_capabilities(raw),
        )
        for raw in objects
    ), key=lambda item: item.object_id))
    observation_stations = tuple(sorted((
        ObservationStation(
            station_id=str(raw["station_id"]),
            position=_vec(raw["position"]),
            target_yaw=float(raw["target_yaw"]),
            hidden_yaw=float(raw["hidden_yaw"]),
            target_horizon=float(raw["target_horizon"]),
            hidden_horizon=float(raw["hidden_horizon"]),
            target_id=str(raw["target_id"]),
            target_type=str(raw["target_type"]),
            surface_id=str(raw["surface_id"]),
            surface_type=str(raw["surface_type"]),
            distractor_ids=tuple(raw.get("distractor_ids") or ()),
            target_pixels=int(raw.get("target_pixels", 0)),
            target_bbox_width=int(raw.get("target_bbox_width", 0)),
            target_bbox_height=int(raw.get("target_bbox_height", 0)),
            target_center_distance=float(raw.get("target_center_distance", 1.0)),
            score=float(raw.get("score", 0.0)),
        )
        for raw in stations
    ), key=lambda item: item.station_id))
    navigation_paths = tuple(sorted((
        StationPath(
            start_station_id=str(raw["start_station_id"]),
            end_station_id=str(raw["end_station_id"]),
            points=tuple(_vec(point) for point in raw["points"]),
            distance_m=float(raw["distance_m"]),
        )
        for raw in station_paths
    ), key=lambda item: (item.start_station_id, item.end_station_id)))
    return SceneProfile.create(
        scene=scene,
        simulator_build=simulator_build,
        reachable=tuple(_vec(position) for position in reachable),
        objects=scene_objects,
        stations=observation_stations,
        station_paths=navigation_paths,
    )


def scene_profile_from_dict(raw: Mapping[str, Any]) -> SceneProfile:
    reachable = tuple(_vec(item) for item in raw["reachable"])
    objects = tuple(SceneObject(
        object_id=str(item["object_id"]),
        object_type=str(item["object_type"]),
        position=_vec(item["position"]),
        parent_receptacles=tuple(item.get("parent_receptacles") or ()),
        pickupable=bool(item.get("pickupable")),
        capabilities=tuple(item.get("capabilities") or ()),
    ) for item in raw["objects"])
    stations = tuple(ObservationStation(
        station_id=str(item["station_id"]),
        position=_vec(item["position"]),
        target_yaw=float(item["target_yaw"]),
        hidden_yaw=float(item["hidden_yaw"]),
        target_horizon=float(item["target_horizon"]),
        hidden_horizon=float(item["hidden_horizon"]),
        target_id=str(item["target_id"]),
        target_type=str(item["target_type"]),
        surface_id=str(item["surface_id"]),
        surface_type=str(item["surface_type"]),
        distractor_ids=tuple(item.get("distractor_ids") or ()),
        target_pixels=int(item.get("target_pixels", 0)),
        target_bbox_width=int(item.get("target_bbox_width", 0)),
        target_bbox_height=int(item.get("target_bbox_height", 0)),
        target_center_distance=float(item.get("target_center_distance", 1.0)),
        score=float(item.get("score", 0.0)),
    ) for item in raw["stations"])
    paths = tuple(StationPath(
        start_station_id=str(item["start_station_id"]),
        end_station_id=str(item["end_station_id"]),
        points=tuple(_vec(point) for point in item["points"]),
        distance_m=float(item["distance_m"]),
    ) for item in raw.get("station_paths", ()))
    profile = SceneProfile.create(
        scene=str(raw["scene"]),
        simulator_build=str(raw["simulator_build"]),
        reachable=reachable,
        objects=objects,
        stations=stations,
        station_paths=paths,
    )
    expected = raw.get("profile_id")
    if expected and profile.profile_id != expected:
        raise ValueError(
            f"scene profile digest mismatch: {profile.profile_id} != {expected}")
    return profile
