"""Compile sparse body and camera plans into a deterministic frame timeline."""
from __future__ import annotations

import bisect

from ..domain.contracts import (
    CameraPlan,
    DenseFrame,
    DenseTrajectory,
    EpisodeRecipe,
    MobilityPlan,
    Vec3,
)


def _minimum_jerk(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress ** 3 * (10.0 + progress * (-15.0 + 6.0 * progress))


def _shortest_delta(source: float, target: float) -> float:
    return (target - source + 180.0) % 360.0 - 180.0


def _body_at(mobility: MobilityPlan, time_s: float) -> Vec3:
    keyframes = mobility.keyframes
    times = [item.time_s for item in keyframes]
    index = bisect.bisect_right(times, time_s) - 1
    if index < 0:
        return keyframes[0].position
    if index >= len(keyframes) - 1:
        return keyframes[-1].position
    left, right = keyframes[index], keyframes[index + 1]
    progress = (time_s - left.time_s) / max(right.time_s - left.time_s, 1e-9)
    return Vec3(
        x=left.position.x + (right.position.x - left.position.x) * progress,
        y=left.position.y + (right.position.y - left.position.y) * progress,
        z=left.position.z + (right.position.z - left.position.z) * progress,
    )


def _camera_at(camera: CameraPlan, time_s: float):
    keyframes = camera.keyframes
    times = [item.time_s for item in keyframes]
    index = bisect.bisect_right(times, time_s) - 1
    if index < 0:
        return keyframes[0]
    if index >= len(keyframes) - 1:
        return keyframes[-1]
    left, right = keyframes[index], keyframes[index + 1]
    progress = _minimum_jerk(
        (time_s - left.time_s) / max(right.time_s - left.time_s, 1e-9))
    yaw = (left.yaw + _shortest_delta(left.yaw, right.yaw) * progress) % 360.0
    horizon = left.horizon + (right.horizon - left.horizon) * progress
    return type(left)(
        time_s=time_s,
        yaw=yaw,
        horizon=horizon,
        phase=left.phase,
        focus_ids=left.focus_ids,
    )


def compile_trajectory(
    mobility: MobilityPlan,
    camera: CameraPlan,
    recipe: EpisodeRecipe,
) -> DenseTrajectory:
    frame_count = int(round(recipe.duration_s * recipe.fps)) + 1
    frames = []
    for index in range(frame_count):
        time_s = min(recipe.duration_s, index / recipe.fps)
        position = _body_at(mobility, time_s)
        view = _camera_at(camera, time_s)
        frames.append(DenseFrame(
            index=index,
            time_s=time_s,
            position=position,
            yaw=view.yaw,
            horizon=view.horizon,
            phase=view.phase,
            focus_ids=view.focus_ids,
        ))
    return DenseTrajectory(fps=recipe.fps, frames=tuple(frames))
