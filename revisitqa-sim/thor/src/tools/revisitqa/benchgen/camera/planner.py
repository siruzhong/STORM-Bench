"""Build camera pose keyframes as part of the episode plan."""
from __future__ import annotations

import bisect
import math

from ..domain.config import BenchmarkConfig
from ..domain.contracts import (
    CameraKeyframe,
    CameraPlan,
    EpisodeRecipe,
    EventProgram,
    MobilityPlan,
    ObservationRequirement,
    SceneProfile,
    Vec3,
)


def _heading(position: Vec3, target: Vec3) -> float:
    return math.degrees(math.atan2(
        target.x - position.x, target.z - position.z)) % 360.0


def _body_position(mobility: MobilityPlan, time_s: float) -> Vec3:
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


def plan_camera(
    profile: SceneProfile,
    mobility: MobilityPlan,
    events: EventProgram,
    recipe: EpisodeRecipe,
    config: BenchmarkConfig,
) -> CameraPlan:
    station_by_id = {item.station_id: item for item in profile.stations}
    object_by_id = {item.object_id: item for item in profile.objects}
    opportunity_by_index = {
        item.index: item for item in mobility.opportunities}
    keyframes = []
    requirements = []
    for intent in events.events:
        opportunity = opportunity_by_index[intent.index]
        before_station = station_by_id[opportunity.before_station_id]
        after_station = station_by_id[opportunity.after_station_id]
        target = object_by_id[intent.target_id].position
        before_position = _body_position(mobility, opportunity.before_time_s)
        after_position = _body_position(mobility, opportunity.after_time_s)
        before_yaw = _heading(before_position, target)
        hidden_yaw = before_station.hidden_yaw
        # Hold the same world-space yaw through the intervention. Recomputing
        # look-at yaw during this short hidden window creates a tiny fast turn.
        trigger_yaw = hidden_yaw
        after_yaw = _heading(after_position, target)
        keyframes.extend((
            CameraKeyframe(
                time_s=opportunity.before_time_s,
                yaw=before_yaw,
                horizon=before_station.target_horizon,
                phase="before",
                focus_ids=(intent.target_id,),
            ),
            CameraKeyframe(
                time_s=opportunity.hidden_time_s,
                yaw=hidden_yaw,
                horizon=before_station.hidden_horizon,
                phase="hidden",
                focus_ids=before_station.distractor_ids,
            ),
            CameraKeyframe(
                time_s=opportunity.trigger_time_s,
                yaw=trigger_yaw,
                horizon=before_station.hidden_horizon,
                phase="intervention",
                focus_ids=before_station.distractor_ids,
            ),
            CameraKeyframe(
                time_s=opportunity.after_time_s,
                yaw=after_yaw,
                horizon=after_station.target_horizon,
                phase="after",
                focus_ids=(intent.target_id,),
            ),
        ))
        requirements.extend((
            ObservationRequirement(
                event_index=intent.index,
                phase="before",
                time_s=opportunity.before_time_s,
                station_id=opportunity.before_station_id,
                focus_ids=(intent.target_id,),
                must_be_visible=(intent.event_type != "appear"),
            ),
            ObservationRequirement(
                event_index=intent.index,
                phase="intervention",
                time_s=opportunity.trigger_time_s,
                station_id=opportunity.before_station_id,
                focus_ids=(intent.target_id,),
                must_be_visible=False,
            ),
            ObservationRequirement(
                event_index=intent.index,
                phase="after",
                time_s=opportunity.after_time_s,
                station_id=opportunity.after_station_id,
                focus_ids=(intent.target_id,),
                must_be_visible=(intent.event_type != "disappear"),
            ),
        ))
    keyframes.sort(key=lambda item: item.time_s)
    if keyframes:
        first = keyframes[0]
        last = keyframes[-1]
        keyframes.insert(0, CameraKeyframe(
            time_s=0.0, yaw=first.yaw, horizon=first.horizon,
            phase="approach", focus_ids=first.focus_ids,
        ))
        keyframes.append(CameraKeyframe(
            time_s=recipe.duration_s, yaw=last.yaw, horizon=last.horizon,
            phase="depart", focus_ids=last.focus_ids,
        ))
    return CameraPlan(
        keyframes=tuple(keyframes),
        requirements=tuple(requirements),
    )
