"""Generate seeded event programs against motion-provided view opportunities."""
from __future__ import annotations

import random
from collections import defaultdict

from ..domain.config import BenchmarkConfig
from ..domain.contracts import (
    EpisodeRecipe,
    EventIntent,
    EventProgram,
    MobilityPlan,
    SceneProfile,
)
from ..domain.failures import RetryScope, StageFailure


def plan_events(
    profile: SceneProfile,
    mobility: MobilityPlan,
    recipe: EpisodeRecipe,
    config: BenchmarkConfig,
) -> EventProgram:
    station_by_id = {item.station_id: item for item in profile.stations}
    occurrences = defaultdict(list)
    for opportunity in mobility.opportunities:
        occurrences[opportunity.target_id].append(opportunity.index)
    enabled = set(config.events.enabled_types)
    presence_enabled = {"disappear", "appear"} <= enabled
    rng = random.Random(recipe.seed ^ 0xE73A11)
    event_kind: dict[int, str] = {}
    if presence_enabled:
        targets = list(occurrences)
        rng.shuffle(targets)
        for target_id in targets:
            indexes = occurrences[target_id]
            for pair_start in range(0, len(indexes) - 1, 2):
                event_kind[indexes[pair_start]] = "disappear"
                event_kind[indexes[pair_start + 1]] = "appear"
    if len(event_kind) != recipe.event_count:
        raise StageFailure(
            stage="events", code="unpaired_presence_opportunities",
            retry_scope=RetryScope.MOTION_PLAN,
            detail=(
                f"planned {len(event_kind)} of {recipe.event_count} events; "
                "every selected target needs paired view opportunities"
            ),
        )
    intents = []
    for opportunity in mobility.opportunities:
        station = station_by_id[opportunity.before_station_id]
        event_type = event_kind[opportunity.index]
        intents.append(EventIntent(
            index=opportunity.index,
            event_type=event_type,
            event_family="presence_change",
            event_action=event_type,
            target_id=station.target_id,
            target_type=station.target_type,
            station_id=station.station_id,
            trigger_time_s=opportunity.trigger_time_s,
        ))
    return EventProgram.create(tuple(intents))
