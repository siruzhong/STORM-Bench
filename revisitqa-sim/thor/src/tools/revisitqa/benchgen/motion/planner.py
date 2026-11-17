"""Plan continuous station-to-station motion on simulator-verified paths."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from ..domain.config import BenchmarkConfig
from ..domain.contracts import (
    BodyKeyframe,
    EpisodeRecipe,
    MobilityMetrics,
    MobilityPlan,
    ObservationStation,
    SceneProfile,
    StationPath,
    Vec3,
    ViewOpportunity,
)
from ..domain.failures import RetryScope, StageFailure


def _distance(left: Vec3, right: Vec3) -> float:
    return math.hypot(right.x - left.x, right.z - left.z)


def _reverse_path(path: StationPath) -> StationPath:
    return StationPath(
        start_station_id=path.end_station_id,
        end_station_id=path.start_station_id,
        points=tuple(reversed(path.points)),
        distance_m=path.distance_m,
    )


def _path_map(profile: SceneProfile) -> dict[tuple[str, str], StationPath]:
    result = {}
    for path in profile.station_paths:
        result[(path.start_station_id, path.end_station_id)] = path
        result.setdefault(
            (path.end_station_id, path.start_station_id), _reverse_path(path))
    return result


def _station_pool(
    profile: SceneProfile,
) -> dict[str, list[ObservationStation]]:
    by_target: dict[str, list[ObservationStation]] = defaultdict(list)
    for station in profile.stations:
        by_target[station.target_id].append(station)
    for stations in by_target.values():
        stations.sort(key=lambda item: -item.score)
    return by_target


def _paired_target_sequence(
    pool: dict[str, list[ObservationStation]], count: int, rng: random.Random,
) -> list[str]:
    if count % 2:
        raise ValueError("presence-change opportunities require an even count")
    target_ids = sorted(
        pool, key=lambda target_id: -pool[target_id][0].score)
    unique_count = min(len(target_ids), max(2, count // 2))
    weighted_pool = target_ids[:max(
        unique_count, min(len(target_ids), unique_count * 3))]
    chosen_targets = rng.sample(weighted_pool, unique_count)
    multiplicities = {target_id: 2 for target_id in chosen_targets}
    remaining = count - unique_count * 2
    while remaining:
        multiplicities[rng.choice(chosen_targets)] += 2
        remaining -= 2
    multiset = [
        target_id
        for target_id, occurrences in multiplicities.items()
        for _ in range(occurrences)
    ]
    for _ in range(200):
        rng.shuffle(multiset)
        if all(left != right for left, right in zip(multiset, multiset[1:])):
            return multiset
    # This is reachable only for unusually skewed future configurations.  The
    # greedy fallback keeps the most frequent remaining target away from the
    # previous slot whenever a different choice exists.
    sequence = []
    remaining_counts = Counter(multiset)
    while remaining_counts:
        candidates = [
            target_id
            for target_id in remaining_counts
            if not sequence or target_id != sequence[-1]
        ]
        if not candidates:
            candidates = list(remaining_counts)
        target_id = max(candidates, key=lambda item: remaining_counts[item])
        sequence.append(target_id)
        remaining_counts[target_id] -= 1
        if not remaining_counts[target_id]:
            del remaining_counts[target_id]
    return sequence


def _station_pair_options(
    pool: dict[str, list[ObservationStation]],
    paths: dict[tuple[str, str], StationPath],
    config: BenchmarkConfig,
) -> dict[str, list[tuple[ObservationStation, ObservationStation]]]:
    options = defaultdict(list)
    for target_id, stations in pool.items():
        for left in stations:
            for right in stations:
                if left.station_id == right.station_id:
                    continue
                if (left.station_id, right.station_id) not in paths:
                    continue
                return_delta = abs(
                    (right.target_yaw - left.hidden_yaw + 180.0)
                    % 360.0 - 180.0)
                peak_rate = (
                    return_delta * 1.875
                    / max(config.timing.after_lag_s, 1e-6))
                if peak_rate > config.camera.max_yaw_rate_dps:
                    continue
                options[target_id].append((left, right))
    return dict(options)


def _route_score(
    route: list[ObservationStation],
    times: list[float],
    paths: dict[tuple[str, str], StationPath],
    config: BenchmarkConfig,
) -> tuple[float, list[StationPath], list[float]] | None:
    segments = []
    speeds = []
    for left, right, start_time, end_time in zip(
            route, route[1:], times, times[1:]):
        path = paths.get((left.station_id, right.station_id))
        if path is None or path.distance_m <= 1e-4:
            return None
        duration = end_time - start_time
        speed = path.distance_m / max(duration, 1e-6)
        if speed > config.motion.max_segment_speed_mps:
            return None
        if path.distance_m > config.motion.max_station_step_m:
            return None
        segments.append(path)
        speeds.append(speed)
    distance = sum(path.distance_m for path in segments)
    if distance < config.quality.min_travel_distance_m:
        return None
    target = config.motion.preferred_speed_mps
    mean_speed = distance / max(times[-1] - times[0], 1e-6)
    target_diversity = len({item.target_id for item in route[1:-1]})
    surface_diversity = len({item.surface_id for item in route[1:-1]})
    reuse_penalty = sum(
        max(0, reuse - config.motion.max_station_reuse)
        for reuse in Counter(item.station_id for item in route).values()
    )
    score = (
        -abs(mean_speed - target) * 80.0
        + target_diversity * 5.0
        + surface_diversity * 2.0
        - reuse_penalty * 20.0
    )
    return score, segments, speeds


def _timed_path_keyframes(
    route: list[ObservationStation],
    route_times: list[float],
    segments: list[StationPath],
) -> tuple[BodyKeyframe, ...]:
    result = [BodyKeyframe(
        time_s=route_times[0],
        position=route[0].position,
        station_id=route[0].station_id,
    )]
    for segment_index, path in enumerate(segments):
        start_time = route_times[segment_index]
        end_time = route_times[segment_index + 1]
        points = path.points
        if not points:
            points = (route[segment_index].position,
                      route[segment_index + 1].position)
        cumulative = [0.0]
        for left, right in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + _distance(left, right))
        total = max(cumulative[-1], 1e-6)
        for point_index, point in enumerate(points[1:], start=1):
            progress = cumulative[point_index] / total
            result.append(BodyKeyframe(
                time_s=start_time + (end_time - start_time) * progress,
                position=point,
                station_id=route[segment_index + 1].station_id,
            ))
    deduped = []
    for frame in result:
        if deduped and frame.time_s <= deduped[-1].time_s + 1e-6:
            continue
        deduped.append(frame)
    return tuple(deduped)


def plan_mobility(
    profile: SceneProfile,
    recipe: EpisodeRecipe,
    config: BenchmarkConfig,
) -> MobilityPlan:
    all_stations = _station_pool(profile)
    paths = _path_map(profile)
    pair_options = _station_pair_options(all_stations, paths, config)
    pool = {
        target_id: all_stations[target_id]
        for target_id in pair_options
    }
    if len(pool) < config.station.min_targets:
        raise StageFailure(
            stage="motion", code="insufficient_stations",
            retry_scope=RetryScope.SCENE_PROFILE,
            detail=(f"scene has only {len(pool)} targets with two connected "
                    "observation stations; requires "
                    f"{config.station.min_targets}"),
        )
    if not paths:
        raise StageFailure(
            stage="motion", code="missing_navigation_edges",
            retry_scope=RetryScope.SCENE_PROFILE,
            detail="scene profile has no simulator-verified station paths",
        )
    station_by_id = {item.station_id: item for item in profile.stations}
    rng = random.Random(recipe.seed ^ 0x4D0710)
    beat_times = list(recipe.beat_times_s)
    best = None
    for _ in range(config.motion.route_candidate_count):
        target_route = _paired_target_sequence(
            pool, recipe.event_count, rng)
        event_pairs = [
            rng.choice(pair_options[target_id]) for target_id in target_route
        ]
        first = event_pairs[0][0]
        last = event_pairs[-1][1]
        start_candidates = [
            station_by_id[path.start_station_id]
            for path in paths.values()
            if path.end_station_id == first.station_id
            and path.start_station_id in station_by_id
            and path.start_station_id != first.station_id
        ]
        end_candidates = [
            station_by_id[path.end_station_id]
            for path in paths.values()
            if path.start_station_id == last.station_id
            and path.end_station_id in station_by_id
            and path.end_station_id != last.station_id
        ]
        if not start_candidates or not end_candidates:
            continue
        route = [rng.choice(start_candidates)]
        route_times = [0.0]
        for (before_station, after_station), beat in zip(
                event_pairs, beat_times):
            route.extend((before_station, after_station))
            route_times.extend((
                max(0.0, beat - config.timing.before_lead_s),
                min(recipe.duration_s, beat + config.timing.after_lag_s),
            ))
        route.append(rng.choice(end_candidates))
        route_times.append(recipe.duration_s)
        evaluated = _route_score(route, route_times, paths, config)
        if evaluated is None:
            continue
        score, segments, speeds = evaluated
        candidate = (
            score, route, route_times, event_pairs, segments, speeds)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise StageFailure(
            stage="motion", code="no_feasible_route",
            retry_scope=RetryScope.EPISODE_SEED,
            detail="no continuous route met duration, speed, and distance gates",
            context={"stations": len(pool), "paths": len(paths)},
        )
    _, route, route_times, event_pairs, segments, speeds = best
    keyframes = _timed_path_keyframes(route, route_times, segments)
    opportunities = tuple(
        ViewOpportunity(
            index=index,
            target_id=before_station.target_id,
            before_station_id=before_station.station_id,
            after_station_id=after_station.station_id,
            before_time_s=max(0.0, beat - config.timing.before_lead_s),
            hidden_time_s=max(0.0, beat - config.timing.hidden_lead_s),
            trigger_time_s=beat,
            after_time_s=min(
                recipe.duration_s, beat + config.timing.after_lag_s),
        )
        for index, ((before_station, after_station), beat) in enumerate(
            zip(event_pairs, beat_times))
    )
    distance = sum(path.distance_m for path in segments)
    duration = max(recipe.duration_s, 1e-6)
    return MobilityPlan(
        keyframes=keyframes,
        opportunities=opportunities,
        metrics=MobilityMetrics(
            distance_m=distance,
            mean_speed_mps=distance / duration,
            max_segment_speed_mps=max(speeds, default=0.0),
            stationary_fraction=0.0,
        ),
    )
