"""Deterministic composition of the pure planning stages."""
from __future__ import annotations

import random

from ..camera.planner import plan_camera
from ..domain.config import BenchmarkConfig
from ..domain.contracts import EpisodePlan, EpisodeRecipe, SceneProfile
from ..events.planner import plan_events
from ..motion.planner import plan_mobility
from .trajectory import compile_trajectory


def make_recipe(
    scene: str, seed: int, config: BenchmarkConfig,
) -> EpisodeRecipe:
    timing = config.timing
    rng = random.Random(int(seed) ^ 0xB31C6E)
    count = timing.event_count
    first = min(
        timing.first_event_max_s,
        max(1.8, timing.first_event_max_s - rng.uniform(0.15, 0.7)),
    )
    last = max(
        timing.last_event_min_s,
        timing.duration_s - rng.uniform(3.0, 4.8),
    )
    if count == 1:
        beats = [first]
    else:
        spacing = (last - first) / (count - 1)
        beats = []
        for index in range(count):
            base = first + index * spacing
            jitter = 0.0 if index in {0, count - 1} else rng.uniform(
                -timing.event_jitter_s, timing.event_jitter_s)
            beats.append(base + jitter)
        beats.sort()
        for index in range(1, len(beats)):
            beats[index] = min(
                beats[index], beats[index - 1] + timing.max_event_gap_s)
    return EpisodeRecipe.create(
        scene=scene,
        seed=int(seed),
        duration_s=timing.duration_s,
        fps=config.render.fps,
        beat_times_s=tuple(round(value, 4) for value in beats),
    )


def compile_episode_plan(
    profile: SceneProfile,
    seed: int,
    config: BenchmarkConfig,
) -> EpisodePlan:
    recipe = make_recipe(profile.scene, seed, config)
    mobility = plan_mobility(profile, recipe, config)
    event_program = plan_events(profile, mobility, recipe, config)
    camera = plan_camera(profile, mobility, event_program, recipe, config)
    trajectory = compile_trajectory(mobility, camera, recipe)
    return EpisodePlan.create(
        config_digest=config.digest,
        profile_id=profile.profile_id,
        recipe=recipe,
        mobility=mobility,
        event_program=event_program,
        camera=camera,
        trajectory=trajectory,
    )
