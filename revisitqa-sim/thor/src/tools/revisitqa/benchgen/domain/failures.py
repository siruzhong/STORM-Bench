"""Typed pipeline failures with explicit retry ownership."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class RetryScope(str, Enum):
    NONE = "none"
    EVENT_PROGRAM = "event_program"
    MOTION_PLAN = "motion_plan"
    SCENE_PROFILE = "scene_profile"
    EPISODE_SEED = "episode_seed"
    SCENE = "scene"


@dataclass
class StageFailure(RuntimeError):
    stage: str
    code: str
    retry_scope: RetryScope
    detail: str
    context: Mapping[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.stage}:{self.code}: {self.detail}"
