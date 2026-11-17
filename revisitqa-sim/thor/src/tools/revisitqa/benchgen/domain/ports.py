"""Simulator-facing ports; pure planners depend only on these contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import BenchmarkConfig
from .contracts import EpisodePlan, ExecutionTrace, SceneProfile, ValidationReport


class SceneProfilerPort(Protocol):
    def profile(self, scene: str, config: BenchmarkConfig) -> SceneProfile: ...


class ReplayPort(Protocol):
    def preflight(
        self, plan: EpisodePlan, config: BenchmarkConfig,
    ) -> tuple[ValidationReport, ExecutionTrace]: ...

    def capture(
        self, plan: EpisodePlan, config: BenchmarkConfig, output_dir: Path,
    ) -> ExecutionTrace: ...

    def close(self) -> None: ...
