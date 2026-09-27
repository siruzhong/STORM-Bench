"""Stable domain contracts shared by all benchmark-generation stages."""

from .config import BenchmarkConfig, load_config
from .contracts import *  # noqa: F401,F403
from .failures import RetryScope, StageFailure

__all__ = ["BenchmarkConfig", "RetryScope", "StageFailure", "load_config"]
