"""Command-line entry point for benchmark batch generation."""
from __future__ import annotations

import argparse
import logging
from dataclasses import replace

from .adapters.ai2thor import Ai2ThorSceneProfiler, Ai2ThorSession
from .domain.config import load_config
from .orchestrator.batch import BatchGenerator
from .runtime.replay import ThorReplay


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and jointly validate a benchmark episode batch.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--episodes", type=int,
        help="override batch.episode_count in the effective saved config",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.episodes is not None:
        config = replace(
            config,
            batch=replace(config.batch, episode_count=args.episodes),
        )
        config.validate()
    session = Ai2ThorSession()
    profiler = Ai2ThorSceneProfiler(session)
    replay = ThorReplay(session)
    generator = BatchGenerator(
        config=config,
        profiler=profiler,
        replay=replay,
        output_root=args.output,
    )
    try:
        result = generator.run(args.seed)
    finally:
        replay.close()
    print(result.run_dir)
    for episode_dir in result.episode_dirs:
        print(episode_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
