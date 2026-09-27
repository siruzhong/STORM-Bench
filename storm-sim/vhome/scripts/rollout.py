#!/usr/bin/env python3
"""Run the configured VirtualHome observer protocol."""

import argparse
from pathlib import Path

import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/randomized.yaml",
    )
    args, _ = parser.parse_known_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("event_protocol") == "mixed":
        from rollout_mixed import main
    else:
        from rollout_observer import main
    main()
