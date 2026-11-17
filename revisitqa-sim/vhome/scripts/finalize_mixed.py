#!/usr/bin/env python3
"""Audit a completed mixed capture and build its QA and debug video."""

import argparse
from pathlib import Path

from rollout_mixed import finalize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    finalize(parser.parse_args().episode.resolve())
