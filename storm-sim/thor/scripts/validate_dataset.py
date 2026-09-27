#!/usr/bin/env python3
"""Validate a reference, rule or polished dataset without invoking the simulator/API."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scripts.dataset_io import validate_bundle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.dataset), indent=2))
