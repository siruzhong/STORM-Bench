"""CLI for exporting or validating a StreamEQA-compatible dataset."""
from __future__ import annotations

import argparse
import json

from .exporter import export_run, validate_dataset
from .keyclip_exporter import export_keyclip_run, validate_keyclip_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Export an accepted Benchgen run in StreamEQA format.",
    )
    parser.add_argument("--run", help="accepted Benchgen run directory")
    parser.add_argument("--output", required=True, help="StreamEQA dataset root")
    parser.add_argument(
        "--layout", choices=("full", "keyclips"), default="full",
        help="export one full video or reference-style key clips per Floor",
    )
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument(
        "--link-mode", choices=("hardlink", "copy"), default="hardlink",
        help="materialize the normal rollout video without exporting audit video",
    )
    parser.add_argument(
        "--speed", type=float, default=1.5,
        help="keyclips layout playback speed",
    )
    parser.add_argument(
        "--blank-sec", type=float, default=0.2,
        help="keyclips layout gap recorded in the full-episode clip plan",
    )
    parser.add_argument(
        "--target-clips", type=int, default=6,
        help="target number of key clips per Floor (clamped to 4–8)",
    )
    parser.add_argument(
        "--context-sec", type=float, default=0.75,
        help="source context retained before and after key events",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate an existing --output dataset without exporting",
    )
    args = parser.parse_args(argv)
    if not args.validate_only and not args.run:
        parser.error("--run is required unless --validate-only is set")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        if args.layout == "keyclips":
            summary = validate_keyclip_dataset(args.output)
        else:
            summary = validate_dataset(args.output)
    elif args.layout == "keyclips":
        summary = export_keyclip_run(
            args.run,
            args.output,
            sample_fps=args.sample_fps,
            speed=args.speed,
            blank_sec=args.blank_sec,
            target_clip_count=args.target_clips,
            context_sec=args.context_sec,
        )
    else:
        summary = export_run(
            args.run,
            args.output,
            sample_fps=args.sample_fps,
            link_mode=args.link_mode,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
