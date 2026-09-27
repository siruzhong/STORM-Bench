#!/usr/bin/env python3
"""Retry a frozen event program and retain only fully validated runs as accepted."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def run_attempts(args):
    if args.max_attempts < 1:
        raise ValueError("max-attempts must be positive")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    shutil.copy2(args.config, inputs / "config.yaml")
    shutil.copy2(args.plan, inputs / "event_program.json")
    program = json.loads((inputs / "event_program.json").read_text())
    attempts = []
    for index in range(args.max_attempts):
        episode = root / f"{root.name}_attempt_{index + 1:02d}"
        log = root / f"attempt_{index + 1:02d}.log"
        command = [sys.executable, str(Path(__file__).with_name("rollout.py")),
                   "--executable", str(args.executable.resolve()),
                   "--config", str(inputs / "config.yaml"),
                   "--plan", str(inputs / "event_program.json"),
                   "--output", str(episode), "--gpu-index", str(args.gpu_index)]
        if args.xorg_root:
            command += ["--xorg-root", str(args.xorg_root.resolve())]
        if args.display:
            command += ["--display", args.display]
        print(f"Attempt {index + 1}/{args.max_attempts}: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        accepted = result.returncode == 0
        if accepted:
            recorded = json.loads((episode / "event_program.json").read_text())
            execution = json.loads((episode / "plan_execution.json").read_text())
            continuity = json.loads((episode / "camera_continuity.json").read_text())
            accepted = (recorded == program and execution["accepted"]
                        and continuity["accepted"]
                        and (episode / "observer_validation.json").is_file()
                        and (episode / "debug_validation.json").is_file())
        attempts.append({"episode": episode.name, "log": log.name,
                         "exit_code": result.returncode, "accepted": bool(accepted)})
        report = {"plan_id": program["plan_id"], "attempts": attempts,
                  "accepted_episode": episode.name if accepted else None}
        (root / "attempts.json").write_text(json.dumps(report, indent=2))
        if accepted:
            print(json.dumps(report, indent=2))
            return episode
    raise RuntimeError(f"No accepted capture after {args.max_attempts} attempts; see {root / 'attempts.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--xorg-root", type=Path)
    parser.add_argument("--display")
    run_attempts(parser.parse_args())


if __name__ == "__main__":
    main()
