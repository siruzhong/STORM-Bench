#!/usr/bin/env python3
"""Validate selected frozen room programs and report coverage without substitution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def snapshot_source(project, source):
    """Copy release code only; remote project roots also contain large captures."""
    stage=source.with_name('.'+source.name+'.staging')
    if source.exists():raise FileExistsError(source)
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        for name in ("src", "scripts", "configs", "docs", "unity_debug", "tests"):
            shutil.copytree(project / name, stage / name,
                            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"))
        for name in (".gitignore", "pyproject.toml", "README.md", "requirements-tested.txt", "FILES.sha256"):
            shutil.copy2(project / name, stage / name)
        os.replace(stage,source)
    except BaseException:
        if stage.exists():shutil.rmtree(stage)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--profiles", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--xorg-root", type=Path)
    p.add_argument("--gpu-index", type=int, default=3)
    p.add_argument("--max-attempts", type=int, default=1)
    args = p.parse_args()
    if args.max_attempts < 1 or len(set(args.profiles)) != len(args.profiles):
        p.error("Use positive attempts and distinct profile names")
    profiles = {e["profile"]: e for e in json.loads((args.inputs / "profiles.json").read_text())}
    for name in args.profiles:
        if name not in profiles or profiles[name]["status"] != "planned":
            p.error(f"No planned profile: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.output / "source"
    project = Path(__file__).resolve().parents[1]
    snapshot_source(project, source)
    hashes = {str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in source.rglob("*") if f.is_file()}
    (args.output / "source_hashes.json").write_text(json.dumps(hashes, indent=2))
    status = dict(status="running", started_at=time.time(), profiles=[])
    def save():
        status["updated_at"] = time.time()
        temp = args.output / "status.tmp"
        temp.write_text(json.dumps(status, indent=2))
        temp.replace(args.output / "status.json")
    save()
    for name in args.profiles:
        entry = dict(profiles[name], status="running", accepted=False)
        status["profiles"].append(entry)
        save()
        command = [sys.executable, str(source / "scripts/rollout_retry.py"),
                   "--config", str((args.inputs / name / "config.yaml").resolve()),
                   "--plan", str((args.inputs / name / "event_program.json").resolve()),
                   "--output", str((args.output / name).resolve()),
                   "--executable", str(args.executable.resolve()),
                   "--max-attempts", str(args.max_attempts), "--gpu-index", str(args.gpu_index)]
        if args.xorg_root:
            command += ["--xorg-root", str(args.xorg_root.resolve())]
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        report_path = args.output / name / "attempts.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        entry.update(status="accepted" if result.returncode == 0 else "rejected",
                     accepted=result.returncode == 0, accepted_episode=report.get("accepted_episode"),
                     exit_code=result.returncode)
        if result.returncode:
            logs = sorted((args.output / name).glob("attempt_*.log"))
            entry["failure"] = logs[-1].read_text()[-3000:] if logs else "See driver log"
        save()
        print(json.dumps(entry), flush=True)
    status["status"] = "complete" if all(e["accepted"] for e in status["profiles"]) else "completed_with_rejections"
    save()


if __name__ == "__main__":
    main()
