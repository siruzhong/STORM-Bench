#!/usr/bin/env python3
"""Build frozen room-specific programs from a saved native scene catalog."""
import argparse
import json
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storm_virtualhome.room_profiles import build_profile
from storm_virtualhome.planning import create_program
from catalog_scenes import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[1] / "configs/randomized.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    template = yaml.safe_load(args.template.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report = []
    for item in json.loads(args.catalog.read_text()):
        if not item["available"]:
            continue
        graph = json.loads((args.catalog.parent / item["graph"]).read_text())
        for room in summarize(graph):
            name = f"scene_{item['scene']}_{room['name']}_{room['id']}"
            entry = dict(profile=name, scene=item["scene"], room_id=room["id"], room_name=room["name"], accepted=False)
            try:
                config = build_profile(graph, template, item["scene"], room["id"], args.seed)
                plan = create_program(graph, config)
            except ValueError as exc:
                entry.update(status="unsupported_recipe", reason=str(exc))
            else:
                folder = args.output / name
                folder.mkdir()
                (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
                (folder / "event_program.json").write_text(json.dumps(plan, indent=2))
                entry.update(status="planned", plan_id=plan["plan_id"])
            report.append(entry)
    (args.output / "profiles.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
