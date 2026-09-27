#!/usr/bin/env python3
"""Save native scene graphs and room capabilities without rendering actions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from storm_virtualhome.client import UnityProcess, SimulatorError
from storm_virtualhome.scene import room_id_of, support_of


def summarize(graph):
    rooms = []
    for room in graph["nodes"]:
        if room.get("category", "").lower() != "rooms":
            continue
        members = [n for n in graph["nodes"] if room_id_of(graph, n["id"]) == room["id"]]
        def selected(prop, state=None):
            return [n["id"] for n in members if prop in n.get("properties", [])
                    and (state is None or state in n.get("states", []))]
        rooms.append(dict(id=room["id"], name=room["class_name"], bounds=room["bounding_box"],
                          openables=selected("CAN_OPEN", "CLOSED"),
                          switches=selected("HAS_SWITCH", "OFF"),
                          portable=[key for key in selected("GRABBABLE") if support_of(graph, key)],
                          surfaces=selected("SURFACES")))
    return rooms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--gpu-index", type=int, default=3)
    parser.add_argument("--xorg-root", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = []
    with UnityProcess(args.executable, args.output / "logs", gpu_index=args.gpu_index,
                      xorg_root=args.xorg_root) as client:
        for scene in args.scenes:
            try:
                client.reset(scene)
                graph = client.graph()
            except SimulatorError as exc:
                result.append(dict(scene=scene, available=False, failure=str(exc)))
            else:
                path = args.output / f"scene_{scene}.json"
                path.write_text(json.dumps(graph, indent=2))
                result.append(dict(scene=scene, available=True, graph=path.name, rooms=summarize(graph)))
            (args.output / "catalog.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result[-1]), flush=True)


if __name__ == "__main__":
    main()
