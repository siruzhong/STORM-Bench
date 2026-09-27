"""Generate and export an exactly scene-balanced four-room StreamEQA set."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .reference_qa import _episode_type_schedule, _type_quotas, export_run


ROOM_RANGES = {
    "kitchen": range(1, 31),
    "living_room": range(201, 231),
    "bedroom": range(301, 331),
    "bathroom": range(401, 431),
}
ROOM_SURFACE_TYPES = {
    "kitchen": ["CounterTop", "DiningTable", "StoveBurner"],
    "living_room": [
        "CoffeeTable",
        "SideTable",
        "DiningTable",
        "Chair",
        "Sofa",
        "TVStand",
    ],
    "bedroom": ["Bed", "Desk", "Dresser", "Shelf", "SideTable"],
    "bathroom": [
        "Bathtub",
        "BathtubBasin",
        "Floor",
        "HandTowelHolder",
        "Shelf",
        "SideTable",
        "Sink",
        "SinkBasin",
        "TowelHolder",
    ],
}
ROOM_TARGET_TYPES = {
    "kitchen": [
        "Apple",
        "Bowl",
        "Bread",
        "Bottle",
        "ButterKnife",
        "Cup",
        "Egg",
        "Fork",
        "Kettle",
        "Knife",
        "Ladle",
        "Lettuce",
        "Mug",
        "Pan",
        "PepperShaker",
        "Plate",
        "Potato",
        "SaltShaker",
        "SoapBottle",
        "Spatula",
        "Spoon",
        "Tomato",
    ],
    "living_room": [
        "Book",
        "Bowl",
        "KeyChain",
        "Laptop",
        "Newspaper",
        "Pillow",
        "Plate",
        "RemoteControl",
        "Statue",
        "TissueBox",
        "Vase",
        "Watch",
    ],
    "bedroom": [
        "AlarmClock",
        "Book",
        "Bowl",
        "CellPhone",
        "Laptop",
        "Mug",
        "Pillow",
        "Statue",
        "TissueBox",
        "Watch",
    ],
    "bathroom": [
        "Candle",
        "Cloth",
        "DishSponge",
        "HandTowel",
        "PaperTowelRoll",
        "Plunger",
        "ScrubBrush",
        "SoapBar",
        "SoapBottle",
        "SprayBottle",
        "ToiletPaper",
        "Towel",
    ],
}
ROOM_HORIZONS = {
    "kitchen": [8, 12, 16, 20],
    "living_room": [8, 12, 16, 20],
    "bedroom": [8, 12, 16, 20],
    "bathroom": [8, 12, 16, 20, 25, 30],
}
ROOM_STATION_OVERRIDES = {
    "kitchen": {},
    "living_room": {
        "max_targets": 24,
        "max_position_candidates": 24,
        "max_total_stations": 18,
        "preferred_distance": 1.35,
        "min_distance": 0.70,
        "max_distance": 2.20,
    },
    "bedroom": {
        "max_targets": 24,
        "max_position_candidates": 24,
        "max_total_stations": 18,
        "preferred_distance": 1.20,
        "min_distance": 0.60,
        "max_distance": 2.00,
    },
    "bathroom": {
        "max_targets": 24,
        "max_position_candidates": 24,
        "max_total_stations": 18,
    },
}
ROOM_MOTION_OVERRIDES = {
    "kitchen": {},
    "living_room": {"max_segment_speed_mps": 2.2, "max_station_step_m": 8.0},
    "bedroom": {"max_segment_speed_mps": 1.6, "max_station_step_m": 5.0},
    "bathroom": {"max_segment_speed_mps": 1.3, "max_station_step_m": 4.0},
}
ROOM_QUALITY_OVERRIDES = {
    "kitchen": {},
    "living_room": {"min_target_pixels": 600, "min_target_dimension": 18},
    "bedroom": {"min_target_pixels": 600, "min_target_dimension": 18},
    "bathroom": {"min_target_pixels": 500, "min_target_dimension": 16},
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _room_for_scene(scene: str) -> str:
    index = int(scene.removeprefix("FloorPlan"))
    for room, room_range in ROOM_RANGES.items():
        if index in room_range:
            return room
    raise ValueError(f"unsupported iTHOR scene: {scene}")


def _episode_scene(episode_dir: Path) -> str:
    plan = json.loads((episode_dir / "episode_plan.json").read_text())
    return str(plan["recipe"]["scene"])


def _balanced_source_order(
    episodes: dict[str, list[Path]], question_count: int,
) -> tuple[list[Path], list[int]]:
    rooms = tuple(ROOM_RANGES)
    counts = {room: len(paths) for room, paths in episodes.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"room episode counts are not balanced: {counts}")
    episode_count = sum(counts.values())
    if question_count % len(rooms):
        raise ValueError("question count must divide evenly across room types")
    schedules = _episode_type_schedule(episode_count, _type_quotas(question_count))
    lengths = [len(schedule) for schedule in schedules]
    target_per_room = question_count // len(rooms)
    positions_by_length: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        positions_by_length.setdefault(length, []).append(index)

    per_room = counts[rooms[0]]
    combinations = []
    for n10 in range(per_room + 1):
        for n11 in range(per_room - n10 + 1):
            n12 = per_room - n10 - n11
            if n10 * 10 + n11 * 11 + n12 * 12 == target_per_room:
                combinations.append({10: n10, 11: n11, 12: n12})
    if not combinations:
        raise ValueError("no per-room episode-length allocation reaches exact balance")
    required = combinations[0]
    for length, count in required.items():
        if len(positions_by_length.get(length, [])) != count * len(rooms):
            raise ValueError(
                f"global schedule cannot be evenly partitioned: length={length} "
                f"available={len(positions_by_length.get(length, []))} required={count * len(rooms)}"
            )

    ordered: list[Path | None] = [None] * episode_count
    room_position_counts = {}
    for room in rooms:
        room_positions = []
        for length in (10, 11, 12):
            for _ in range(required[length]):
                room_positions.append(positions_by_length[length].pop(0))
        room_positions.sort()
        for position, episode_dir in zip(room_positions, episodes[room]):
            ordered[position] = episode_dir
        room_position_counts[room] = Counter(lengths[position] for position in room_positions)
    if any(path is None for path in ordered):
        raise AssertionError("not every export position received an episode")
    print(
        "balanced export schedule:",
        {room: dict(sorted(counts.items())) for room, counts in room_position_counts.items()},
        flush=True,
    )
    return [path for path in ordered if path is not None], lengths


def _generate_room(
    room: str,
    base_config: dict[str, Any],
    work_root: Path,
    episodes_per_room: int,
    seed: int,
) -> tuple[Path, list[Path]]:
    room_root = work_root / room
    room_root.mkdir(parents=True, exist_ok=False)
    config = json.loads(json.dumps(base_config))
    config["batch"]["episode_count"] = episodes_per_room
    config["batch"]["scenes"] = [
        f"FloorPlan{index}" for index in ROOM_RANGES[room]
    ]
    config["station"]["surface_types"] = ROOM_SURFACE_TYPES[room]
    config["station"]["target_types"] = (
        ROOM_TARGET_TYPES[room] if room == "kitchen" else []
    )
    config["station"]["horizons"] = ROOM_HORIZONS[room]
    config["station"].update(ROOM_STATION_OVERRIDES[room])
    config["motion"].update(ROOM_MOTION_OVERRIDES[room])
    config["quality"].update(ROOM_QUALITY_OVERRIDES[room])
    config_path = room_root / "effective_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    rollout_root = room_root / "rollouts"
    print(
        f"ROOM_START room={room} episodes={episodes_per_room} seed={seed}",
        flush=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.storm.benchgen.cli",
            "--config",
            str(config_path),
            "--output",
            str(rollout_root),
            "--episodes",
            str(episodes_per_room),
            "--seed",
            str(seed),
            "--log-level",
            "INFO",
        ],
        check=True,
    )
    run_dirs = sorted(rollout_root.glob("run_*"))
    if len(run_dirs) != 1:
        raise RuntimeError(f"expected one run for {room}, found {run_dirs}")
    run_dir = run_dirs[0]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    episode_dirs = [run_dir / row["directory"] for row in manifest["episodes"]]
    if len(episode_dirs) != episodes_per_room:
        raise RuntimeError(f"{room} generated {len(episode_dirs)} episodes")
    print(f"ROOM_DONE room={room} run={run_dir}", flush=True)
    return run_dir, episode_dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes-per-room", type=int, default=7)
    parser.add_argument("--question-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--reuse-run",
        action="append",
        default=[],
        metavar="ROOM=PATH",
        help="reuse an accepted room batch instead of generating it",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    work_root = Path(args.work_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if work_root.exists() or output.exists():
        raise FileExistsError("work-root and output must both be new paths")
    work_root.mkdir(parents=True)
    base_config = yaml.safe_load(Path(args.config).read_text())
    reused = {}
    for raw in args.reuse_run:
        room, separator, path = raw.partition("=")
        if not separator or room not in ROOM_RANGES or room in reused:
            raise ValueError(f"invalid --reuse-run value: {raw}")
        reused[room] = Path(path).expanduser().resolve()
    run_dirs = {}
    episodes = {}
    for room_index, room in enumerate(ROOM_RANGES):
        if room in reused:
            run_dir = reused[room]
            manifest = json.loads((run_dir / "manifest.json").read_text())
            validation = json.loads((run_dir / "batch_validation.json").read_text())
            if not manifest.get("accepted") or not validation.get("accepted"):
                raise ValueError(f"reused room run is not accepted: {run_dir}")
            room_episodes = [run_dir / row["directory"] for row in manifest["episodes"]]
            if len(room_episodes) != args.episodes_per_room:
                raise ValueError(
                    f"reused {room} run has {len(room_episodes)} episodes; "
                    f"expected {args.episodes_per_room}"
                )
            print(f"ROOM_REUSED room={room} run={run_dir}", flush=True)
        else:
            room_seed = args.seed + room_index * 100_000_007
            run_dir, room_episodes = _generate_room(
                room,
                base_config,
                work_root,
                args.episodes_per_room,
                room_seed,
            )
        run_dirs[room] = run_dir
        episodes[room] = room_episodes

    ordered, schedule_lengths = _balanced_source_order(episodes, args.question_count)
    print(f"EXPORT_START output={output}", flush=True)
    summary = export_run(
        None,
        output,
        episode_dirs=ordered,
        question_count=args.question_count,
        sample_fps=1.0,
        link_mode="hardlink",
    )
    room_video_counts = Counter()
    room_question_counts = Counter()
    for path, question_length in zip(ordered, schedule_lengths):
        room = _room_for_scene(_episode_scene(path))
        room_video_counts[room] += 1
        room_question_counts[room] += question_length
    if set(room_video_counts.values()) != {args.episodes_per_room}:
        raise AssertionError(f"video room balance failed: {room_video_counts}")
    expected_questions = args.question_count // len(ROOM_RANGES)
    if set(room_question_counts.values()) != {expected_questions}:
        raise AssertionError(f"question room balance failed: {room_question_counts}")
    four_scene_summary = {
        "dataset_dir": str(output),
        "work_root": str(work_root),
        "run_dirs_by_room": {room: str(path) for room, path in run_dirs.items()},
        "video_counts_by_room": dict(room_video_counts),
        "question_counts_by_room": dict(room_question_counts),
        "reference_aligned_summary": summary,
    }
    _write_json(output / "four_scene_summary.json", four_scene_summary)
    print(json.dumps(four_scene_summary, ensure_ascii=False, indent=2), flush=True)
    print("FOUR_SCENE_ROLLOUT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
