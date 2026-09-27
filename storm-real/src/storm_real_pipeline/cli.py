"""Run the complete pipeline or one stage with explicit input/output paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from pathlib import Path

from .client import VLMClient
from .domains import DOMAINS, get_domain
from .io_utils import write_json

logger = logging.getLogger(__name__)
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def discover_videos(input_path: Path, output_dir: Path | None = None) -> list[Path]:
    input_path = Path(input_path).resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Input must be a supported video file")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError("Input video or directory does not exist")
    excluded = Path(output_dir).resolve() if output_dir is not None else None
    videos = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        and (excluded is None or not path.resolve().is_relative_to(excluded))
    )
    if not videos:
        raise ValueError("No supported input videos found")
    return videos


def anonymous_video_id(path: Path, input_path: Path) -> str:
    """Keep IDs stable across reruns and relocation of the input directory."""
    root = input_path.resolve()
    logical_path = path.parent.resolve() / path.name
    key = logical_path.relative_to(root).as_posix() if root.is_dir() else path.name
    return "video_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError("Artifact ID must contain only letters, digits, underscores or hyphens")
    return value


def make_client() -> VLMClient:
    return VLMClient()


def merge_questions(paths: list[Path], output: Path) -> int:
    """Write only question rows, rejecting duplicates or malformed documents."""
    from .qa import validate_question

    seen: set[str] = set()
    rows = []
    for path in sorted(set(paths)):
        document = json.loads(path.read_text(encoding="utf-8"))
        questions = document.get("questions") if isinstance(document, dict) else None
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"QA document has no question list: {path.name}")
        episode_id = _safe_id(document.get("episode_id"))
        domain = get_domain(document.get("domain", "cook")).name
        if document.get("schema_version") not in {None, get_domain(domain).qa_schema}:
            raise ValueError("QA schema does not match its domain")
        duration = document.get("duration_sec")
        timestamps = document.get("sample_timestamps")
        for question in questions:
            if not isinstance(question, dict):
                raise ValueError("Question must be an object")
            identifier = _safe_id(question.get("id"))
            canonical = validate_question(question, episode_id, duration, timestamps, domain=domain)
            if question.get("episode_id") != episode_id or canonical is None:
                raise ValueError("Invalid question schema or temporal support: " + identifier)
            if identifier in seen:
                raise ValueError("Duplicate question ID in merge: " + identifier)
            seen.add(identifier)
            canonical["id"] = identifier
            if "change_intensity" in question:
                intensity = question["change_intensity"]
                if type(intensity) is not int or intensity < 1:
                    raise ValueError("change_intensity must be a positive visit count")
                canonical["change_intensity"] = intensity
            rows.append(canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return len(rows)


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and positive")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Value must be finite and nonnegative")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def _common(parser: argparse.ArgumentParser) -> None:
    _domain_option(parser)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing artifacts")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default=None,
                        help="Optional provider-specific thinking extension")


def _domain_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", choices=tuple(DOMAINS), default="cook",
                        help="English domain workflow (default: cook)")


def _seg_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-sec", type=_positive_float, default=300.0)
    parser.add_argument("--min-tail-sec", type=_nonnegative_float, default=45.0)
    parser.add_argument("--sample-fps", type=_positive_float, default=1.0,
                        help="Segmentation sampling rate; the run command keeps QA at 1 FPS")
    parser.add_argument("--max-frames-per-call", type=_positive_int, default=64)


def _revisit_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--region", help="Select a named region from the chosen domain")
    parser.add_argument("--target-sec", type=_positive_float, default=45.0)
    parser.add_argument("--max-total-sec", type=_positive_float, default=60.0)
    parser.add_argument("--speed", type=_positive_float, default=1.5)
    parser.add_argument("--blank-sec", type=_nonnegative_float, default=0.2)
    parser.add_argument("--fade-sec", type=_nonnegative_float, default=0.2)
    parser.add_argument("--no-llm-selection", action="store_true",
                        help="Use the deterministic revisit proposal only")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Construct same-region video episodes and grounded QA")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run all three stages and merge question rows")
    run.add_argument("input", type=Path, help="A video or a directory searched recursively")
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    _common(run)
    _seg_options(run)
    _revisit_options(run)
    segment = commands.add_parser("segment", help="Create English segmentation records")
    segment.add_argument("input", type=Path)
    segment.add_argument("--output-dir", type=Path, default=Path("outputs"))
    _common(segment)
    _seg_options(segment)
    assemble = commands.add_parser("assemble", help="Assemble one segmentation window")
    assemble.add_argument("segmentation", type=Path)
    assemble.add_argument("--video", type=Path, required=True, help="Original source video")
    assemble.add_argument("--output-dir", type=Path, help="Default: outputs/<domain>/episodes")
    assemble.add_argument("--overwrite", action="store_true")
    _domain_option(assemble)
    _revisit_options(assemble)
    qa = commands.add_parser("qa", help="Generate QA from one assembled episode")
    qa.add_argument("video", type=Path)
    qa.add_argument("--plan", type=Path, help="Defaults to the adjacent .plan.json file")
    qa.add_argument("--episode-id", help="Defaults to the episode ID in its plan")
    qa.add_argument("--output-dir", type=Path, help="Default: outputs/<domain>/qa")
    qa.add_argument("--sample-fps", type=_positive_float, default=1.0)
    _common(qa)
    merge = commands.add_parser("merge", help="Merge per-episode QA JSON into JSONL")
    merge.add_argument("input", type=Path, nargs="+", help="One or more directories containing only QA JSON files")
    merge.add_argument("--output", type=Path, default=Path("outputs/questions.jsonl"))
    return parser


def _seg_kwargs(args: argparse.Namespace) -> dict:
    if args.min_tail_sec > args.window_sec:
        raise ValueError("min-tail-sec must not exceed window-sec")
    return dict(window_sec=args.window_sec, min_tail_sec=args.min_tail_sec,
                sample_fps=args.sample_fps, max_frames_per_call=args.max_frames_per_call,
                thinking=args.thinking, overwrite=args.overwrite, domain=args.domain)


def _revisit_kwargs(args: argparse.Namespace) -> dict:
    spec = get_domain(args.domain)
    if args.region is not None and args.region not in spec.regions - {"other_area"}:
        raise ValueError(f"Region must belong to {spec.name}: " + ", ".join(sorted(spec.regions - {"other_area"})))
    if args.target_sec > args.max_total_sec:
        raise ValueError("target-sec must not exceed max-total-sec")
    return dict(region=args.region, target_sec=args.target_sec,
                max_total_sec=args.max_total_sec, speed=args.speed,
                blank_sec=args.blank_sec, fade_sec=args.fade_sec, overwrite=args.overwrite,
                domain=args.domain)


def _run_sources(args: argparse.Namespace) -> int:
    from .segmentation import segment_source

    output_root = args.output_dir.resolve()
    output_dir = output_root / args.domain
    videos = discover_videos(args.input, output_root)
    seg_kwargs = _seg_kwargs(args)
    revisit_kwargs = _revisit_kwargs(args) if args.command == "run" else {}
    client = make_client()
    qa_paths = []
    failures = []
    completed_windows = 0
    for source in videos:
        video_id = args.domain + "_" + anonymous_video_id(source, args.input)
        logger.info("Processing %s", video_id)
        try:
            records = segment_source(source, output_dir / "segments", client,
                                     video_id=video_id, **seg_kwargs)
        except Exception as error:
            logger.error("Segmentation failed for %s: %s", video_id, error)
            failures.append({"id": video_id, "stage": "segment", "error_type": type(error).__name__})
            continue
        if args.command == "segment":
            completed_windows += len(records)
            continue
        from .qa import generate_episode_qa
        from .revisit import construct_episode

        for record in records:
            window_id = _safe_id(json.loads(record.read_text(encoding="utf-8"))["video_id"])
            episode_id = window_id + "_revisit"
            episode_path = output_dir / "episodes" / (episode_id + ".mp4")
            stage = "assemble"
            try:
                construct_episode(record, source, episode_path,
                                  None if args.no_llm_selection else client, **revisit_kwargs)
                stage = "qa"
                qa_path = output_dir / "qa" / (episode_id + ".json")
                generate_episode_qa(episode_path, qa_path, client, episode_id=episode_id,
                                    plan_path=episode_path.with_suffix(".plan.json"),
                                    thinking=args.thinking, overwrite=args.overwrite, domain=args.domain)
                qa_paths.append(qa_path)
                completed_windows += 1
            except Exception as error:
                logger.error("%s failed for %s: %s", stage, episode_id, error)
                failures.append({"id": episode_id, "stage": stage, "error_type": type(error).__name__})
    question_count = None
    if args.command == "run" and qa_paths:
        question_count = merge_questions(qa_paths, output_dir / "questions.jsonl")
    summary = {"command": args.command, "domain": args.domain, "source_count": len(videos),
               "completed_windows": completed_windows, "question_count": question_count,
               "failures": failures}
    write_json(output_dir / ("run_summary.json" if args.command == "run" else "segment_summary.json"), summary)
    logger.info("Completed %s windows; %s failures", completed_windows, len(failures))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    try:
        if args.command in {"run", "segment"}:
            return _run_sources(args)
        if args.command == "assemble":
            from .revisit import construct_episode

            meta = json.loads(args.segmentation.read_text(encoding="utf-8"))
            episode_id = _safe_id(meta["video_id"]) + "_revisit"
            output_dir = args.output_dir or Path("outputs") / args.domain / "episodes"
            revisit_kwargs = _revisit_kwargs(args)
            construct_episode(args.segmentation, args.video,
                              output_dir / (episode_id + ".mp4"),
                              None if args.no_llm_selection else make_client(), **revisit_kwargs)
        elif args.command == "qa":
            from .qa import generate_episode_qa

            plan = args.plan or args.video.with_suffix(".plan.json")
            if args.plan is not None and not plan.is_file():
                raise ValueError("The explicitly supplied revisit plan does not exist")
            episode_id = args.episode_id
            if episode_id is None and plan.is_file():
                episode_id = json.loads(plan.read_text(encoding="utf-8")).get("episode_id")
            if episode_id is None:
                raise ValueError("Supply --episode-id when no episode plan is available")
            episode_id = _safe_id(episode_id)
            output_dir = args.output_dir or Path("outputs") / args.domain / "qa"
            generate_episode_qa(args.video, output_dir / (episode_id + ".json"), make_client(),
                                episode_id=episode_id, plan_path=plan if plan.is_file() else None,
                                sample_fps=args.sample_fps, thinking=args.thinking, overwrite=args.overwrite,
                                domain=args.domain)
        elif args.command == "merge":
            paths = sorted({path for directory in args.input for path in directory.rglob("*.json")})
            if not paths:
                raise ValueError("No per-episode QA JSON files found")
            count = merge_questions(paths, args.output)
            logger.info("Merged %s questions", count)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        logger.error("%s", error)
        return 1
