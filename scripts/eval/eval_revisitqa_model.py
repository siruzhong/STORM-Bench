#!/usr/bin/env python
"""Evaluate one registered model on RevisitQA Offline or Online frame-list QA."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import dataset_manifest, get_sample_media_name, load_records, normalize_mcq_sample
from scripts.eval.revisitqa_model_adapters import MCQ_PROMPT_VERSION, load_adapter
from scripts.eval.revisitqa_model_registry import MODEL_SPECS
from scripts.eval.revisitqa_reliability import format_headline, score_selective
from scripts.eval.revisitqa_streaming import (
    CLOCK_FPS,
    MODEL_INPUT,
    NATIVE_RECURRENT_STATE,
    PROTOCOL_NAME,
    EpisodeClock,
    blank_visible_frames,
    frames_to_timestamps,
    select_episode_subset,
    shuffle_visible_frames,
)


def extract_choice(text: str, option_count: int = 4) -> str | None:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:option_count]
    patterns = (
        rf"(?is)<answer>.*?\b([{allowed}])\b.*?</answer>",
        rf"(?i)(?:answer|答案|选项|choice)\s*[:：]?\s*\(?([{allowed}])\)?",
        rf"(?i)^\s*\(?([{allowed}])\)?(?:\s|$|[。,.，])",
        rf"(?i)\b([{allowed}])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text.strip())
        if match:
            return match.group(1).upper()
    return None


def resolve_video(video_root: Path, sample: dict) -> Path:
    path = video_root / get_sample_media_name(sample)
    if path.is_file():
        return path
    raise FileNotFoundError(f"Missing RevisitQA video: {path}")


def grouped_chunk(rows: list[dict], num_chunks: int, chunk_idx: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(get_sample_media_name(row), []).append(row)
    ordered = list(groups.values())
    for group in ordered:
        group.sort(key=lambda row: float(row.get("query_time", 0.0)))
    return [row for index, group in enumerate(ordered) if index % num_chunks == chunk_idx for row in group]


def iter_video_groups(rows: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_key = None
    for row in rows:
        key = get_sample_media_name(row)
        if current and key != current_key:
            groups.append(current)
            current = []
        current.append(row)
        current_key = key
    if current:
        groups.append(current)
    return groups


def protocol_label(
    protocol: str,
    buffer_frames: int,
    text_only: bool = False,
    shuffle_frames: bool = False,
    blank_frames: bool = False,
    options_only: bool = False,
    forced_choice: bool = False,
    forced_video: bool = False,
) -> str:
    if forced_video:
        return "online_all_forced_choice"
    if forced_choice:
        return "text_forced_choice"
    if options_only:
        return "options_only"
    if text_only:
        return "offline_text_only" if protocol == "offline" else "online_all_text_only"
    if protocol == "offline":
        label = "offline"
    else:
        label = "online_all" if buffer_frames <= 0 else f"online_b{buffer_frames}"
    if shuffle_frames:
        label += "_shuffle"
    if blank_frames:
        label += "_blank"
    return label


def build_eval_signature(
    args,
    model_path: Path,
    dataset_sha256: str | None = None,
    manifest: dict | None = None,
) -> tuple[dict, str]:
    data = {
        "schema_version": PROTOCOL_NAME,
        "protocol": args.protocol,
        "protocol_label": protocol_label(
            args.protocol,
            args.buffer_frames,
            bool(getattr(args, "text_only", False)),
            bool(getattr(args, "shuffle_frames", False)),
            bool(getattr(args, "blank_frames", False)),
            bool(getattr(args, "options_only", False)),
            bool(getattr(args, "forced_choice", False)),
            bool(getattr(args, "forced_video", False)),
        ),
        "clock_fps": args.fps,
        "buffer_policy": "sliding_window",
        "buffer_frames": None if args.protocol == "offline" or args.buffer_frames <= 0 else args.buffer_frames,
        "max_pixels": args.max_pixels,
        "model_input": MODEL_INPUT,
        "temporal_resample": False,
        "native_recurrent_state": NATIVE_RECURRENT_STATE,
        "question_history_reused": False,
        "model_key": args.model_key,
        "model_path": str(model_path.resolve()),
        "question_normalization_version": "v2_preserve_question_stem",
        "mcq_prompt_version": MCQ_PROMPT_VERSION,
    }
    if getattr(args, "options_only", False):
        data["options_only"] = True
    if getattr(args, "forced_choice", False):
        data["forced_choice"] = True
        data["mcq_prompt_version"] = FORCED_PROMPT_VERSION
    if getattr(args, "forced_video", False):
        data["forced_choice"] = True
        data["mcq_prompt_version"] = FORCED_VIDEO_PROMPT_VERSION
    if getattr(args, "text_only", False) or getattr(args, "options_only", False) or getattr(args, "forced_choice", False):
        data["buffer_policy"] = None
        data["buffer_frames"] = None
        data["model_input"] = "text"
        data["text_only"] = True
    if getattr(args, "blank_frames", False):
        data["blank_visible_frames"] = True
    if getattr(args, "shuffle_frames", False):
        data["shuffle_visible_frames"] = True
        data["shuffle_seed"] = args.seed
    if dataset_sha256 is not None:
        data["dataset_sha256"] = dataset_sha256
    if manifest is not None:
        for key in (
            "dataset_sha256",
            "dataset_ids_sha256",
            "dataset_count",
            "dataset_episode_count",
        ):
            if key in manifest:
                data[key] = manifest[key]
    if args.model_key == "glm41v_9b":
        data["glm_thinking_budget"] = args.glm_thinking_budget
    if getattr(args, "episode_limit", None) is not None:
        data["episode_limit"] = args.episode_limit
        data["seed"] = args.seed
    if getattr(args, "sample_limit", None) is not None:
        data["sample_limit"] = args.sample_limit
    return data, json.dumps(data, sort_keys=True)


def _prediction_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate prediction ids in {path}")
    return result


def answer_with_frames(adapter, snapshot, question: str, fps: float = CLOCK_FPS) -> str:
    timestamps = frames_to_timestamps(snapshot, fps)
    return adapter.answer(snapshot, question, timestamps)


def answer_prompt(adapter, question: str, *, snapshot=None, fps: float = CLOCK_FPS, text_only: bool = False) -> str:
    if text_only:
        if not hasattr(adapter, "answer_text"):
            raise NotImplementedError("This model adapter does not support text-only evaluation")
        return adapter.answer_text(question)
    return answer_with_frames(adapter, snapshot, question, fps)


def options_only_prompt(question: str) -> str:
    """Drop the question stem (and any VirtualHome preamble), keep only the lettered options."""
    lines = question.splitlines()
    first = next(i for i, line in enumerate(lines) if re.match(r"^\(A\) ", line))
    return "Question: (not shown)\n" + "\n".join(lines[first:])


FORCED_PROMPT_VERSION = "text_forced_three_choice"
FORCED_INSTRUCTION = (
    "Answer the following three-choice question. No video is provided; choose the most likely option "
    "from the question alone. You must pick one. Output only the letter (A, B, or C); do not explain.\n\n"
)


FORCED_VIDEO_PROMPT_VERSION = "video_forced_three_choice"
FORCED_VIDEO_INSTRUCTION = (
    "Answer the following three-choice question based on the video observed so far. You must pick one. "
    "Output only the letter (A, B, or C); do not explain.\n\n"
)


def forced_choice_prompt(question: str, abstention_choice: int | None) -> str:
    """Drop the abstention option line so the model must commit to a substantive answer."""
    if abstention_choice is None:
        return question
    letter = chr(65 + int(abstention_choice))
    return "\n".join(line for line in question.splitlines() if not line.startswith(f"({letter}) "))


def _score_revisitqa_row(
    adapter,
    row: dict,
    args,
    model_path: Path,
    signature: str,
    signature_data: dict,
    *,
    snapshot,
    clock,
    text_only: bool,
) -> dict:
    started = time.perf_counter()
    question = options_only_prompt(row["question"]) if getattr(args, "options_only", False) else row["question"]
    option_count = len(row.get("options", [])) or 4
    if getattr(args, "forced_choice", False) or getattr(args, "forced_video", False):
        question = forced_choice_prompt(question, row.get("abstention_choice"))
        option_count -= row.get("abstention_choice") is not None
    raw_prediction = answer_prompt(
        adapter, question, snapshot=snapshot, fps=args.fps, text_only=text_only
    )
    choice = extract_choice(raw_prediction, option_count)
    prediction = ord(choice) - ord("A") if choice is not None else None
    timestamps = [] if text_only else frames_to_timestamps(snapshot, args.fps)
    query_time = row.get("query_time")
    return {
        "id": row["id"],
        "episode_id": row.get("episode_id"),
        "question": row["question"],
        "answer": row["answer"],
        "pred": raw_prediction,
        "pred_choice": choice,
        "pred_index": prediction,
        "correct": prediction == row["answer"],
        "option_permutation": row.get("option_permutation"),
        "model_key": args.model_key,
        "model_path": str(model_path.resolve()),
        "eval_signature": signature,
        "protocol": signature_data["protocol_label"],
        "query_time": query_time,
        "question_type": row.get("question_type"),
        "question_subtype": row.get("question_subtype"),
        "source_answer_index": row.get("source_answer_index"),
        "native_recurrent_state": NATIVE_RECURRENT_STATE,
        "visual_state_persistent": False if text_only else args.protocol == "online",
        "question_history_reused": False,
        "buffered_frames": 0 if text_only else len(snapshot),
        "buffer_capacity": 0 if text_only else clock.window.limit,
        "stream_frames_seen": 0 if text_only else clock.window.seen,
        "buffer_timestamps": timestamps,
        "future_leakage": (
            False
            if text_only
            else (
                query_time is not None
                and any(stamp > float(query_time) + 1e-6 for stamp in timestamps)
            )
        ),
        "window_evictions": 0 if text_only else clock.window.num_evictions,
        "latency_seconds": time.perf_counter() - started,
    }


def _log_revisitqa_row(model_key, signature_data, completed, pending_count, row, result) -> None:
    print(
        f"[{model_key}/{signature_data['protocol_label']}] "
        f"{completed}/{pending_count} {row['id']}: {result.get('pred_choice') or 'INVALID'}",
        flush=True,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=MODEL_SPECS, required=True)
    parser.add_argument("--checkpoint-root", default="ckpt")
    parser.add_argument("--protocol", choices=["offline", "online"], required=True)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--shuffle-frames",
        action="store_true",
        help="Permute the visible causal prefix and restamp timestamps.",
    )
    parser.add_argument(
        "--forced-video",
        action="store_true",
        help="Video control without the abstention option: the model must pick a substantive answer.",
    )
    parser.add_argument(
        "--forced-choice",
        action="store_true",
        help="Text-only control without the abstention option: the model must pick a substantive answer.",
    )
    parser.add_argument(
        "--options-only",
        action="store_true",
        help="Closed-book control without the question stem: only the four options are shown.",
    )
    parser.add_argument(
        "--blank-frames",
        action="store_true",
        help="Replace every visible frame with a black frame (same count, size and timestamps).",
    )
    parser.add_argument("--gt-file", default="data/eval_video/RevisitQA/revisitqa_cook/cook_question_answer.jsonl")
    parser.add_argument("--video-root", default="data/eval_video/RevisitQA/revisitqa_cook/video")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=CLOCK_FPS)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=0,
        help="Online sliding-window capacity. 0 keeps the whole causal clock stream.",
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--glm-thinking-budget", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_chunks <= 0 or not 0 <= args.chunk_idx < args.num_chunks:
        parser.error("--chunk-idx must be in [0, --num-chunks)")
    if args.sample_limit is not None and args.sample_limit <= 0:
        parser.error("--sample-limit must be positive")
    if args.episode_limit is not None and args.episode_limit <= 0:
        parser.error("--episode-limit must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.buffer_frames < 0:
        parser.error("--buffer-frames must be >= 0")
    if args.glm_thinking_budget <= 0:
        parser.error("--glm-thinking-budget must be positive")
    if sum(map(bool, (args.text_only, args.shuffle_frames, args.blank_frames, args.options_only, args.forced_choice,
                      args.forced_video))) > 1:
        parser.error("--text-only, --options-only, --shuffle-frames and --blank-frames are mutually exclusive")
    return args


def run_revisitqa_protocol(args: argparse.Namespace) -> dict | None:
    spec = MODEL_SPECS[args.model_key]
    model_path = spec.local_path(args.checkpoint_root)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Model is not downloaded: {model_path}")

    gt_path = Path(args.gt_file)
    manifest = dataset_manifest(gt_path)
    rows = [normalize_mcq_sample(row) for row in load_records(gt_path)]
    if args.episode_limit is not None:
        rows = select_episode_subset(rows, args.episode_limit, args.seed)
    rows = grouped_chunk(rows, args.num_chunks, args.chunk_idx)
    if args.sample_limit is not None:
        rows = rows[: args.sample_limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"pred_{args.num_chunks}_{args.chunk_idx}.jsonl"
    signature_data, signature = build_eval_signature(
        args, model_path, manifest["dataset_sha256"], manifest
    )

    # More workers than videos (e.g. 32 chunks vs 28 sim FloorPlans) leaves some
    # shards empty. Write the file and exit 0 so merge can still run.
    if not rows:
        if args.overwrite or not pred_path.exists():
            pred_path.write_text("", encoding="utf-8")
        print(
            f"[{args.model_key}/{signature_data['protocol_label']}] "
            f"empty chunk {args.chunk_idx}/{args.num_chunks}, skip",
            flush=True,
        )
        return None

    existing = {} if args.overwrite else _prediction_map(pred_path)
    if any(row.get("eval_signature") != signature for row in existing.values()):
        raise RuntimeError(f"Existing predictions use a different configuration: {pred_path}")

    pending_ids = {str(row["id"]) for row in rows} - set(existing)
    text_only = bool(getattr(args, "text_only", False) or getattr(args, "options_only", False)
                     or getattr(args, "forced_choice", False))
    if pending_ids:
        if getattr(args, "forced_choice", False) or getattr(args, "forced_video", False):
            import scripts.eval.revisitqa_model_adapters as adapters_module

            adapters_module.MCQ_INSTRUCTION = (
                FORCED_VIDEO_INSTRUCTION if getattr(args, "forced_video", False) else FORCED_INSTRUCTION
            )
        adapter = load_adapter(spec, model_path, args.fps, args.max_pixels)
        if args.model_key == "glm41v_9b":
            adapter.THINKING_BUDGET = args.glm_thinking_budget
        if text_only and not hasattr(adapter, "answer_text"):
            raise NotImplementedError(f"{args.model_key} does not support text-only evaluation")
        mode = "w" if args.overwrite else "a"
        completed = 0
        with pred_path.open(mode, encoding="utf-8") as handle:
            if text_only:
                for row in rows:
                    if str(row["id"]) not in pending_ids:
                        continue
                    completed += 1
                    result = _score_revisitqa_row(
                        adapter,
                        row,
                        args,
                        model_path,
                        signature,
                        signature_data,
                        snapshot=None,
                        clock=None,
                        text_only=True,
                    )
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing[str(row["id"])] = result
                    _log_revisitqa_row(args.model_key, signature_data, completed, len(pending_ids), row, result)
            else:
                for group in iter_video_groups(rows):
                    if all(str(row["id"]) not in pending_ids for row in group):
                        continue
                    clock = EpisodeClock(
                        resolve_video(Path(args.video_root), group[0]),
                        args.fps,
                        0 if args.protocol == "offline" else args.buffer_frames,
                    )
                    for row in group:
                        snapshot = clock.observe(args.protocol, row.get("query_time"))
                        if getattr(args, "shuffle_frames", False):
                            snapshot = shuffle_visible_frames(
                                snapshot, args.seed, str(row["id"]), args.fps
                            )
                        if getattr(args, "blank_frames", False):
                            snapshot = blank_visible_frames(snapshot)
                        if str(row["id"]) not in pending_ids:
                            continue
                        completed += 1
                        result = _score_revisitqa_row(
                            adapter,
                            row,
                            args,
                            model_path,
                            signature,
                            signature_data,
                            snapshot=snapshot,
                            clock=clock,
                            text_only=False,
                        )
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                        existing[str(row["id"])] = result
                        _log_revisitqa_row(args.model_key, signature_data, completed, len(pending_ids), row, result)
                    del clock

    ordered = [existing[str(row["id"])] for row in rows if str(row["id"]) in existing]
    if args.num_chunks == 1:
        merged = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
        pred_path.write_text(merged, encoding="utf-8")
        (output_dir / "pred.jsonl").write_text(merged, encoding="utf-8")
    if args.num_chunks != 1 or not ordered:
        return None

    gt_rows = [row for row in rows if str(row["id"]) in existing]
    score = score_selective(
        gt_rows,
        ordered,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed,
    )
    score.update(manifest)
    from scripts.eval.revisitqa_streaming import streaming_audit

    audit = streaming_audit(ordered, args.protocol)
    if audit:
        score["streaming_audit"] = audit
    score["eval_signature"] = signature
    (output_dir / "result.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(f"[result] {args.model_key}/{signature_data['protocol_label']}: {format_headline(score)}", flush=True)
    return score


def main(argv: list[str] | None = None) -> None:
    run_revisitqa_protocol(parse_args(argv))


if __name__ == "__main__":
    main()
