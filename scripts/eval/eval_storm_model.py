#!/usr/bin/env python
"""Evaluate one registered model on STORM-Bench Offline or Online frame-list QA."""

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
from scripts.eval.storm_model_adapters import load_adapter
from scripts.eval.storm_model_registry import MODEL_SPECS
from scripts.eval.storm_streaming import (
    CLOCK_FPS,
    MODEL_INPUT,
    NATIVE_RECURRENT_STATE,
    PROTOCOL_NAME,
    EpisodeClock,
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
    raise FileNotFoundError(f"Missing STORM-Bench video: {path}")


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
) -> str:
    if text_only:
        return "offline_text_only" if protocol == "offline" else "online_all_text_only"
    if protocol == "offline":
        return "offline_shuffle" if shuffle_frames else "offline"
    if buffer_frames <= 0:
        return "online_all_shuffle" if shuffle_frames else "online_all"
    label = f"online_b{buffer_frames}"
    return f"{label}_shuffle" if shuffle_frames else label


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
        "with_status": bool(getattr(args, "with_status", False)),
    }
    if getattr(args, "text_only", False):
        data["buffer_policy"] = None
        data["buffer_frames"] = None
        data["model_input"] = "text"
        data["text_only"] = True
    if getattr(args, "shuffle_frames", False):
        data["shuffle_visible_frames"] = True
        data["shuffle_seed"] = args.seed
    if data["with_status"]:
        data["epistemic_probe_version"] = "v2_multilabel_causes"
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


def _score_storm_row(
    adapter,
    row: dict,
    args,
    model_path: Path,
    signature: str,
    signature_data: dict,
    status_helper,
    *,
    snapshot,
    clock,
    text_only: bool,
) -> dict:
    started = time.perf_counter()
    raw_prediction = answer_prompt(
        adapter, row["question"], snapshot=snapshot, fps=args.fps, text_only=text_only
    )
    choice = extract_choice(raw_prediction, len(row.get("options", [])) or 4)
    prediction = ord(choice) - ord("A") if choice is not None else None
    raw_status = None
    pred_status = None
    raw_source = None
    pred_sources: list[str] = []
    if status_helper is not None:
        (
            build_status_question,
            extract_status,
            statuses,
            build_source_question,
            extract_sources,
        ) = status_helper
        raw_status = answer_prompt(
            adapter,
            build_status_question(row["question"], statuses),
            snapshot=snapshot,
            fps=args.fps,
            text_only=text_only,
        )
        pred_status = extract_status(raw_status, statuses)
        if pred_status == "uncertain":
            raw_source = answer_prompt(
                adapter,
                build_source_question(row["question"]),
                snapshot=snapshot,
                fps=args.fps,
                text_only=text_only,
            )
            pred_sources = extract_sources(raw_source)
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
        "pred_epistemic_raw": raw_status,
        "pred_epistemic_status": pred_status,
        "pred_uncertainty_raw": raw_source,
        "pred_uncertainty_sources": pred_sources,
        "pred_uncertainty_source": pred_sources[0] if pred_sources else None,
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


def _log_storm_row(model_key, signature_data, completed, pending_count, row, result, status_helper) -> None:
    pred_status = result.get("pred_epistemic_status")
    pred_sources = result.get("pred_uncertainty_sources") or []
    print(
        f"[{model_key}/{signature_data['protocol_label']}] "
        f"{completed}/{pending_count} {row['id']}: {result.get('pred_choice') or 'INVALID'}"
        + (f" status={pred_status or 'INVALID'}" if status_helper is not None else "")
        + (
            f" sources={','.join(pred_sources) or 'INVALID'}"
            if pred_status == "uncertain"
            else ""
        ),
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
    parser.add_argument("--gt-file", default="data/eval_video/storm_real/qa_results/questions.jsonl")
    parser.add_argument("--video-root", default="data/eval_video/storm_real/video")
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
    parser.add_argument("--with-status", action="store_true")
    parser.add_argument("--glm-thinking-budget", type=int, default=512)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.buffer_frames < 0:
        parser.error("--buffer-frames must be >= 0")
    if args.glm_thinking_budget <= 0:
        parser.error("--glm-thinking-budget must be positive")
    if args.text_only and args.shuffle_frames:
        parser.error("--text-only and --shuffle-frames cannot be combined")
    return args


def run_storm_protocol(args: argparse.Namespace) -> dict | None:
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

    status_helper = None
    if args.with_status:
        from scripts.eval.eval_storm_diagnostics import (
            build_source_question,
            build_status_question,
            extract_sources,
            extract_status,
        )
        from scripts.eval.storm_diagnostics import audit_ground_truth, epistemic_statuses

        audit_ground_truth(rows)
        statuses = epistemic_statuses(rows)
        status_helper = (
            build_status_question,
            extract_status,
            statuses,
            build_source_question,
            extract_sources,
        )

    pending_ids = {str(row["id"]) for row in rows} - set(existing)
    text_only = bool(getattr(args, "text_only", False))
    if pending_ids:
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
                    result = _score_storm_row(
                        adapter,
                        row,
                        args,
                        model_path,
                        signature,
                        signature_data,
                        status_helper,
                        snapshot=None,
                        clock=None,
                        text_only=True,
                    )
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing[str(row["id"])] = result
                    _log_storm_row(args.model_key, signature_data, completed, len(pending_ids), row, result, status_helper)
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
                        if str(row["id"]) not in pending_ids:
                            continue
                        completed += 1
                        result = _score_storm_row(
                            adapter,
                            row,
                            args,
                            model_path,
                            signature,
                            signature_data,
                            status_helper,
                            snapshot=snapshot,
                            clock=clock,
                            text_only=False,
                        )
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                        existing[str(row["id"])] = result
                        _log_storm_row(args.model_key, signature_data, completed, len(pending_ids), row, result, status_helper)
                    del clock

    ordered = [existing[str(row["id"])] for row in rows if str(row["id"]) in existing]
    if args.num_chunks == 1:
        merged = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered)
        pred_path.write_text(merged, encoding="utf-8")
        (output_dir / "pred.jsonl").write_text(merged, encoding="utf-8")
    if args.num_chunks != 1 or not args.with_status or not ordered:
        return None

    from scripts.eval.storm_diagnostics import attach_storm_scores

    gt_rows = [row for row in rows if str(row["id"]) in existing]
    score = attach_storm_scores(
        ordered,
        gt_rows,
        signature_data,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed,
    )
    (output_dir / "result.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    headline = score.get("storm_score")
    headline_label = "STORM-BR"
    if headline is None:
        headline = score.get("storm_score_observed")
        headline_label = "STORM-BR(observed-only)"
    print(
        f"[result] {args.model_key}: answer={score['answer_accuracy']:.2%} "
        f"known-volatility={score['known_volatility_score']:.2%} "
        f"epistemic={score['epistemic_uncertainty_score']:.2%} "
        f"{headline_label}={headline:.2%}"
        + (
            f" cause-F1={score['uncertainty_cause_macro_f1']:.2%}"
            if score.get("uncertainty_cause_macro_f1") is not None
            else ""
        ),
        flush=True,
    )
    return score


def main(argv: list[str] | None = None) -> None:
    run_storm_protocol(parse_args(argv))


if __name__ == "__main__":
    main()
