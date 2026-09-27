#!/usr/bin/env python
"""Run the independent STORM answer + epistemic-status evaluation path."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import dataset_manifest, get_sample_media_name, load_records, normalize_mcq_sample
from scripts.eval.eval_storm_model import extract_choice, iter_video_groups, protocol_label, resolve_video
from scripts.eval.storm_model_adapters import load_adapter
from scripts.eval.storm_model_registry import MODEL_SPECS
from scripts.eval.storm_streaming import CLOCK_FPS, PROTOCOL_NAME, EpisodeClock, frames_to_timestamps
from scripts.eval.storm_diagnostics import (
    UNCERTAINTY_SOURCE_CHOICES,
    UNCERTAINTY_SOURCE_LABELS,
    audit_ground_truth,
    epistemic_statuses,
    score_predictions,
    stratified_sample,
)


def build_status_question(question: str, statuses: tuple[str, ...]) -> str:
    status_choice_map(statuses)
    question_stem = re.split(r"\n\s*\(A\)\s+", question, maxsplit=1)[0].strip()
    choices = (
        "(A) Directly known: the frames show the answer directly, clearly, and uniquely",
        "(B) Known after reasoning: combining the visible evidence still uniquely determines the answer",
        "(C) Missing observation: key frames are missing, the target is not visible, or evidence is insufficient",
        "(D) Ambiguous evidence: multiple candidates fit the frames, or the evidence conflicts",
    )
    return (
        f"Question: {question_stem}\n\n"
        "Judge the epistemic status of the evidence needed to answer this question, "
        "based only on the frames seen so far. "
        "Do not answer the original question. Do not fill in off-screen information from common sense. "
        "Choose one of the following and output only the letter:\n" + "\n".join(choices)
    )


def status_choice_map(statuses: tuple[str, ...]) -> dict[str, str]:
    status_set = set(statuses)
    if status_set <= {"known", "uncertain"}:
        return {"A": "known", "B": "known", "C": "uncertain", "D": "uncertain"}
    raise ValueError(f"Unsupported epistemic label space: {statuses}")


def extract_status(text: str, statuses: tuple[str, ...]) -> str | None:
    choice = extract_choice(text, option_count=4)
    return status_choice_map(statuses).get(choice) if choice else None


def build_source_question(question: str) -> str:
    question_stem = re.split(r"\n\s*\(A\)\s+", question, maxsplit=1)[0].strip()
    letters = "ABCDEF"
    lines = [
        f"({letters[index]}) {description}"
        for index, (_label, description) in enumerate(UNCERTAINTY_SOURCE_CHOICES)
    ]
    return (
        f"Question: {question_stem}\n\n"
        "The evidence for this question is insufficient. "
        "Select all reasons that apply, based only on the frames seen so far. "
        "Do not answer the original question. Output only the corresponding letters "
        "separated by commas (for example, A,C):\n" + "\n".join(lines)
    )


def extract_sources(text: str) -> list[str]:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(UNCERTAINTY_SOURCE_LABELS)]
    answer_match = re.search(r"(?is)<answer>(.*?)</answer>", text)
    candidate = answer_match.group(1) if answer_match else text.strip()
    choices = re.findall(rf"(?i)(?<![A-Z])([{allowed}])(?![A-Z])", candidate)
    unique_choices = list(dict.fromkeys(choice.upper() for choice in choices))
    return [UNCERTAINTY_SOURCE_LABELS[ord(choice) - ord("A")] for choice in unique_choices]


def extract_source(text: str) -> str | None:
    """Return the first source for compatibility with legacy callers and files."""
    sources = extract_sources(text)
    return sources[0] if sources else None


def _load_prediction_map(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate ids in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=MODEL_SPECS, required=True)
    parser.add_argument("--protocol", choices=["offline", "online"], required=True)
    parser.add_argument("--checkpoint-root", default="ckpt")
    parser.add_argument("--gt-file", default="data/annotations/storm_cook.questions.jsonl")
    parser.add_argument("--video-root", default="data/videos/storm_cook")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-answer-pred", default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--sampling", choices=["stratified", "head"], default="stratified")
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--fps", type=float, default=CLOCK_FPS)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=0,
        help="Online sliding-window capacity. 0 = Online(B=all).",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.buffer_frames < 0:
        parser.error("--buffer-frames must be >= 0")

    gt_path = Path(args.gt_file)
    manifest = dataset_manifest(gt_path)
    all_rows = [normalize_mcq_sample(row) for row in load_records(gt_path)]
    contract = audit_ground_truth(all_rows)
    if args.sample_limit is None:
        rows = all_rows
    elif args.sampling == "stratified":
        rows = stratified_sample(all_rows, args.sample_limit, args.seed)
    else:
        rows = all_rows[: args.sample_limit]
    statuses = epistemic_statuses(rows)
    if set(statuses) != set(contract["status_labels"]):
        raise ValueError(
            "Selected subset does not cover the full epistemic label space; "
            "increase --sample-limit or use stratified sampling"
        )

    spec = MODEL_SPECS[args.model_key]
    model_path = spec.local_path(args.checkpoint_root)
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Model is not downloaded: {model_path}")
    reused = _load_prediction_map(Path(args.reuse_answer_pred)) if args.reuse_answer_pred else {}
    selected_ids = [str(row["id"]) for row in rows]
    if reused:
        missing = sorted(set(selected_ids) - set(reused))[:5]
        if missing:
            raise ValueError(f"Reused answer file is missing selected ids: {missing}")

    signature_data = {
        "schema_version": "storm_balanced_reliability_v2",
        "clock_protocol": PROTOCOL_NAME,
        "model_key": args.model_key,
        "model_path": str(model_path.resolve()),
        "protocol": args.protocol,
        "protocol_label": protocol_label(args.protocol, args.buffer_frames),
        **manifest,
        "selected_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
        "selected_count": len(rows),
        "sampling": args.sampling if args.sample_limit is not None else "full",
        "seed": args.seed,
        "status_labels": list(statuses),
        "status_prompt": "evidence_state_probe_v2_multilabel_causes",
        "clock_fps": args.fps,
        "max_pixels": args.max_pixels,
        "buffer_frames": None if args.protocol == "offline" or args.buffer_frames <= 0 else args.buffer_frames,
        "buffer_policy": "sliding_window",
        "model_input": "frame_list",
        "temporal_resample": False,
        "native_recurrent_state": False,
        "reuse_answer_pred": str(Path(args.reuse_answer_pred).resolve()) if args.reuse_answer_pred else None,
    }
    signature = json.dumps(signature_data, sort_keys=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    pred_path = output_dir / "pred.jsonl"
    if manifest_path.exists() and not args.overwrite:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("eval_signature") != signature:
            raise RuntimeError(f"Existing output has a different configuration: {output_dir}")
    manifest_path.write_text(
        json.dumps({**signature_data, "dataset_contract": contract, "eval_signature": signature}, indent=2) + "\n",
        encoding="utf-8",
    )

    existing = {}
    if pred_path.exists() and not args.overwrite:
        existing = _load_prediction_map(pred_path)
        if any(row.get("eval_signature") != signature for row in existing.values()):
            raise RuntimeError(f"Existing predictions have a different configuration: {pred_path}")
    pending_ids = {str(row["id"]) for row in rows} - set(existing)
    if pending_ids:
        adapter = load_adapter(spec, model_path, args.fps, args.max_pixels)
        mode = "w" if args.overwrite else "a"
        ordered_rows = sorted(
            rows,
            key=lambda row: (get_sample_media_name(row), float(row.get("query_time", 0.0))),
        )
        completed = 0
        with pred_path.open(mode, encoding="utf-8") as handle:
            for group in iter_video_groups(ordered_rows):
                if all(str(row["id"]) not in pending_ids for row in group):
                    continue
                clock = EpisodeClock(
                    resolve_video(Path(args.video_root), group[0]),
                    args.fps,
                    0 if args.protocol == "offline" else args.buffer_frames,
                )
                for row in group:
                    snapshot = clock.observe(args.protocol, row.get("query_time"))
                    if str(row["id"]) not in pending_ids:
                        continue
                    completed += 1
                    started = time.perf_counter()
                    reused_answer = reused.get(str(row["id"]))
                    if reused_answer is None:
                        raw_answer = adapter.answer(
                            snapshot, row["question"], frames_to_timestamps(snapshot, args.fps)
                        )
                        answer_choice = extract_choice(raw_answer, len(row.get("options", [])) or 4)
                        pred_index = ord(answer_choice) - ord("A") if answer_choice else None
                    else:
                        raw_answer = reused_answer.get("pred", "")
                        answer_choice = reused_answer.get("pred_choice")
                        pred_index = reused_answer.get("pred_index")
                        if reused_answer.get("option_permutation") != row.get("option_permutation"):
                            raise ValueError(f"Option permutation mismatch for {row['id']}")
                    raw_status = adapter.answer(
                        snapshot,
                        build_status_question(row["question"], statuses),
                        frames_to_timestamps(snapshot, args.fps),
                    )
                    pred_status = extract_status(raw_status, statuses)
                    raw_source = None
                    pred_sources = []
                    if pred_status == "uncertain":
                        raw_source = adapter.answer(
                            snapshot,
                            build_source_question(row["question"]),
                            frames_to_timestamps(snapshot, args.fps),
                        )
                        pred_sources = extract_sources(raw_source)
                    result = {
                        "id": row["id"],
                        "episode_id": row.get("episode_id"),
                        "pred": raw_answer,
                        "pred_choice": answer_choice,
                        "pred_index": pred_index,
                        "pred_epistemic_raw": raw_status,
                        "pred_epistemic_status": pred_status,
                        "pred_uncertainty_raw": raw_source,
                        "pred_uncertainty_sources": pred_sources,
                        "pred_uncertainty_source": pred_sources[0] if pred_sources else None,
                        "option_permutation": row.get("option_permutation"),
                        "model_key": args.model_key,
                        "protocol": protocol_label(args.protocol, args.buffer_frames),
                        "eval_signature": signature,
                        "latency_seconds": time.perf_counter() - started,
                    }
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing[str(row["id"])] = result
                    print(
                        f"[{args.model_key}/{args.protocol}] {completed}/{len(pending_ids)} "
                        f"{row['id']}: answer={answer_choice or 'INVALID'} "
                        f"status={pred_status or 'INVALID'}"
                        + (
                            f" sources={','.join(pred_sources) or 'INVALID'}"
                            if pred_status == "uncertain"
                            else ""
                        ),
                        flush=True,
                    )
                del clock

    ordered_predictions = [existing[sample_id] for sample_id in selected_ids]
    pred_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_predictions),
        encoding="utf-8",
    )
    result = score_predictions(
        rows,
        ordered_predictions,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.seed,
    )
    result["run"] = signature_data
    for key in ("dataset_sha256", "dataset_ids_sha256", "dataset_count", "dataset_episode_count"):
        result[key] = signature_data[key]
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    headline = result.get("storm_score")
    headline_label = "STORM-BR"
    if headline is None:
        headline = result.get("storm_score_observed")
        headline_label = "STORM-BR(observed-only)"
    print(
        f"[result] answer={result['answer_accuracy']:.2%} "
        f"known-volatility={result['known_volatility_score']:.2%} "
        f"epistemic={result['epistemic_uncertainty_score']:.2%} "
        f"{headline_label}={headline:.2%}"
        + (
            f" cause-F1={result['uncertainty_cause_macro_f1']:.2%}"
            if result.get("uncertainty_cause_macro_f1") is not None
            else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
