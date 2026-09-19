#!/usr/bin/env python
"""Launch and merge STORM-Bench evaluations for the registered model suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import dataset_manifest, load_records, normalize_mcq_sample
from scripts.eval.storm_diagnostics import accuracy_summary, attach_storm_scores, score_answer_predictions
from scripts.eval.storm_model_registry import MODEL_SPECS, parse_model_keys
from scripts.eval.storm_streaming import CLOCK_FPS

RUNTIME_PYTHONS = {
    "vlm": REPO_ROOT / ".venv-storm-vlm/bin/python",
    "videollama3": REPO_ROOT / ".venv-storm-videollama3/bin/python",
    "modern": REPO_ROOT / ".venv-storm-modern/bin/python",
    "legacy": REPO_ROOT / ".venv-storm-legacy/bin/python",
}


def detect_cuda_devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        return visible
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return ",".join(line.strip() for line in result.stdout.splitlines() if line.strip())


def _eval_signature(rows: list[dict]) -> dict:
    raw = rows[0].get("eval_signature") if rows else None
    if not raw:
        return {}
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def merge_results(
    output_dir: Path,
    num_chunks: int,
    gt_file: str | Path | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260817,
) -> dict:
    rows = []
    for index in range(num_chunks):
        path = output_dir / f"pred_{num_chunks}_{index}.jsonl"
        if not path.exists():
            continue
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    prediction_ids = [str(row["id"]) for row in rows]
    if len(set(prediction_ids)) != len(prediction_ids):
        raise ValueError(f"Duplicate prediction ids found under {output_dir}")
    rows.sort(key=lambda row: row["id"])
    merged_path = output_dir / "pred.jsonl"
    merged_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    signature = _eval_signature(rows)
    summary = accuracy_summary(rows)
    gt_path = Path(gt_file) if gt_file is not None else None
    if gt_path is not None and gt_path.is_file():
        manifest = dataset_manifest(gt_path)
        for key in ("dataset_sha256", "dataset_ids_sha256", "dataset_count", "dataset_episode_count"):
            if key in signature and signature[key] != manifest[key]:
                raise ValueError(
                    f"Prediction manifest mismatch for {key}: "
                    f"{signature[key]!r} != {manifest[key]!r}"
                )
        gt_by_id = {str(row["id"]): normalize_mcq_sample(row) for row in load_records(gt_path)}
        pred_ids = {str(row["id"]) for row in rows}
        missing = [str(row["id"]) for row in rows if str(row["id"]) not in gt_by_id]
        missing_gt = [gid for gid in gt_by_id if gid not in pred_ids]
        subset_eval = bool(signature.get("sample_limit") or signature.get("episode_limit"))
        if missing_gt and not subset_eval:
            raise ValueError(
                f"Predictions are missing {len(missing_gt)} ground-truth ids: {missing_gt[:5]}"
            )
        if missing:
            raise ValueError(f"Ground truth is missing prediction ids: {missing[:5]}")
        else:
            gt_rows = [gt_by_id[str(row["id"])] for row in rows]
            if signature.get("with_status"):
                summary = attach_storm_scores(
                    rows,
                    gt_rows,
                    signature,
                    bootstrap_replicates=bootstrap_replicates,
                    bootstrap_seed=bootstrap_seed,
                )
            else:
                summary.update(score_answer_predictions(gt_rows, rows))
        summary.update(manifest)
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--protocols", default="offline,online")
    parser.add_argument("--checkpoint-root", default="ckpt")
    parser.add_argument("--output-root", default="outputs/storm_bench/models")
    parser.add_argument("--gt-file", default="data/eval_video/storm_real/qa_results/questions.jsonl")
    parser.add_argument("--video-root", default="data/eval_video/storm_real/video")
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--fps", type=float, default=CLOCK_FPS)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=0,
        help="Online sliding-window capacity. 0 = Online(B=all).",
    )
    parser.add_argument("--with-status", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--shuffle-frames", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = parse_model_keys(args.models)
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    if any(protocol not in {"offline", "online"} for protocol in protocols):
        parser.error("--protocols accepts offline and online")
    cuda_devices = args.cuda_devices or detect_cuda_devices()
    devices = [item.strip() for item in cuda_devices.split(",") if item.strip()]
    if not devices:
        parser.error("No visible CUDA devices detected")
    num_chunks = 1 if args.sample_limit is not None else len(devices)
    output_root = Path(args.output_root)
    if args.sample_limit is not None:
        output_root = output_root / "_smoke"

    for key in keys:
        spec = MODEL_SPECS[key]
        python = RUNTIME_PYTHONS[spec.runtime]
        if not args.dry_run and not python.exists():
            raise FileNotFoundError(f"Missing runtime for {key}: {python}")
        for protocol in protocols:
            protocol_dir = protocol
            if protocol == "online" and args.buffer_frames > 0:
                protocol_dir = f"online_b{args.buffer_frames}"
            if args.shuffle_frames:
                protocol_dir = f"{protocol_dir}_shuffle"
            if args.text_only:
                protocol_dir = "offline_text_only" if protocol == "offline" else "online_all_text_only"
            output_dir = output_root / protocol_dir / key
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / "result.json"
            if not args.overwrite and result_path.exists() and not args.dry_run:
                print(f"[skip] {key}/{protocol} already has {result_path}", flush=True)
                continue
            processes = []
            for chunk_idx in range(num_chunks):
                command = [
                    str(python), str(REPO_ROOT / "scripts/eval/eval_storm_model.py"),
                    "--model-key", key,
                    "--checkpoint-root", args.checkpoint_root,
                    "--protocol", protocol,
                    "--output-dir", str(output_dir),
                    "--gt-file", args.gt_file,
                    "--video-root", args.video_root,
                    "--num-chunks", str(num_chunks),
                    "--chunk-idx", str(chunk_idx),
                    "--fps", str(args.fps),
                    "--max-pixels", str(args.max_pixels),
                    "--buffer-frames", str(args.buffer_frames),
                    "--seed", str(args.seed),
                    "--bootstrap-replicates", str(args.bootstrap_replicates),
                ]
                if args.sample_limit is not None:
                    command += ["--sample-limit", str(args.sample_limit)]
                if args.with_status:
                    command.append("--with-status")
                if args.text_only:
                    command.append("--text-only")
                if args.shuffle_frames:
                    command.append("--shuffle-frames")
                if args.overwrite:
                    command.append("--overwrite")
                print("[exec] " + " ".join(command), flush=True)
                if args.dry_run:
                    continue
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = devices[chunk_idx]
                env["PYTHONUNBUFFERED"] = "1"
                processes.append(
                    subprocess.Popen(command, cwd=REPO_ROOT, env=env, start_new_session=True)
                )
            failures = [process.wait() for process in processes]
            if any(code != 0 for code in failures):
                print(f"[fail] {key}/{protocol} workers failed: {failures}", flush=True)
                continue
            if not args.dry_run:
                summary = merge_results(
                    output_dir,
                    num_chunks,
                    gt_file=args.gt_file,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.seed,
                )
                line = (
                    f"[result] {key}/{protocol}: {summary['correct']}/{summary['total']} "
                    f"({summary['accuracy']:.2%})"
                )
                headline = summary.get("storm_score")
                headline_label = "STORM-BR"
                if headline is None:
                    headline = summary.get("storm_score_observed")
                    headline_label = "STORM-BR(observed-only)"
                if headline is not None:
                    line += (
                        f" known-volatility={summary['known_volatility_score']:.2%} "
                        f"epistemic={summary['epistemic_uncertainty_score']:.2%} "
                        f"{headline_label}={headline:.2%}"
                    )
                    if summary.get("uncertainty_cause_macro_f1") is not None:
                        line += f" cause-F1={summary['uncertainty_cause_macro_f1']:.2%}"
                print(line, flush=True)


if __name__ == "__main__":
    main()
