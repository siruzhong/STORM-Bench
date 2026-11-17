#!/usr/bin/env python
"""Launch and merge RevisitQA evaluations for the registered model suite."""

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
from scripts.eval.revisitqa_streaming import streaming_audit
from scripts.eval.revisitqa_reliability import format_headline, score_selective
from scripts.eval.revisitqa_model_registry import MODEL_SPECS, parse_model_keys
from scripts.eval.revisitqa_streaming import CLOCK_FPS
from scripts.eval.revisitqa_datasets import REVISITQA_SUBSETS, REVISITQA_ALIASES, revisitqa_paths

RUNTIME_PYTHONS = {
    "vlm": REPO_ROOT / ".venv-revisitqa-vlm/bin/python",
    "videollama3": REPO_ROOT / ".venv-revisitqa-videollama3/bin/python",
    "modern": REPO_ROOT / ".venv-revisitqa-modern/bin/python",
    "legacy": REPO_ROOT / ".venv-revisitqa-legacy/bin/python",
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
    summary = {"count": len(rows)}
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
            summary = score_selective(
                gt_rows,
                rows,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
        summary.update(manifest)
    audit = streaming_audit(rows, signature.get("protocol"))
    if audit:
        summary["streaming_audit"] = audit
    summary["eval_signature"] = json.dumps(signature, sort_keys=True)
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--protocols", default="online")
    parser.add_argument("--checkpoint-root", default="ckpt")
    parser.add_argument("--output-root", default="outputs/revisitqa/released")
    parser.add_argument("--gt-file", default=None)
    parser.add_argument("--subsets", default="all", help="all or comma-separated cook,bike,healthy,music,sports,thor,vhome")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--fps", type=float, default=CLOCK_FPS)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=0,
        help="Online sliding-window capacity. 0 = Online(B=all).",
    )
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--shuffle-frames", action="store_true")
    parser.add_argument("--blank-frames", action="store_true")
    parser.add_argument("--options-only", action="store_true")
    parser.add_argument("--forced-choice", action="store_true")
    parser.add_argument("--forced-video", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--glm-thinking-budget", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.gt_file is None:
        if args.video_root is not None:
            parser.error("--video-root requires --gt-file")
        subsets = list(REVISITQA_SUBSETS) if args.subsets == "all" else [REVISITQA_ALIASES.get(x.strip(), x.strip()) for x in args.subsets.split(",")]
        if not subsets or any(x not in REVISITQA_SUBSETS for x in subsets):
            parser.error("Unknown RevisitQA subset")
        failures = []
        for subset in dict.fromkeys(subsets):
            gt_file, video_root = revisitqa_paths(subset)
            selected = argparse.Namespace(**vars(args))
            selected.gt_file, selected.video_root = gt_file, video_root
            selected.output_root = str(Path(args.output_root) / subset / "models")
            try:
                run_models(selected, parser)
            except RuntimeError as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError("; ".join(failures))
    else:
        args.video_root = args.video_root or str(Path(args.gt_file).parent / "video")
        run_models(args, parser)


def run_models(args, parser):

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
    if args.sample_limit is not None or args.episode_limit is not None:
        output_root = output_root / "_smoke"

    failed_runs = []
    for key in keys:
        spec = MODEL_SPECS[key]
        python = Path(os.environ.get(f"REVISITQA_{spec.runtime.upper()}_PYTHON", str(RUNTIME_PYTHONS[spec.runtime])))
        if not args.dry_run and not python.exists():
            raise FileNotFoundError(f"Missing runtime for {key}: {python}")
        for protocol in protocols:
            protocol_dir = protocol
            if protocol == "online" and args.buffer_frames > 0:
                protocol_dir = f"online_b{args.buffer_frames}"
            if args.shuffle_frames:
                protocol_dir = f"{protocol_dir}_shuffle"
            if args.blank_frames:
                protocol_dir = f"{protocol_dir}_blank"
            if args.text_only:
                protocol_dir = "offline_text_only" if protocol == "offline" else "online_all_text_only"
            if args.options_only:
                protocol_dir = "options_only"
            if args.forced_choice:
                protocol_dir = "text_forced_choice"
            if args.forced_video:
                protocol_dir = "online_all_forced_choice"
            output_dir = output_root / protocol_dir / key
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / "result.json"
            if not args.overwrite and result_path.exists() and not args.dry_run:
                from scripts.eval.eval_revisitqa_model import build_eval_signature
                expected_args = argparse.Namespace(**vars(args), model_key=key, protocol=protocol)
                _, expected_signature = build_eval_signature(
                    expected_args, spec.local_path(args.checkpoint_root), manifest=dataset_manifest(args.gt_file)
                )
                recorded = json.loads(result_path.read_text()).get("eval_signature")
                if isinstance(recorded, dict):
                    recorded = json.dumps(recorded, sort_keys=True)
                if recorded != expected_signature:
                    raise RuntimeError(f"Existing result uses a different dataset or protocol: {result_path}; choose a new output root or --overwrite")
                print(f"[skip] {key}/{protocol} already has {result_path}", flush=True)
                continue
            processes = []
            for chunk_idx in range(num_chunks):
                command = [
                    str(python), str(REPO_ROOT / "scripts/eval/eval_revisitqa_model.py"),
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
                    "--glm-thinking-budget", str(args.glm_thinking_budget),
                ]
                if args.sample_limit is not None:
                    command += ["--sample-limit", str(args.sample_limit)]
                if args.episode_limit is not None:
                    command += ["--episode-limit", str(args.episode_limit)]
                if args.text_only:
                    command.append("--text-only")
                if args.shuffle_frames:
                    command.append("--shuffle-frames")
                if args.blank_frames:
                    command.append("--blank-frames")
                if args.options_only:
                    command.append("--options-only")
                if args.forced_choice:
                    command.append("--forced-choice")
                if args.forced_video:
                    command.append("--forced-video")
                if args.overwrite:
                    command.append("--overwrite")
                print("[exec] " + " ".join(command), flush=True)
                if args.dry_run:
                    continue
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = devices[chunk_idx].replace("+", ",")  # "0+1": one process on two GPUs
                env["PYTHONUNBUFFERED"] = "1"
                processes.append(
                    subprocess.Popen(command, cwd=REPO_ROOT, env=env, start_new_session=True)
                )
            failures = [process.wait() for process in processes]
            if any(code != 0 for code in failures):
                print(f"[fail] {key}/{protocol} workers failed: {failures}", flush=True)
                failed_runs.append(f"{key}/{protocol}: {failures}")
                continue
            if not args.dry_run:
                summary = merge_results(
                    output_dir,
                    num_chunks,
                    gt_file=args.gt_file,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.seed,
                )
                if "effective_reliability" in summary:
                    print(f"[result] {key}/{protocol_dir}: {format_headline(summary)}", flush=True)

    if failed_runs:
        raise RuntimeError("RevisitQA workers failed: " + "; ".join(failed_runs))


if __name__ == "__main__":
    main()
