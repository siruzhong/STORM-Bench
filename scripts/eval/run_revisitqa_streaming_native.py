#!/usr/bin/env python
"""Launch native streaming RevisitQA evaluations over the released subsets and merge the results.

Example:
  python scripts/eval/run_revisitqa_streaming_native.py --models flash_vstream_7b --subsets all \
      --output-root outputs/revisitqa/released --cuda-devices 0,1,2,3
Each chunk is a separate process on one device entry; repeat an id (e.g. 0,0,1,1) to place two
chunks on one GPU, and join ids with "+" (e.g. 0+1,2+3) for models that need several GPUs per process.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.run_revisitqa_models import detect_cuda_devices, merge_results  # noqa: E402
from scripts.eval.revisitqa_datasets import REVISITQA_ALIASES, REVISITQA_SUBSETS, revisitqa_paths  # noqa: E402
from scripts.eval.revisitqa_reliability import format_headline  # noqa: E402
from scripts.eval.streaming_adapters import specs  # noqa: E402


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--subsets", default="all")
    parser.add_argument("--output-root", default="outputs/revisitqa/released")
    parser.add_argument("--checkpoint-root", default="ckpt")
    parser.add_argument("--cuda-devices", default=None)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--claim", action="store_true",
                        help="Claim the output root for this host; launchers on other hosts then skip it, so two nodes "
                             "sharing the disk never write the same prediction files.")
    args = parser.parse_args(argv)
    registry = specs()
    models = [m.strip() for m in args.models.split(",")]
    unknown = [m for m in models if m not in registry]
    if unknown:
        parser.error(f"unknown streaming models {unknown}; registered: {sorted(registry)}")
    subsets = list(REVISITQA_SUBSETS) if args.subsets == "all" else [REVISITQA_ALIASES.get(s, s) for s in args.subsets.split(",")]
    devices = [d.strip() for d in (args.cuda_devices or detect_cuda_devices()).split(",") if d.strip()]
    num_chunks = 1 if args.sample_limit is not None else len(devices)
    root = Path(args.output_root)
    if args.sample_limit is not None or args.episode_limit is not None:
        root = root / "_smoke"
    claim = root / ".streaming_claim"
    host = socket.gethostname()
    if claim.exists() and claim.read_text().strip() != host:
        print(f"[skip] {root} is claimed by {claim.read_text().strip()}", flush=True)
        return
    if args.claim:
        root.mkdir(parents=True, exist_ok=True)
        claim.write_text(host + "\n")
    failed = []
    for key in models:
        python = registry[key].runtime_path()
        for subset in subsets:
            gt_file, video_root = revisitqa_paths(subset)
            out = root / subset / "models" / "streaming_native" / key
            out.mkdir(parents=True, exist_ok=True)
            procs = []
            for idx in range(num_chunks):
                cmd = [str(python), str(REPO_ROOT / "scripts/eval/eval_revisitqa_streaming.py"), "--model-key", key,
                       "--checkpoint-root", args.checkpoint_root, "--gt-file", gt_file, "--video-root", video_root,
                       "--output-dir", str(out), "--num-chunks", str(num_chunks), "--chunk-idx", str(idx),
                       "--max-pixels", str(args.max_pixels)]
                for flag in ("sample_limit", "episode_limit"):
                    if getattr(args, flag) is not None:
                        cmd += [f"--{flag.replace('_', '-')}", str(getattr(args, flag))]
                if args.overwrite:
                    cmd.append("--overwrite")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[idx % len(devices)].replace("+", ","), PYTHONUNBUFFERED="1",
                           PYTHONPATH=str(REPO_ROOT))
                print("[exec] " + " ".join(cmd), flush=True)
                procs.append(subprocess.Popen(cmd, cwd=REPO_ROOT, env=env))
            codes = [p.wait() for p in procs]
            if any(codes):
                failed.append(f"{key}/{subset}: {codes}")
                print(f"[fail] {key}/{subset}: {codes}", flush=True)
                continue
            if num_chunks > 1:
                summary = merge_results(out, num_chunks, gt_file=gt_file)
                print(f"[result] {key}/{subset}/streaming_native: {format_headline(summary)}", flush=True)
    if failed:
        raise SystemExit("failed: " + "; ".join(failed))


if __name__ == "__main__":
    main()
