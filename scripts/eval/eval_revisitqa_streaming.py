#!/usr/bin/env python
"""Evaluate a native streaming model on RevisitQA with persistent visual state.

Per episode the adapter is reset once; at each query time t_q the frames that became visible since the
previous query (1 FPS clock, timestamps <= t_q) are passed to `observe`, and the question is answered
from the adapter's current state. No frame is re-sent and no question or answer is kept across queries.
Predictions use the same schema as `eval_revisitqa_model.py`, so `run_revisitqa_models.merge_results` and
`revisitqa_reliability.score_selective` score them unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.data_io import dataset_manifest, load_records, normalize_mcq_sample  # noqa: E402
from scripts.eval.eval_revisitqa_model import extract_choice, grouped_chunk, iter_video_groups, resolve_video  # noqa: E402
from scripts.eval.revisitqa_model_adapters import MCQ_INSTRUCTION, MCQ_PROMPT_VERSION  # noqa: E402
from scripts.eval.revisitqa_reliability import format_headline, score_selective  # noqa: E402
from scripts.eval.revisitqa_streaming import CLOCK_FPS, IncrementalVideoSource, select_episode_subset  # noqa: E402
from scripts.eval.streaming_adapters import build, specs  # noqa: E402

PROTOCOL = "revisitqa_streaming_native_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--checkpoint-root", default="ckpt")
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--fps", type=float, default=CLOCK_FPS)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def signature(args, model_path: Path, manifest: dict) -> tuple[dict, str]:
    data = {
        "schema_version": PROTOCOL,
        "protocol": "online",
        "protocol_label": "streaming_native",
        "model_key": args.model_key,
        "model_path": str(model_path.resolve()),
        "clock_fps": args.fps,
        "max_pixels": args.max_pixels,
        "native_recurrent_state": True,
        "question_history_reused": False,
        "mcq_prompt_version": MCQ_PROMPT_VERSION,
        **{k: manifest[k] for k in ("dataset_sha256", "dataset_ids_sha256", "dataset_count", "dataset_episode_count")},
    }
    for key in ("episode_limit", "sample_limit"):
        if getattr(args, key) is not None:
            data[key] = getattr(args, key)
    return data, json.dumps(data, sort_keys=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    spec = specs()[args.model_key]
    model_path = Path(args.checkpoint_root) / spec.checkpoint
    gt_path = Path(args.gt_file)
    manifest = dataset_manifest(gt_path)
    rows = [normalize_mcq_sample(r) for r in load_records(gt_path)]
    if args.episode_limit is not None:
        rows = select_episode_subset(rows, args.episode_limit, args.seed)
    rows = grouped_chunk(rows, args.num_chunks, args.chunk_idx)
    if args.sample_limit is not None:
        rows = rows[: args.sample_limit]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"pred_{args.num_chunks}_{args.chunk_idx}.jsonl"
    sig_data, sig = signature(args, model_path, manifest)
    existing = {}
    if pred_path.exists() and not args.overwrite:
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("eval_signature") != sig:
                    raise RuntimeError(f"Existing predictions use a different configuration: {pred_path}")
                existing[str(row["id"])] = row
    pending = {str(r["id"]) for r in rows} - set(existing)
    # A partially evaluated episode must be replayed from its start to rebuild the streaming state.
    done_episodes = {r["episode_id"] for r in rows if all(str(x["id"]) in existing for x in rows if x["episode_id"] == r["episode_id"])}
    if pending:
        adapter = build(args.model_key, model_path, args)
        with pred_path.open("w" if args.overwrite else "a", encoding="utf-8") as handle:
            for group in iter_video_groups(rows):
                if group[0]["episode_id"] in done_episodes:
                    continue
                adapter.reset()
                source = IncrementalVideoSource(resolve_video(Path(args.video_root), group[0]), args.fps)
                for row in group:
                    tq = float(row["query_time"])
                    frames = source.take_until(tq)
                    if any(f.timestamp > tq + 1e-6 for f in frames):
                        raise RuntimeError(f"future frame for {row['id']}")
                    t0 = time.perf_counter()
                    adapter.observe(frames)
                    t1 = time.perf_counter()
                    if str(row["id"]) in existing:
                        continue  # replaying the episode only to rebuild state
                    raw = adapter.answer(MCQ_INSTRUCTION + row["question"], len(row["options"]))
                    t2 = time.perf_counter()
                    choice = extract_choice(raw, len(row["options"]))
                    index = ord(choice) - ord("A") if choice is not None else None
                    result = {
                        "id": row["id"], "episode_id": row["episode_id"], "question": row["question"],
                        "answer": row["answer"], "pred": raw, "pred_choice": choice, "pred_index": index,
                        "correct": index == row["answer"], "model_key": args.model_key,
                        "eval_signature": sig, "protocol": sig_data["protocol_label"], "query_time": tq,
                        "question_type": row.get("question_type"), "option_permutation": row.get("option_permutation"),
                        "native_recurrent_state": True, "visual_state_persistent": True, "question_history_reused": False,
                        "stream_frames_seen": source.visible_count, "buffered_frames": source.visible_count,
                        "buffer_capacity": None, "future_leakage": False,
                        "observe_seconds": t1 - t0, "answer_seconds": t2 - t1, **adapter.state_info(),
                    }
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing[str(row["id"])] = result
                    print(f"[{args.model_key}/streaming_native] {len(existing)}/{len(rows)} {row['id']}: {choice or 'INVALID'}", flush=True)
    if args.num_chunks == 1:
        ordered = [existing[str(r["id"])] for r in rows if str(r["id"]) in existing]
        (out_dir / "pred.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
        if ordered:
            score = score_selective([r for r in rows if str(r["id"]) in existing], ordered,
                                    bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.seed)
            score.update(manifest)
            score["eval_signature"] = sig
            (out_dir / "result.json").write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
            print(f"[result] {args.model_key}/streaming_native: {format_headline(score)}", flush=True)


if __name__ == "__main__":
    main()
