#!/usr/bin/env python
"""Download registered RevisitQA benchmark models from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.revisitqa_model_registry import MODEL_SPECS, parse_model_keys


def is_complete_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index["weight_map"].values())
        except (KeyError, TypeError, json.JSONDecodeError):
            return False
        return bool(shards) and all(
            (path / shard).is_file() and (path / shard).stat().st_size > 0
            for shard in shards
        )
    weight_patterns = ("*.safetensors", "*.bin", "*.pt")
    return any(
        weight.is_file() and weight.stat().st_size > 0
        for pattern in weight_patterns
        for weight in path.glob(pattern)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="Comma-separated registry keys or all")
    parser.add_argument("--checkpoint-root", default="ckpt")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN") and not args.dry_run:
        raise RuntimeError("HF_TOKEN must be supplied through the environment")

    keys = parse_model_keys(args.models)
    checkpoint_root = Path(args.checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_root / "revisitqa_models.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}

    if not args.dry_run:
        from huggingface_hub import snapshot_download

    for key in keys:
        spec = MODEL_SPECS[key]
        local_dir = spec.local_path(checkpoint_root)
        status = "present" if is_complete_model_dir(local_dir) and not args.force else "download"
        print(f"[{status}] {key}: {spec.repo_id} -> {local_dir}", flush=True)
        if status == "download" and not args.dry_run:
            snapshot_download(
                repo_id=spec.repo_id,
                revision=args.revision,
                local_dir=local_dir,
                token=os.environ["HF_TOKEN"],
            )
            if not is_complete_model_dir(local_dir):
                raise RuntimeError(f"Downloaded directory is incomplete: {local_dir}")
        manifest[key] = {
            "repo_id": spec.repo_id,
            "local_path": str(local_dir),
            "family": spec.family,
            "runtime": spec.runtime,
            "complete": is_complete_model_dir(local_dir),
        }

    if not args.dry_run:
        manifest_path = checkpoint_root / "revisitqa_models.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[manifest] {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
