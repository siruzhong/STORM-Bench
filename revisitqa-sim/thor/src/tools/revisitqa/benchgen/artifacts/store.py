"""Filesystem layout and atomic JSON writes for benchmark runs."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..domain.contracts import primitive


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create_run(self) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.root / f"run_{stamp}"
        suffix = 1
        while candidate.exists():
            candidate = self.root / f"run_{stamp}_{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True)
        return candidate

    def episode_dir(
        self, run_dir: Path, index: int, scene: str, seed: int,
    ) -> Path:
        path = run_dir / f"{index:05d}_{scene}_seed{seed}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def write_json(self, path: str | Path, value: Any) -> None:
        target = Path(path)
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                primitive(value), handle, indent=2, sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
        os.replace(temporary, target)
