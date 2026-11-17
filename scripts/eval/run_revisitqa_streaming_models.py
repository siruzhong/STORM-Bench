#!/usr/bin/env python
"""Seven-subset online evaluation with a finite frame buffer; reuses the shared launcher."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.run_revisitqa_models import main


if __name__ == "__main__":
    main(["--protocols", "online", "--buffer-frames", "16", *sys.argv[1:]])
