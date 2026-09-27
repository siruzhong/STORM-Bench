#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "Usage: $0 EVIDENCE_JSON VISIBILITY_JSON REVIEW_JSON REFERENCE_JSONL WORK_DIR OUTPUT_DIR" >&2
  exit 2
fi

package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${package_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON:-python3}" -m storm_virtualhome_qa run \
  --evidence "$1" --visibility "$2" --review "$3" --reference "$4" \
  --work "$5" --output "$6" --time-limit "${QA_SOLVER_SECONDS:-90}"
