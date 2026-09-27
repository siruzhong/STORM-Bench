#!/usr/bin/env bash
# Create the four per-family evaluation environments used by
# scripts/eval/run_storm_models.py.
#
# Each environment is a lightweight venv created with --system-site-packages
# from one base interpreter that already provides a CUDA build of PyTorch.
# Only the family-specific Python packages are installed on top, so the four
# environments share a single torch/transformers-independent CUDA runtime.
#
#   .venv-storm-vlm          Qwen2.5-VL, Qwen3-VL, InternVL, Molmo, MiniCPM, Eagle
#   .venv-storm-modern       Qwen3.5
#   .venv-storm-legacy       InternVideo2.5, LLaVA-NeXT-Video
#   .venv-storm-videollama3  VideoLLaMA3, GLM-4.1V
#
# Usage:
#   BASE_PYTHON=/path/to/torch/python bash scripts/setup_storm_envs.sh
#
# If BASE_PYTHON is unset, the script tries /opt/conda/envs/prem/bin/python and
# then python3 on PATH. The base interpreter must be able to `import torch`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${BASE_PYTHON:-}" ]]; then
  for cand in /opt/conda/envs/prem/bin/python "$(command -v python3 || true)"; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
      BASE_PYTHON="${cand}"
      break
    fi
  done
fi
if [[ -z "${BASE_PYTHON:-}" || ! -x "${BASE_PYTHON}" ]]; then
  echo "No base python found. Set BASE_PYTHON to an interpreter with PyTorch installed." >&2
  exit 1
fi
if ! "${BASE_PYTHON}" -c 'import torch' >/dev/null 2>&1; then
  echo "BASE_PYTHON=${BASE_PYTHON} cannot import torch. Install a CUDA-compatible PyTorch build first." >&2
  exit 1
fi
echo "[storm-env] BASE_PYTHON=${BASE_PYTHON}"
"${BASE_PYTHON}" -c 'import torch, transformers; print("[storm-env] base torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)' 2>/dev/null || true

# FlashAttention-2 is required by the official VideoLLaMA3 and Eagle2.5 inference
# paths. If the prebuilt wheel is unavailable for the local torch/CUDA, installs
# are skipped and those two families fall back to the repo's SDPA path.
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3.post1+cu.12.8.torch.2.11}"
FLASH_ATTN_INDEX_URL="${FLASH_ATTN_INDEX_URL:-https://wheels.astral.sh/simple/cu128/}"

create_env() {
  local path="$1"
  if [[ ! -x "$path/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$path"
  fi
  "$path/bin/python" -m pip install --upgrade pip
}

install_flash_attn() {
  local python="$1"
  "$python" -m pip install "flash-attn==${FLASH_ATTN_VERSION}" \
    --index-url "${FLASH_ATTN_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    || echo "[storm-env] WARN: flash-attn install failed for ${python}; VideoLLaMA3/Eagle will use SDPA." >&2
}

create_env "$ROOT_DIR/.venv-storm-vlm"
"$ROOT_DIR/.venv-storm-vlm/bin/python" -m pip install \
  'transformers==4.57.1' 'qwen-vl-utils>=0.0.14' 'accelerate>=1.2' \
  'decord2>=0.0.1' 'molmo-utils==0.0.1' 'sentencepiece' 'timm' \
  'av' 'ffmpeg-python' 'imageio' 'opencv-python-headless' 'protobuf'
install_flash_attn "$ROOT_DIR/.venv-storm-vlm/bin/python"

create_env "$ROOT_DIR/.venv-storm-videollama3"
"$ROOT_DIR/.venv-storm-videollama3/bin/python" -m pip install \
  'transformers==4.46.3' 'accelerate==1.0.1' 'decord>=0.6' \
  'av' 'ffmpeg-python' 'imageio' 'opencv-python-headless' 'sentencepiece' 'protobuf'
install_flash_attn "$ROOT_DIR/.venv-storm-videollama3/bin/python"

create_env "$ROOT_DIR/.venv-storm-modern"
"$ROOT_DIR/.venv-storm-modern/bin/python" -m pip install \
  'transformers==5.14.1' 'qwen-vl-utils>=0.0.14' 'accelerate>=1.2' \
  'decord2>=0.0.1' 'sentencepiece'

create_env "$ROOT_DIR/.venv-storm-legacy"
"$ROOT_DIR/.venv-storm-legacy/bin/python" -m pip install \
  'transformers==4.40.1' 'accelerate>=0.30' 'decord>=0.6' \
  'sentencepiece==0.1.99' 'timm'

for env_path in .venv-storm-vlm .venv-storm-videollama3 .venv-storm-modern .venv-storm-legacy; do
  "$ROOT_DIR/$env_path/bin/python" -c 'import torch, transformers; print("['"$env_path"']", transformers.__version__, torch.__version__)'
done
