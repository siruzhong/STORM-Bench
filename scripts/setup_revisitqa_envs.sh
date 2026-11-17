#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${BASE_PYTHON:-}" ]]; then
  for cand in \
    /opt/conda/bin/python \
    /opt/conda/envs/ostem/bin/python \
    "$(command -v python3 || true)"; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
      BASE_PYTHON="${cand}"
      break
    fi
  done
fi
if [[ -z "${BASE_PYTHON:-}" || ! -x "${BASE_PYTHON}" ]]; then
  echo "Set BASE_PYTHON to a CUDA-matched Python interpreter." >&2
  exit 1
fi
echo "[revisitqa-env] BASE_PYTHON=${BASE_PYTHON}"

create_env() {
  local path="$1"
  if [[ ! -x "$path/bin/python" ]]; then
    "$BASE_PYTHON" -m venv --system-site-packages "$path"
  fi
  "$path/bin/python" -m pip install --upgrade pip
}

create_env "$ROOT_DIR/.venv-revisitqa-vlm"
"$ROOT_DIR/.venv-revisitqa-vlm/bin/python" -m pip install \
  'transformers==4.57.1' 'qwen-vl-utils>=0.0.14' 'accelerate>=1.2' \
  'decord2>=0.0.1' 'molmo-utils==0.0.1' 'sentencepiece' 'timm' \
  'av' 'ffmpeg-python' 'imageio' 'opencv-python-headless' 'protobuf'
# Eagle2.5 official demo uses attn_implementation=flash_attention_2.
"$ROOT_DIR/.venv-revisitqa-vlm/bin/python" -m pip install \
  'flash-attn==2.8.3.post1+cu.12.8.torch.2.11' \
  --index-url https://wheels.astral.sh/simple/cu128/ \
  --extra-index-url https://pypi.org/simple

create_env "$ROOT_DIR/.venv-revisitqa-videollama3"
"$ROOT_DIR/.venv-revisitqa-videollama3/bin/python" -m pip install \
  'transformers==4.46.3' 'accelerate==1.0.1' 'decord>=0.6' \
  'av' 'ffmpeg-python' 'imageio' 'opencv-python-headless' 'sentencepiece' 'protobuf'
# Official VideoLLaMA3 inference uses attn_implementation=flash_attention_2
# (vision encoder: flash_attn_varlen_func). SDPA OOMs on 128-frame videos.
"$ROOT_DIR/.venv-revisitqa-videollama3/bin/python" -m pip install \
  'flash-attn==2.8.3.post1+cu.12.8.torch.2.11' \
  --index-url https://wheels.astral.sh/simple/cu128/ \
  --extra-index-url https://pypi.org/simple

create_env "$ROOT_DIR/.venv-revisitqa-modern"
"$ROOT_DIR/.venv-revisitqa-modern/bin/python" -m pip install \
  'transformers==5.14.1' 'qwen-vl-utils>=0.0.14' 'accelerate>=1.2' \
  'decord2>=0.0.1' 'sentencepiece'

create_env "$ROOT_DIR/.venv-revisitqa-legacy"
"$ROOT_DIR/.venv-revisitqa-legacy/bin/python" -m pip install \
  'transformers==4.40.1' 'accelerate>=0.30' 'decord>=0.6' \
  'sentencepiece==0.1.99' 'timm'

for env_path in .venv-revisitqa-vlm .venv-revisitqa-videollama3 .venv-revisitqa-modern .venv-revisitqa-legacy; do
  "$ROOT_DIR/$env_path/bin/python" -c 'import torch, transformers; print(transformers.__version__, torch.__version__)'
done
