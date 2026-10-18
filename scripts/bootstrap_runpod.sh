#!/usr/bin/env bash
# Verified on RunPod image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# All installs/caches use ephemeral pod disk, NEVER /workspace (FUSE global volume).
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run inside the RunPod container as root" >&2
  exit 1
fi
export UV_CACHE_DIR=/root/.cache/uv
export UV_LINK_MODE=copy
export TMPDIR=/root/tmp
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE="${HF_HOME}/hub"
export MAX_JOBS="${MAX_JOBS:-4}"
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HUB_CACHE"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install uv
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.12 .venv
fi
source .venv/bin/activate

uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==5.19.0 peft accelerate datasets safetensors numpy tqdm flash-linear-attention
uv pip install --no-cache --no-build-isolation causal-conv1d

python - <<'PY'
import torch, fla, causal_conv1d
import transformers, peft, datasets, safetensors
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__, "PEFT:", peft.__version__)
print("Fast-path packages import successfully (runtime forward/backward still required)")
PY
uv pip freeze > /root/o-lens-training-env.txt
echo "Saved environment snapshot: /root/o-lens-training-env.txt"
echo "Activate via: source $REPO_DIR/.venv/bin/activate"
