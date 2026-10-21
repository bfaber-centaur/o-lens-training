#!/usr/bin/env bash
# Separate vLLM inference venv. DO NOT overwrite the working HF/PEFT SFT venv.
# Intended image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404.
set -euo pipefail
export UV_CACHE_DIR=/root/.cache/uv
export UV_LINK_MODE=copy
export TMPDIR=/root/tmp
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME=/root/.cache
mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HUB_CACHE"
if [[ "$EUID" -ne 0 ]]; then
  echo "Run inside the RunPod container as root" >&2
  exit 1
fi
if [[ -d /root/olens-vllm-venv && ! -x /root/olens-vllm-venv/bin/python ]]; then
  echo "Partial vLLM virtualenv at /root/olens-vllm-venv; inspect manually" >&2
  exit 1
fi
command -v uv >/dev/null || python3 -m pip install uv
if [[ ! -x /root/olens-vllm-venv/bin/python ]]; then
  uv venv --python 3.12 /root/olens-vllm-venv
fi
# vLLM 0.32.0 ships cu129 and cu130 wheels; pick cu129 for CUDA 12.8 image.
VLLM_VERSION=0.32.0
VLLM_CUDA=129
WHEEL="https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION+cu$VLLM_CUDA-cp38-abi3-manylinux_2_28_x86_64.whl"
echo "Installing pinned vLLM $VLLM_VERSION cu$VLLM_CUDA in isolated venv"
uv pip install --python /root/olens-vllm-venv/bin/python "$WHEEL" \
  --extra-index-url "https://download.pytorch.org/whl/cu$VLLM_CUDA"
uv pip check --python /root/olens-vllm-venv/bin/python
/root/olens-vllm-venv/bin/python - <<'PY'
import torch, transformers, vllm, safetensors
print("GPU",torch.cuda.get_device_name(0))
print("Torch",torch.__version__,"CUDA",torch.version.cuda)
print("vLLM",vllm.__version__,"Transformers",transformers.__version__)
assert torch.cuda.is_available(), "CUDA missing"
PY
uv pip freeze --python /root/olens-vllm-venv/bin/python > /root/olens-vllm-env.txt
echo "Environment saved: /root/olens-vllm-env.txt"
echo "Inference interpreter: /root/olens-vllm-venv/bin/python"
