# RunPod A100 bootstrap

Known-good image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

On a fresh pod with the persistent harvest volume mounted:

```bash
cd /root
git clone https://github.com/bubskee/o-lens-training.git
cd o-lens-training
bash scripts/bootstrap_runpod.sh
source .venv/bin/activate
```

The script installs Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.19.0,
PEFT and the flash-linear-attention and causal-conv1d fast paths. The original
container's PyTorch version is replaced **inside the virtual environment only**.

All caches and CUDA build temporary files live under /root rather than the
FUSE-mounted /workspace network storage. The causal-conv1d CUDA compilation
can take a while; MAX_JOBS defaults to 4 to limit compilation memory.

This setup was manually confirmed working on the specified image. The
bootstrap script still needs an automated smoke test on a fresh pod; successful
imports do not prove the CUDA forward/backward path. Use a bounded AR run before
committing expensive production training.

Training datasets, manifests, and final checkpoints belong on persistent
storage. The virtual environment and its cache are disposable with the pod.
