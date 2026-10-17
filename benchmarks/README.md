# Reusable AR GPU benchmark ladder

The recovered scripts in legacy/ remain unchanged. The bounded trainer is
benchmarks/train_ar_bounded.py; benchmarks/run_gpu.py starts concurrent
independent training processes with separate outputs and logs.

## Pod setup

Use standard RunPod PyTorch/CUDA image. On ephemeral pod disk:

    export UV_CACHE_DIR=/root/.cache/uv UV_LINK_MODE=copy TMPDIR=/root/tmp
    mkdir -p /root/.cache/uv /root/tmp
    uv venv --python 3.12 /root/o-lens-training/.venv
    source /root/o-lens-training/.venv/bin/activate

Known functional starting point (not yet a frozen lockfile):

    uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
    uv pip install transformers==5.19.0 peft accelerate datasets safetensors numpy tqdm
    uv pip install flash-linear-attention
    uv pip install --no-cache --no-build-isolation causal-conv1d

RunPod network /workspace is a FUSE volume: keep uv, HF and build caches on
pod disk, and persist harvested data/checkpoints to /workspace. Confirm the
fla and causal_conv1d imports and no kernel-fallback warnings during training.

## Commands

From /root/o-lens-training with the environment active:

A100 one-wide with existing L30 whitener:

    python benchmarks/run_gpu.py --label a100-one-l30 --layers 30 \
      --whitener 30=/workspace/o-lens-qwen35-4b/ar/l30_200k_v3/whitener_l30.safetensors \
      --gpu-hourly 1.39

A100 two-wide fresh whitening for L22 and L26:

    python benchmarks/run_gpu.py --label a100-two-l22-l26 \
      --layers 22 26 --gpu-hourly 1.39

B200 three-wide fresh whitening:

    python benchmarks/run_gpu.py --label b200-three-l22-l26-l30 \
      --layers 22 26 30 --gpu-hourly 5.98

Replace hourly rates with the actual prices shown in RunPod. Default benchmark
is 20 optimizer steps per layer with microbatch 32, accumulation 8 and
evaluation at the stop. Each launch refuses to overwrite an existing output.

Output: /workspace/o-lens-benchmarks/<label>/ contains benchmark_config.json,
lXX.log, lXX/ checkpoint outputs, gpu.csv and summary.json. Summaries include
total wall time, exit codes, sampled maximum GPU memory/utilization, and
approximate compute cost; the timeline combines setup, whitening, validation
and checkpoints, so do not present it as training-only throughput.

The only currently harvested pair layers are 22, 26 and 30. Four-wide testing
requires another layer's activation pairs. nvidia-smi utilization samples
measure GPU busy time, not floating-point saturation; its reported memory
can miss short-lived peaks.

## Historical anchors

- A100 one-wide L30 (pre-existing whitener): 20 steps reached at ~93 s;
  total training/validation/checkpoint ~157 s; MSE 1.020024.
- A100 two-wide L22/L26 (fresh whitening): both completed step 20 and
  checkpoints; measured peak 34,109 MiB in 409 telemetry samples.
  These runs differ in layer depths and whitening setup, so aggregate
  speedup remains to be measured rigorously.

Original historical checkpoints under /workspace/o-lens-qwen35-4b/ar/ are
never benchmark destinations.
