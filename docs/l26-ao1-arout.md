# L26 AO-1: frozen AR(span) → text

This is a bounded reconstruction-conditioned AO-1 pilot following the methodological
direction of Agam Bhatia's Oracle Lens implementation:
https://github.com/agu18dec/olens_draft
This is NOT a reproduction of its 27B model, joint AR, data scale, learned prompt
or downstream distillation/RL. It is a controlled comparison to our
GT-activation baseline on Qwen3.5-4B L26.

## Stage 1: precompute AR(span) vectors

Use the original completed L26 AR (do not modify its checkpoints).
Each exact text span is passed to the frozen AR with the training-time
[Layer 26] tag. This produces reconstructed residual vectors **not**
the original harvested activations. Source rows come exclusively from the
AR validation partition, then are deterministically divided by rollout into
oracle training and validation pools, with SHA-256 checks against the GT
pilot's index selection.

```bash
cd /root/o-lens-training
source .venv/bin/activate
python -u oracle/precompute_arout.py \
  --checkpoint /workspace/o-lens-qwen35-4b/ar/l26_200k_all32/checkpoints/step_001424 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-arout-v1 \
  2>&1 | tee /root/l26-arout-v1.log
```

## Stage 2: fresh AO-1 SFT, 20-step smoke

```bash
python -u oracle/train_ao1_arout.py \
  --arout /workspace/o-lens-qwen35-4b/oracle/l26-arout-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-ao1-20-v1 \
  --layer 26 --max-steps 20 --batch-size 4 --grad-accum 4 \
  2>&1 | tee /root/l26-ao1-20-v1.log
```

The trainer refuses mismatched seed, layer, model, caps, file checksum or exact
split-index hashes; it writes the contract and adapter to a new directory.
The 20-step budget is a smoke test, not a scientifically tuned training run.

## Stage 3: matched AO-1 controls

```bash
python -u oracle/eval_ao1_arout.py \
  --checkpoint /workspace/o-lens-qwen35-4b/oracle/l26-ao1-20-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-ao1-20-eval-v1 \
  --sample-count 6 \
  2>&1 | tee /root/l26-ao1-20-eval-v1.log
```

Compare teacher-forced CE on correct AR outputs, shuffled AR outputs, and zero
vectors. Existing GT-activation eval applies an analogous correct/shuffled/zero
control at the identical 256 validation rows, but input distributions differ.
Qualitative generation is exploratory. Final evaluation needs genuinely
untouched prompts and controls for AR reward gaming. The AR's held-out pool is
now used to train the oracle and is NOT an untouched end-to-end test.

## Audit gates

- Precompute output tensor shape 2048×2560 train, 256×2560 val.
- Check checkpoint COMPLETE markers; avoid overwrites.
- Precompute genuinely uses frozen AR forward, not raw target residuals.
- Verify no CUDA fallback warnings and actual live LoRA modules after loading.
- Inspect correct-vs-shuffled CE gap and the actual generated samples.
- No claims of oracle faithfulness or Agam methodological parity before
  independent testing. No RL, distillation or additional model changes here.

The exact implementation is proposed for GPU smoke testing; these commands
have not yet been executed by this PR author.
