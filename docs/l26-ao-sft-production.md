# L26 AO SFT: production-capable first rung

Goal: **train a usable L26 AR-output-conditioned oracle** as a stepping
stone toward Agam Bhatia/Camila Blank's full O-Lens pipeline.
Reference: https://github.com/agu18dec/olens_draft/blob/main/docs/pipeline.md

This PR ports the *supervised AR-output -> span* objective, masked causal
token CE, soft-token injection, frozen AR, LoRA r16, alpha=8000 from our
working 4B pilot. It adds larger data caps and restartable optimizer state.
It deliberately does **not** claim exact upstream parity:

- Agam's AO pool uses **fresh, non-overlapping continuation crops**, with
  deduplication and exclusion of AR training crop prefixes; our first rung
  uses already-harvested AR-validation spans, with rollout-level
  train/validation disjointness. This is smaller and less diverse.
- Agam uses a joint multi-layer 27B AR and additional training stages.
  Here we use one frozen L26 4B AR.
- Our prompt, last-slot unit normalization and alpha=8000 are inherited
  from the pilot, not a verified match to Agam's later raw-scaled,
  stage-dependent contracts. We must match his contract when implementing
  subsequent distillation and RL.
- This is supervised inversion only, **not** a complete O-Lens.

## Stage 1 — larger frozen AR output cache

On a pod with a CUDA environment, before paid training:

```bash
cd /root/o-lens-training
git pull --ff-only
source .venv/bin/activate
python -u oracle/precompute_arout.py \
  --max-train 12000 --max-val 1024 \
  --checkpoint /workspace/o-lens-qwen35-4b/ar/l26_200k_all32/checkpoints/step_001424 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-arout-12k-v1 \
  2>&1 | tee /root/l26-arout-12k-v1.log
```

The precompute CLI refuses overwriting existing output, checks the source
AR COMPLETE marker, and writes source-index hashes and a safetensors checksum.
It may produce fewer than 12,000/1,024 rows if the AR holdout lacks enough.

## Stage 2 — small **new** SFT smoke test

```bash
python -u oracle/train_ao_sft_production.py \
  --arout /workspace/o-lens-qwen35-4b/oracle/l26-arout-12k-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-prod-smoke-v1 \
  --max-train 12000 --max-val 1024 \
  --max-steps 5 --save-every 2 \
  2>&1 | tee /root/l26-ao-sft-prod-smoke-v1.log
```

Verify checkpoints/step_000002, 000004 and 000005 each contain
COMPLETE, meta.json, tokenizer/, lora/, state.pt, metrics.json.
latest.json must name the final complete checkpoint.
Resume an interrupted 5-step smoke using exactly the same flags plus
`--resume`; to test genuine resume, interrupt after a completed intermediate
checkpoint and run with `--resume`, comparing fixed per-step sampled indices
and validation CE to an uninterrupted control.

## Stage 3 — real SFT run (separate new directory)

After passing smoke tests, explicitly choose training budget. Example
below is a **starting configuration, not an upstream-authorized optimum**:

```bash
python -u oracle/train_ao_sft_production.py \
  --arout /workspace/o-lens-qwen35-4b/oracle/l26-arout-12k-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1 \
  --max-train 12000 --max-val 1024 \
  --max-steps 500 --save-every 50 \
  2>&1 | tee /root/l26-ao-sft-500-v1.log
```

On unexpected pod loss, reissue **the identical command** with `--resume`.
No training step is repeated beyond the last completed checkpoint.
Training samples are deterministic per (seed, optimizer step, microbatch);
LoRA optimizer and torch/CUDA RNG states are restored. Checkpoints use a
staging directory and atomic rename; the latest pointer is updated last.
An interrupted, partially written staging directory must be inspected and
removed manually before reattempting that checkpoint.

Evaluate a finished checkpoint with the existing
`oracle/eval_ao1_arout.py` (the evaluator accepts each checkpoint's
lora/tokenizer/meta and COMPLETE marker). The production cache must remain
at its original path for evaluation.

**Scientific limit:** the AR-validation pool becomes AO training data;
neither AO validation nor prior held-out AR quality scores constitute a
fresh end-to-end test. No assertions of faithfulness or cross-model parity.

All artifacts and checkpoints belong on /workspace persistent storage.
Terminate the RunPod instance after verifying outputs; persistent volume
remains available independently of GPU rental.
