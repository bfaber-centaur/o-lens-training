# L26 true-activation soft-token SFT — bounded baseline

This is a **pilot**, not a complete O-Lens. It tests whether a LoRA-adapted
Qwen3.5-4B can recover its own original token span from the true post-block
L26 activation injected at one input-embedding slot. It does **not** train
the AR(out) inversion objective, or use multi-concept distillation or RL.

The design was informed by the upstream Oracle Lens implementation:
https://github.com/agu18dec/olens_draft
In particular: explicit injection contracts, deterministic splits, masked
cross-entropy, and checkpoint metadata. We do not vendor their private training
harness or assume their 27B model-specific checkpoint formats work on 4B.

## Preflight and smoke checks

From repository root with the known-good RunPod uv environment active:

    python -m unittest discover -s tests -p 'test_oracle_pilot.py' -v
    python oracle/train_gt_sft.py --help

Run in an independent output directory:

    python -u oracle/train_gt_sft.py \
      --layer 26 \
      --data /workspace/o-lens-qwen35-4b/data/olens_200k_all32 \
      --output /workspace/o-lens-qwen35-4b/oracle/l26-gt-sft-pilot-v1 \
      --max-steps 20 --batch-size 4 --grad-accum 4 \
      2>&1 | tee /root/l26-gt-sft-pilot-v1.log

Use a new output name if retrying; overwrites are prohibited. The A100's
CUDA fast-path dependencies must be present. The pilot writes meta.json,
metrics.json, tokenizer/, lora/ and COMPLETE after successful save.

## Method and safeguards

- Source examples are drawn *only* from the existing AR validation
  partition. That keeps oracle training text disjoint from AR training
  rollouts. The AR validation partition is consequently consumed and
  must **not** be reused to claim independent final AR/oracle test performance.
- Oracle train and pilot-validation rows are split by a stable hash of
  shard ID and rollout ID; all spans from a rollout stay in one partition.
- Current training is capped to 2,048 train rows and 256 validation rows.
- The vector is unit-normalized then scaled to alpha 8000 and overwrites the
  last rendered prompt token's embedding. The exact wording, slot, source,
  transform, alpha, model revision, split hashes, and LoRA settings are saved
  in meta.json. No inference utility should override that contract.
- Target tokens are masked to remove prompt and padded positions. Scored
  validation is cross-entropy on **the same distribution as this pilot**,
  not independent explanation fidelity.
- The checkpoint is a LoRA adapter, not a standalone inference model.
- A more faithful AO-1 comparison requires separately reconstructing
  AR(text), using the frozen AR and its whitener, plus **true vs AR-out vs
  shuffled** input controls and disjoint final evaluation data.

## Next gate

Before longer SFT training or any RL, demonstrate that the saved adapter
reloads, produces distinct generations for distinct activations, and
outperforms a shuffled-vector baseline on held-out token CE. This PR
does not claim those checks have run on GPU.
