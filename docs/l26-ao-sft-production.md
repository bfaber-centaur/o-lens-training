# L26 AO SFT: fresh-crop first rung (Agam-inspired)

Reference: https://github.com/agu18dec/olens_draft/blob/main/src/oracle_lens/pipeline/ao_pool.py

This PR **replaces** the pilot-pairs AO source with seeded non-overlapping
64-token windows from raw rollout `output_ids`. One length in
{2,4,8,16,32,64} is chosen per window. Exact full-window duplicates, same-length
crop duplicates, special tokens and anonymization placeholders are dropped.
Every AO crop matching a prefix of **any L26 pairs row** is excluded, so the
frozen AR cannot have trained on precisely that crop. Conversations are assigned
to train/validation by stable rollout hash; no windows from one rollout cross
the partition. Pool metadata and AR-output tensor artifacts are checksummed.

Upstream deviations remain: Qwen3.5-4B vs 3.6-27B, single L26 AR vs joint
multi-layer AR, existing rollout provenance, training prompt and unit/alpha
from our pilot rather than Agam's later stage-specific injection contract.
This is **SFT only**, not distillation or RL. No claim of strict replication.

**First: CPU-only pool build and audit.** Requires rollout sidecars to contain
`output_ids` as token-id arrays. If they don't, builder fails closed;
we must regenerate raw rollouts, not substitute AR pair spans.

```bash
cd /root/o-lens-training
git pull --ff-only
source .venv/bin/activate
python -u oracle/build_fresh_ao_pool.py \
  --rollout-dir /workspace/o-lens-qwen35-4b/data/olens_200k_all32/rollouts \
  --pairs-dir /workspace/o-lens-qwen35-4b/data/olens_200k_all32/pairs \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-fresh-ao-pool-v1 \
  2>&1 | tee /root/l26-fresh-ao-pool.log
```

Check pool metadata for nonzero train/val counts and duplicate/exclusion
statistics before committing GPU time. There are **no pre-set 12k/1k caps**;
all eligible windows are used.

**Second: frozen AR precomputation** (GPU):

```bash
python -u oracle/precompute_fresh_ao_arout.py \
  --pool /workspace/o-lens-qwen35-4b/oracle/l26-fresh-ao-pool-v1 \
  --checkpoint /workspace/o-lens-qwen35-4b/ar/l26_200k_all32/checkpoints/step_001424 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-fresh-arout-v1 \
  2>&1 | tee /root/l26-fresh-arout.log
```

**Third: 5-step SFT smoke test** (GPU):

```bash
python -u oracle/train_ao_sft_production.py \
  --pool /workspace/o-lens-qwen35-4b/oracle/l26-fresh-ao-pool-v1 \
  --arout /workspace/o-lens-qwen35-4b/oracle/l26-fresh-arout-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-smoke-v1 \
  --max-steps 5 --save-every 2 \
  2>&1 | tee /root/l26-ao-sft-smoke.log
```

Verify checkpoints 2, 4, 5, and latest.json. Resume with **identical
flags plus** `--resume` only following interruption. Checkpoint stages
save LoRA, tokenizer, run metadata, optimizer and RNG state.

Production training requires a **new output directory**, with explicit
budget selection, e.g. `--max-steps 500 --save-every 50`. This number is a
starting budget, not established as an upstream optimal configuration.

The old `eval_ao1_arout.py` is **not compatible** with this new pool
contract; a fresh-pool evaluation adapter is still needed before interpreting
the checkpoint. Validation loss during training remains available.
Do **not** run a long production job until the smoke, resume, data provenance,
and new evaluation interface have passed review.

All artifacts go on the persistent /workspace volume. Stop the RunPod GPU
after reviewing the results; closing SSH alone does not stop billing.
