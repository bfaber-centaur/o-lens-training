# L26: more distillation data with fixed validation

The decoding pilot supports **greedy64** as the working readout: FVE 0.1683
versus 0.1682 at 160 tokens, with 60% fewer generated tokens. Sampling at
0.7 reached 0.1841 at 109.4 tokens; keep it as a later candidate. These are
12 selected validation items, not an independent test or semantic truth.
Historical generation defaults and artifacts remain intact.

## Frozen comparison

Compare original accepted training targets (374 from 512 raw items) against
those same targets plus accepted targets from 3,584 additional raw items.
Both runs initialize from AO step 500, use seed 42, AdamW LR 5e-5,
batch 1 × accumulation 4, and **1,000 optimizer steps**, saving every 250.
The comparison endpoint is step 1,000, not whichever checkpoint scores best.
This is a fixed compute comparison; expanded examples get fewer exposures.
Sampling with replacement gives 4,000 training draws, not a complete epoch.
The old 50-step student is a historical reference, not the matched control.

Keep the existing accepted validation targets and selection report unchanged.
Primary exploratory readout is paired per-item frozen-AR FVE on the same 12
workout items at greedy64, plus original-target shuffled controls, validity,
first/later-bullet FVE and cap rate. Report all items, including invalid zeros.
Do not infer significance from aggregate means or treat decoding seeds as
independent observations. This repeatedly inspected validation set supports
iteration; it cannot certify generalization. An eventual independent test
needs a historical exposure audit and its own frozen evaluation plan.

## Expansion contract

`prepare_distill_items.py --base-items ... --n-train 4096` preserves the old
512 training entries as an exact prefix and copies all validation tensors.
It excludes every old train/validation rollout when selecting additions,
reserves 256 additional rollouts under `test_*`, and records corpus hashes.
The reserved pool is **prospective**: prior AO exposure has not been audited.
It is neither sampled nor assembled nor used for checkpoint selection here.
Distinctness is by `(shard, rollout)` in the existing harvest; this does not
detect duplicate text or conversations across different rollout IDs.

Preflight checks old source rows against the current corpus, including actual
vectors and span tokens, and selects enough eligible distinct new rollouts.
It fails if capacity is insufficient. Do not change the partition or include
validation to fill the quota. Review inventory and choose a smaller documented
run or harvest more data. This code does not assume 4,096 fit the existing pool.

Only entries 512–4095 get fresh 64×100 teacher sampling and NNOMP. Original
accepted targets are reused without regeneration. Assembly uses no_unigram,
minimum four picks, and the same frozen AR checkpoint. The combine command
checks source indices/vectors, target policies and expansion lineage. The
trainer permits different train/validation bank hashes only after verifying
this lineage, the preserved validation tensors, and rollout disjointness.

## RunPod execution

Merge this PR and update `/root/o-lens-training`. If root was wiped, rebuild
`.venv` with `scripts/bootstrap_runpod.sh` and the separate vLLM environment
with `scripts/bootstrap_vllm_runpod.sh`. Inspect those scripts' requirements
before running. All research artifacts below persist under `/workspace`;
models, environments and caches use `/root`.

```bash
cd /root/o-lens-training
set -euo pipefail
bash scripts/run_l26_data_expansion.sh preflight
bash scripts/run_l26_data_expansion.sh prepare
# Only if the ephemeral merged teacher does not already exist:
bash scripts/run_l26_data_expansion.sh merge-teacher
bash scripts/run_l26_data_expansion.sh sample
bash scripts/run_l26_data_expansion.sh select
bash scripts/run_l26_data_expansion.sh assemble
bash scripts/run_l26_data_expansion.sh check-control
bash scripts/run_l26_data_expansion.sh check-expanded
bash scripts/run_l26_data_expansion.sh train-control
bash scripts/run_l26_data_expansion.sh train-expanded
bash scripts/run_l26_data_expansion.sh evaluate
```

The script assumes the original bank contains exactly 512 training items;
verify its `meta.json` before sampling. Teacher sampling is 229,376
continuations, capped at 22,937,600 generated tokens; runtime is unknown until
measured on the pod. This PR does not execute GPU work.

Sampling and selection can resume by rerunning their commands. Training can
resume with `train-control --resume` or `train-expanded --resume`. Preparation,
assembly and combine refuse existing destinations. If assembly finished but
combine did not, rerun `oracle/combine_distill_targets.py` directly using the
paths in the script. Preserve completed artifacts; do not delete original
banks to recover an interrupted run. A partial preparation/assembly must be
inspected and moved aside before retrying to a new empty destination.

Final reports:
- `oracle/l26-data-control-1000-fve-v1/report.md`
- `oracle/l26-data-expanded-1000-fve-v1/report.md`

Both paths are relative to `/workspace/o-lens-qwen35-4b`.
