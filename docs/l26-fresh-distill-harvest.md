# Fresh L26 rollouts for the 4,096-item expansion

The original harvest has only 206 eligible training rollouts and 33 reservation
rollouts remaining after the fixed pilot bank is excluded. The capacity gate
correctly stopped before writes. Keep the 4,096-item experiment and obtain
new prompts; do not relax old train/validation exclusions.

## Upstream scale

Agam's [Stage 8 at a325900](https://github.com/agu18dec/olens_draft/blob/a32590085a6070d75f950155d3c2b3db3eec97ef/docs/pipeline.md#stage-8--distillation-the-multi-bullet-student)
records student datasets of 775 → 5,256 → 14,266 → **29,564 training rows**.
The final student has validation CE 1.6765. These are observed run sizes, not
a demonstrated minimum requirement, and they are not raw rollout counts.
The same document records 1,485,997 chat conversations / 798.4M output tokens
for the broader AR/AO corpus, and 16.0M spans consumed by the final chat AR.
Those are different stages. Our 4,096 raw items (fewer accepted targets) are
an intermediate scale; cross-model FVE values are not directly comparable.

## Recipe

Use the existing v3 harvester, unadapted Qwen3.5-4B, WildChat-1M first user
prompts, temperature 1, top-p 1, max 1,536 output tokens, thinking disabled.
New seed 43; capture **only L26**, at the token immediately preceding the
sampled continuation. Preserve the uniform 1–32 token span recipe. Capture
at most **one span per rollout**: breadth is the limiting resource.

Target 8,192 captured pairs, therefore at least 8,192 distinct accepted
rollouts, with small batch/shard overshoot possible. This leaves filtering
headroom for 3,584 train additions plus 256 prospective test items. Selection
still fails closed if the usable yield is lower than needed. Short spans and
degenerate text are filtered by the unchanged item selector. All new rollouts
are eligible for the distillation partition, rather than routing 90% to an
AR-training partition we will not use. Do not retrain the AR or whiteners.

Exclude whitespace-normalized first-user prompts from **every** original
rollout file. Exclusion file hashes are part of the resume contract, and a
second audit checks actual fresh prompts before bank preparation. This guards
prompt reuse, not semantic near-duplicates or all historical exposure. Keep
the test reservation labeled prospective.

Output: `/workspace/o-lens-qwen35-4b/data/l26-distill-fresh8192-v1`.
It is separate from the original harvest. Pair/rollout files commit together
via the existing shard manifest protocol. Resume skips committed prompts and
recreates incomplete shards. Exact uninterrupted RNG replay is not promised.
Changed exclusion files or a changed resolved model revision stop resume.
The wrapper uses the parent's revision when available; otherwise the resolved
new revision is recorded, but historical model identity is not retrospectively
verified. No new model-size or semantic-truth claims follow from these runs.

## Launch on the pod

After merging and pulling, use the existing training environment, not the
vLLM teacher environment. Its `datasets` package is required for full harvest.
Preflight needs no CUDA and writes nothing. Run in tmux so disconnects do not
kill the process. Cache/environment stay on `/root`; artifacts stay on
`/workspace`.

```bash
cd /root/o-lens-training
set -euo pipefail
bash scripts/harvest_l26_distill.sh preflight
bash scripts/harvest_l26_distill.sh smoke 2>&1 | tee /root/l26-harvest-smoke.log
cat /workspace/o-lens-qwen35-4b/data/l26-distill-fresh8192-v1/progress.json
```

The smoke stops after roughly 256 one-span rollouts. Inspect elapsed time,
GPU memory, prompt/output examples in the rollout JSONL, and pair/rollout
counts before extending. The legacy harvester validates finite nonzero
activations, dimensions, token alignment, and the preceding-token boundary.
The fresh Qwen GPU run itself has not been executed by this PR.

```bash
# Extends the SAME resumable harvest; does not regenerate committed shards.
bash scripts/harvest_l26_distill.sh full 2>&1 | tee /root/l26-harvest-full.log
bash scripts/run_l26_data_expansion.sh preflight
bash scripts/run_l26_data_expansion.sh prepare
```

Then continue at `merge-teacher` (if needed), `sample`, `select`, and the
remaining stages in [the expansion runbook](l26-data-expansion.md).
Do not run the harvester concurrently with item preparation. The new bank
records corpus hashes and namespace offsets, copies the original train prefix
and validation tensors exactly, and preserves the original accepted targets.

A rough generation budget for 8,192 responses at the cap is 12.6M tokens;
rejections can require more responses. Actual runtime depends on response
length and the hybrid-model kernels. Use the smoke's observed throughput to
estimate cost. Teacher candidate generation remains a separate, subsequent
229,376-continuation job; the harvest command does not start it.

## 2026-10-23: readable manifests omitted from FUSE listings

The completed harvest reported 8,432 pairs, 8,602 rollouts, 7,544,719 output
tokens and 32 shards. A subsequent manifest directory listing showed only 30:
shards 00004 and 00007 were absent, although direct reads of both JSON files
succeeded. Their data files were present. This is an observed listing/read
discrepancy, not evidence that those shards need regeneration.

Manifest discovery now combines directory names with the saved progress count
and directly reads the expected commit paths. Progress is only a discovery
hint: it cannot replace a missing or malformed manifest. Cleanup rechecks a
candidate's manifest directly before deleting any uncommitted data. If saved
progress claims a committed shard whose manifest cannot be read, cleanup
stops without removing that shard.

After pulling the fix, verify discovery without loading the model:

```bash
cd /root/o-lens-training
python3 - <<'PY'
from pathlib import Path
from legacy.harvest_manifests import scan_manifests
root = Path('/workspace/o-lens-qwen35-4b/data/l26-distill-fresh8192-v1')
records = scan_manifests(root/'manifests')
print('Committed shards:', len(records), sorted(records))
print('Pairs:', sum(m['pairs'] for m in records.values()))
PY
bash scripts/run_l26_data_expansion.sh preflight
bash scripts/run_l26_data_expansion.sh prepare
```

Expected: 32 committed shards (0–31), 8,432 pairs. No manifest reconstruction
or harvest rerun is needed for this observed incident. The existing preparation
checks still validate pair counts and reject unexplained extra data files.
