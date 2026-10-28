# L30 start

L26 smoke and HF publication completed, reported by Bobby on 2026-10-27.
Release: https://huggingface.co/bubskee/qwen3.5-4b-o-lenses.
L26 stays frozen; card shortening is deferred.

Reuse L30 AR step 1412 and its original 89-shard
`olens_200k_l22_l26_l30` corpus. No new harvest or AR training.
First stage: fresh AO crop pool, frozen AR outputs, five-step AO SFT smoke.
The L26 adapters are not L30 initialization. Later distillation needs its own
layer-aware contracts before running; the L26 distillation runner is not ready
for L30. Pain-axis examples remain external diagnostics, outside fitting.

After merging and pulling, run the CPU preflight:
```bash
cd /root/o-lens-training
LAB=/workspace/o-lens-qwen35-4b
mkdir -p "$LAB/logs"
set -o pipefail
.venv/bin/python scripts/preflight_l30.py \
  --output "$LAB/oracle/l30-source-receipt-v1.json" \
  2>&1 | tee "$LAB/logs/l30-preflight.log"
```

If sources are missing, stop and locate them; do not substitute all32 silently.
The receipt hashes explicitly named files, including AR head and adapter.
These freeze present identity, not an unrecorded historical identity.

Build and inspect the CPU pool before GPU work:
```bash
.venv/bin/python -u oracle/build_fresh_ao_pool.py --layer 30 \
  --rollout-dir "$LAB/data/olens_200k_l22_l26_l30/rollouts" \
  --pairs-dir "$LAB/data/olens_200k_l22_l26_l30/pairs" \
  --source-config "$LAB/ar/l30_200k_v3/checkpoints/step_001412/run_config.json" \
  --output "$LAB/oracle/l30-fresh-ao-pool-v1" \
  2>&1 | tee "$LAB/logs/l30-pool.log"
```
Pool metadata must enumerate all 89 source pair shards and 89 rollout shards.
The source configuration supplies exact filenames, bypassing incomplete directory listings. Prefix exclusion is against L30 pair spans.

After that inventory agrees:
```bash
.venv/bin/python -u oracle/precompute_fresh_ao_arout.py --layer 30 \
  --pool "$LAB/oracle/l30-fresh-ao-pool-v1" \
  --checkpoint "$LAB/ar/l30_200k_v3/checkpoints/step_001412" \
  --output "$LAB/oracle/l30-fresh-arout-v1" --batch-size 8 \
  2>&1 | tee "$LAB/logs/l30-arout.log"

.venv/bin/python -u oracle/train_ao_sft_production.py --layer 30 \
  --pool "$LAB/oracle/l30-fresh-ao-pool-v1" \
  --arout "$LAB/oracle/l30-fresh-arout-v1" \
  --output "$LAB/oracle/l30-ao-sft-smoke-v1" \
  --max-steps 5 --save-every 2 --batch-size 1 --grad-accum 16 \
  2>&1 | tee "$LAB/logs/l30-ao-smoke.log"
```
Same effective batch 16 and unit/alpha8000 as the L26 AO stage, with a smaller
microbatch for A40. Review finite losses and checkpoints 2/4/5 before a production
run. All artifacts and logs persist on /workspace. GPU steps have not been
executed by the remote author.
