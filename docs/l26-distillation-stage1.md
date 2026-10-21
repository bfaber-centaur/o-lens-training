# L26 O-Lens distillation: true-activation items → teacher samples

Reference: Agam's [Stage 8](https://github.com/agu18dec/olens_draft/blob/main/docs/pipeline.md)
and `scripts/distill/ao_distill_sample.py`.

**This PR implements the first two of five distillation stages**, without
starting GPU work:

1. Prepare true L26 residuals and persist rollout-disjoint source indices
2. Sample candidate continuations with the trained AO SFT teacher
3. TODO: embed candidates with the frozen L26 AR, whiten, NNOMP-4 + NNLS
4. TODO: assemble and filter four-bullet targets
5. TODO: warm-start the student LoRA and train on **true** residuals

Do not claim that teacher sampling alone is a distilled O-Lens.

## Inputs and integrity

Teacher: completed production SFT checkpoint
`/workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500`.
Only its complete `lora/`, `tokenizer/`, `meta.json`, and `COMPLETE`
are needed. Source: L26 **true residuals** in the existing 200k full-depth
pair harvest. Draw from the AR **held-out** pool, one eligible span per
rollout; deterministic rollout-level 85/15 train/val split for distillation.
Use upstream F1 degeneracy filtering and persist row/rollout IDs and checksum.
These are pilot distillation splits, **not** independent final evaluation:
the earlier AO pilot also used some AR-held-out examples.

Teacher prompt, replacement slot, transform and alpha come **exclusively**
from its checkpoint meta: `Read the following representation ...`,
last prompt embedding slot, `unit`, **α=8000**. Do **not** substitute Agam's
later `continuation_raw` prompt/α=16000 contract: our teacher has never
been trained under it. This is an explicit 4B-lineage deviation.
Upstream's stage-8 sample shape is 64 continuations per residual,
temperature=1.0, top-p=.95, max=100 tokens. Samples are saved as per-item
JSON with token IDs/text and source row IDs, with hash-bound manifest and
atomic, resumable output. Verify adapter tensors map to live LoRA weights
before generating.

## CPU preparation (next pod, no heavy model inference)

```bash
cd /root/o-lens-training
source .venv/bin/activate
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
python -m unittest discover -s tests -p 'test_distill_teacher.py' -v

python -u oracle/prepare_distill_items.py \
  --teacher /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 \
  --n-train 512 --n-val 64 \
  2>&1 | tee /root/l26-distill-items.log
```

Check `meta.json` for train=512, val=64, seed=42, layer=26 and
`items_sha256`, before running any inference.

## Check-only then **very small GPU smoke**

```bash
COMMON="--teacher /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500 --items /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 --output /workspace/o-lens-qwen35-4b/oracle/l26-distill-teacher-smoke-v1 --split train --count 2 --n-samples 4 --max-new-tokens 24 --sample-batch 2"
python oracle/sample_distill_teacher.py $COMMON --dry-run
python -u oracle/sample_distill_teacher.py $COMMON \
  2>&1 | tee /root/l26-distill-teacher-smoke.log
```

Read the two `item_*.json` files. Check 4 candidates each, sensible
token/text decoding, and `adapter verified: N/M tensors live`. If the
transformers + Qwen3.5 hybrid `model.generate(inputs_embeds=...)` path
does not handle the soft-token embedding/caching correctly, **fix that
before scaling**. The sampling kernel is deliberately unproven on GPU.

## After smoke: actual teacher sample run

Use a **different new output directory** for each split. For the first
512 train items, with Agam's full 64×100 sampling budget:

```bash
python -u oracle/sample_distill_teacher.py \
  --teacher /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500 \
  --items /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-distill-teacher-train512-v1 \
  --split train --count 512 --n-samples 64 --max-new-tokens 100 --sample-batch 8 \
  2>&1 | tee /root/l26-distill-teacher-train512.log
```

**This is a materially larger generation job.** Don't start until sample
quality, throughput, padding/marker behavior and artifact parsing have passed
the small smoke. If interrupted, rerun the identical command; complete item
JSONs are skipped after verifying the manifest. The validation set needs a
separate sampling run with `--split val --count 64 --output <new-path>`.

No NNOMP/student in this PR. When implementing those stages, preserve
Agam's frozen AR whiteners and true-residual query; select k=4 complementary
candidates with NNOMP+NNLS and assemble tag-free `- ` bullet targets.
The distillation student must read *true* activations.

Persistent output lives under `/workspace`; caches and environment remain
on ephemeral root disk. Stop the GPU rental after stage completion.
