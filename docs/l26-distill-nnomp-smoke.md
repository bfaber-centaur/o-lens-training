# L26 distillation: two-item NNOMP smoke (selection only)

Upstream reference: Agam's
[`scripts/distill/ao_gt_omp_readout.py`](https://github.com/agu18dec/olens_draft/blob/main/scripts/distill/ao_gt_omp_readout.py),
[`r2_select.py`](https://github.com/agu18dec/olens_draft/blob/main/src/oracle_lens/pipeline/r2_select.py)
and [`nnomp.py`](https://github.com/agu18dec/olens_draft/blob/main/src/oracle_lens/core/nnomp.py).

This is a **selection-stage smoke**, not distillation student training.
From already saved **64-sample × 2-item** teacher generations:

1. Draw unique candidates from token prefixes at 2/4/8/16/32/64 (upstream
   F1 degeneracy filter).
2. Embed all candidates with the **completed frozen L26 AR**.
3. Apply the original train-only frozen L26 whitener to both AR images and
   the **true L26 activation** query; normalize the whitened AR rows but
   **do not unit-normalize the query**.
4. Greedily choose up to 4 candidate directions with positive correlation
   to the current residual; after every pick refit nonnegative coefficients
   using projected-gradient NNLS (same 80 iterations and 1/trace(Gram)
   step convention as the upstream core).
5. Save source candidate IDs, text, picks, coefficients and **joint FVE**.
   No shortest-prefix shrinking (retired from Agam's production data path).

The selection consumes only saved teacher candidates; no new teacher
generation or fine-tuning.

## Tomorrow's bounded test

On the next A100 pod, after the PR is merged and the venv is bootstrapped:

```bash
cd /root/o-lens-training
git pull --ff-only
source .venv/bin/activate
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
python -m unittest discover -s tests -p 'test_distill_nnomp.py' -v

COMMON="--samples /workspace/o-lens-qwen35-4b/oracle/l26-distill-teacher-fullshape-smoke-v1 \
--items /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 \
--ar-checkpoint /workspace/o-lens-qwen35-4b/ar/l26_200k_all32/checkpoints/step_001424 \
--output /workspace/o-lens-qwen35-4b/oracle/l26-distill-nnomp-smoke-v1"

python -u oracle/select_distill_nnomp.py $COMMON --dry-run
python -u oracle/select_distill_nnomp.py $COMMON \
  2>&1 | tee /root/l26-distill-nnomp-smoke.log
```

Inspect report.json. For each item, expect `n_candidates >= 4`,
up to four distinct `omp_sel` candidate texts, nonnegative coefficients,
and finite `omp_fve`. The selected concepts do **not** have to be
semantically faithful on this smoke; this gate validates candidate
embedding and reconstruction selection.

Notes:
- Each vector must come from the original L26 AR checkpoint and frozen
  `whitener_l26.safetensors`; do not refit the whitener.
- The final student will consume **true** residuals, not AR images.
- Upstream uses Qwen3.6-27B and a joint AR; this is an L26 4B adaptation.
- No GPU execution has been performed by this PR.
- The existing teacher code samples via HF generate; optimize and benchmark
  that separately before scaling to hundreds of items.
