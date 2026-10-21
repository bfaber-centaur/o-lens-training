# L26 distillation — upstream-style bullet assembly

This is a CPU-only stage on the path to the multi-concept student.
Reference: Agam's ao_assemble_distill.py in agu18dec/olens_draft.
Input is our saved NNOMP selection report and true-L26 residual bank.

The assembler:
- Trims leading punctuation, sorts selected spans shortest-first, emits
  four lines prefixed '- ', and appends the tokenizer EOS token.
- Applies Agam's filter-of-record checks in their original order:
  >=25 candidates; exactly four selections; >=2 content units each;
  no U+FFFD, repeated 1/2/3-grams, low alphabetic ratio, embedded '- '
  bullet lines, prefix containment, or lexical Jaccard >= 0.45.
- Rejects mismatched GT item checksum, incorrect activation source/layer,
  duplicate or missing sample indices, incomplete teacher, and existing
  destination. Uses original TRUE residuals, not candidate AR images.
- Outputs audit.json (including rejected items), meta.json and, when any
  items pass, targets.safetensors with ragged target token IDs, offsets,
  true L26 vectors, and row provenance.
- Does not require CUDA or instantiate the base model.

Run this after PR merge, with the pod's existing venv:

    cd /root/o-lens-training
    git pull --ff-only
    source .venv/bin/activate
    export HF_HOME=/root/.cache/huggingface
    export HF_HUB_CACHE=/root/.cache/huggingface/hub
    python -m unittest discover -s tests -p 'test_distill_bullets.py' -v
    python -u oracle/assemble_distill_bullets.py \
      --selection /workspace/o-lens-qwen35-4b/oracle/l26-distill-nnomp-smoke-v1 \
      --items /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 \
      --output /workspace/o-lens-qwen35-4b/oracle/l26-distill-bullets-smoke-v1 \
      2>&1 | tee /root/l26-distill-bullets-smoke.log

Inspect the counts and rejection reasons:

    cat /workspace/o-lens-qwen35-4b/oracle/l26-distill-bullets-smoke-v1/meta.json
    cat /workspace/o-lens-qwen35-4b/oracle/l26-distill-bullets-smoke-v1/audit.json

With only two NNOMP items, zero examples surviving is an entirely valid
outcome. In that case the assembler records the audit and COMPLETE but does
not create targets.safetensors. Never disable upstream filtering to force
the smoke to pass.

The subsequent student trainer will read targets.safetensors, warm-start
the step-500 SFT adapter, use the concepts_raw instruction and true
activation unit injection. This PR does NOT implement the trainer, teacher
sampling scale-up, or RL; do not call these outputs a distilled checkpoint.

We preserve the source train/val split from the true-activation bank rather
than making an arbitrary new 1-in-20 split over two smoke items. At full
scale, use separate train and val source selections and keep validation fixed.

This is our L26 4B adaptation, not a direct checkpoint reproduction of
the upstream joint-layer Qwen3.6-27B recipe. Do not change the injection
contract or rollout strategy without a recorded stage transition.
