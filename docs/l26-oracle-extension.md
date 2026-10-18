# L26 oracle: bounded warm-start SFT

This is a deliberately narrow second experiment. It **continues the adapter
weights** from the successful 20-step true-activation pilot, not its optimizer
state. The optimizer starts fresh; the metadata records that distinction.
It is NOT AR(out) inversion, semantic explanation, distillation, or RL.

Source checkpoint: /workspace/o-lens-qwen35-4b/oracle/l26-gt-sft-pilot-v1

Before training, the script checks the original checkpoint COMPLETE marker,
model/layer, exact prompt, injection transform and scale, split identity and
data caps; it verifies both actual sampled-index hashes match the parent.
Refuses output-directory overwrite. Uses the same activation/prompt contract.

Run a new 100-step extension:

    cd /root/o-lens-training
    source .venv/bin/activate
    python -u oracle/train_gt_sft_extend.py \
      --init-adapter /workspace/o-lens-qwen35-4b/oracle/l26-gt-sft-pilot-v1 \
      --output /workspace/o-lens-qwen35-4b/oracle/l26-gt-sft-extend100-v1 \
      --layer 26 --max-steps 100 --batch-size 4 --grad-accum 4 \
      2>&1 | tee /root/l26-gt-sft-extend100-v1.log

This run has no intermediate checkpoint/resume, so treat 100 steps as a bounded
experimental extension, NOT a long unattended production job.

After training, use existing oracle/eval_gt_sft.py with the new checkpoint and
a new output path. Compare **correct, shuffled, zero** activation CE with the
saved 20-step baseline. Inspect the text generations; lower token CE is NOT
proof of correct semantic explanations. Explicit semantic scoring would be a
separate, preregistered evaluation design, with an independent test set.

Do not publish this as a fully trained O-Lens. The AR reward, reconstructed
activation input, multi-concept distillation and RL stages remain unimplemented.
