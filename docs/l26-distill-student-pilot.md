# L26 multi-concept student LoRA: first SFT

Warm-start the completed step-500 AO-SFT LoRA and train on assembled four-bullet targets and **true layer-26 residuals** (unit-normalized, alpha=8000). The student uses Agam's **concepts_raw** wording and a chat-templated verified single-token injection slot inside activation tags, intentionally distinct from the parent's plain-text final-token prompt. Qwen3.6-27B upstream used 16000; this 4B checkpoint and its samples used 8000.

Immutable inputs: train/val targets.safetensors with SHA-256 provenance, frozen step-500 parent checkpoint. Checkpoints save trainable LoRA and optimizer state under persistent /workspace. Identical CLI arguments required on resume.

## Recover after RunPod migration

```bash
nvidia-smi
ls -ld /root/o-lens-training/.venv/bin/python /root/olens-vllm-venv/bin/python 2>/dev/null || true
ls -l /root/olens-merged/l26-step500/olens_merge_manifest.json 2>/dev/null || true
df -h /root /workspace
cd /root/o-lens-training && git status --short && git pull --ff-only
```

If /root was replaced, run scripts/bootstrap_runpod.sh again for the *training* environment. The separate scripts/bootstrap_vllm_runpod.sh is needed only for teacher inference; neither is needed if the existing environment passes import checks.

## Validation prereq

A complete 64-item teacher sample bank does NOT imply completed NNOMP/assembled validation. Check the COMPLETE marker in /workspace/o-lens-qwen35-4b/oracle/l26-vllm-nnomp-val64-v1, then create the fixed no_unigram validation assembly if missing.

```bash
cd /root/o-lens-training
S=/workspace/o-lens-qwen35-4b/oracle/l26-vllm-nnomp-val64-v1
I=/workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1
.venv/bin/python oracle/assemble_distill_bullets.py \
  --selection "$S" --items "$I" --filter-policy no_unigram \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-bullets-val64-no_unigram-v1
```

Choose a fresh destination if it already exists; do not overwrite. If validation NNOMP is missing, run select_distill_nnomp.py on the complete l26-vllm-val64-v1 samples against the frozen AR checkpoint step_001424 first.

## CPU tests and no-GPU dry run

```bash
cd /root/o-lens-training
.venv/bin/python -m unittest discover -s tests -p 'test_distill_student.py' -v
.venv/bin/python oracle/train_distill_student.py \
  --train /workspace/o-lens-qwen35-4b/oracle/l26-bullets-train512-no-unigram-v1 \
  --val /workspace/o-lens-qwen35-4b/oracle/l26-bullets-val64-no_unigram-v1 \
  --init-from /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-student-no-unigram-50step-v1 \
  --max-steps 50 --save-every 10 --batch-size 1 --grad-accum 4 --dry-run
```

Review the parent hash, injection slot, rendered prompt, train/val counts and an example bullet target.

## GPU pilot

In tmux run the identical dry-run command WITHOUT --dry-run and append:

```bash
2>&1 | tee /root/l26-student-no-unigram.log
```

Use set -o pipefail. Check INITIAL VALIDATION and step-10 metrics. To resume from the last completed checkpoint, rerun exactly the same command with --resume; do not change batch, max-steps, filter/data, or seed.

## Matched-policy comparison

To compare strict 206 versus no-unigram 374 targets, train the *strict* run on the upstream training artifact but **reuse the identical no-unigram validation artifact**, changing only --train and --output. This keeps held-out targets matched. Compare target-token CE and qualitative generations, not just assembly yield. The fixed 64-item validation is a pilot from the AR held-out pool, not independent final scientific evaluation. GRPO is not implemented.
