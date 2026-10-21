# Qwen3.5-4B L26 distillation teacher sampling in vLLM

Reference method: Agam's Stage 8, especially scripts/distill/ao_distill_sample.py
in https://github.com/agu18dec/olens_draft . This is a **drop-in replacement
for our slow HF generate-based sampling stage**, not a new scientific objective.

vLLM's offline API supports tensors passed as prompt_embeds and n=64 samples
per item. Pin this implementation to vLLM 0.32.0, released 2026-10-18.
We have NOT yet run this against Qwen3.5-4B on an A100; the first 2-item
GPU smoke, merged-checkpoint load and output-format checks remain gates.

## Design / upstream contract

- Teacher: completed L26 AO SFT step_000500.
- Frozen prompt: exactly the teacher checkpoint's prompt string and tokenizer.
- Activation input: raw **true** L26 residual from gt_distill_items_v1, transformed
  with h / ||h|| * alpha=8000, replacing the **last prompt embedding**.
  Upstream's 27B α=16000/continuation_raw prompt MUST NOT be substituted.
- Draw 64 completions per item, temperature 1.0, top_p .95, max 100 tokens.
- Merge LoRA into its HF base once, save to an ephemeral LOCAL disk directory,
  verify adapter liveness and pre/post merge next-token logits cosine/argmax,
  then read text embeddings from the saved, checksummed shard.
- vLLM's prompt embeds are the WHOLE already rendered prompt, not a new chat
  template. NO double tokenization or injected placeholder shift.
- A request gets a deterministic per-item seed independent of shard layout.
- Outputs mimic original oracle/sample_distill_teacher.py: manifest.json,
  item_NNNNNN.json with {"item_idx","split","pair_row","n_samples","samples":
  [{"ids":[...],"text":"..."}]}. Compatible with select_distill_nnomp.py.
- Atomic per-item writes; identical manifest required to resume. Four workers
  can independently process four disjoint shards into separate directories.
- throughput.json records load time separately from generated-token throughput.

## One-time prep on RunPod: merge in the old SFT venv

The merge should run in the known-good HF/PEFT environment. **Do not merge on
the /workspace FUSE mount**. Local disk needs roughly 12 GiB free beyond
the existing model cache. The original LoRA and corpus remain persistent.

    cd /root/o-lens-training
    source .venv/bin/activate
    export HF_HOME=/root/.cache/huggingface
    export HF_HUB_CACHE=/root/.cache/huggingface/hub
    export XDG_CACHE_HOME=/root/.cache

    python -m unittest discover -s tests -p 'test_distill_vllm.py' -v

    python -u oracle/merge_teacher_vllm.py \
      --teacher /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500 \
      --output /root/olens-merged/l26-step500 \
      2>&1 | tee /root/olens-vllm-merge.log

Expected: "adapter verified: 496/496", merged logits cos >= 0.999,
argmax matches, merged model *.safetensors, tokenizer, and
olens_merge_manifest.json. The merge refuses existing outputs to avoid
mixing checkpoint lineages. This checkpoint is ephemeral: if you destroy
the pod it must be regenerated (or explicitly copied to a persistent
artifact destination for staging to the next pod).

## Separate vLLM environment

    bash scripts/bootstrap_vllm_runpod.sh

This installs pinned vLLM 0.32.0 CUDA 12.9 in /root/olens-vllm-venv,
**not** .venv. The A100 driver must support CUDA 12.9; check nvidia-smi.
If the wheel or driver is incompatible, stop and fix environment rather
than downgrading/upgrading the existing SFT environment.
The install snapshots /root/olens-vllm-env.txt.

## Cheap contract dry-run (no model load)

    /root/olens-vllm-venv/bin/python oracle/sample_distill_teacher_vllm.py \
      --teacher /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500 \
      --items /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 \
      --merged /root/olens-merged/l26-step500 \
      --output /workspace/o-lens-qwen35-4b/oracle/l26-vllm-smoke4-v1 \
      --count 2 --n-samples 4 --max-new-tokens 24 --chunk-items 2 \
      --dry-run

## First GPU smoke: 2 items x 4 samples x 24 tokens

Run the exact command above WITHOUT --dry-run in a tmux session, with
tee and set -o pipefail. Model loading is the uncertain compatibility
gate: Qwen3.5 hybrid recurrent kernels and the prompt-embeds path must
work together under the vLLM build. Default engine config uses
tensor_parallel_size=1, max_num_seqs=128, bf16 and 85% GPU memory.
If supported and useful, --language-model-only asks vLLM to skip the
multimodal tower (not enabled by default; test separately).

Inspect both item files: count=4 each, token_ids only GENERATED tokens
(no prompt prefix), nonempty decoded text, correct item_idx/pair_row.
Confirm text is conditioned on genuine input: compare rough topics to the
prior HF sampler for those same two items. Do not require identical random
samples across HF and vLLM (different samplers and seeds).

## Full-shape speed benchmark (2 items x 64 x 100)

Use a NEW output path (manifest differences forbid mix):
    
    /root/olens-vllm-venv/bin/python oracle/sample_distill_teacher_vllm.py \
      --teacher /workspace/o-lens-qwen35-4b/oracle/l26-ao-sft-500-v1/checkpoints/step_000500 \
      --items /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 \
      --merged /root/olens-merged/l26-step500 \
      --output /workspace/o-lens-qwen35-4b/oracle/l26-vllm-fullshape2-v1 \
      --count 2 --n-samples 64 --max-new-tokens 100 \
      --chunk-items 2 --max-num-seqs 128 \
      2>&1 | tee /root/olens-vllm-fullshape2.log

Inspect throughput.json. The measured **generated_tokens/new_generation_seconds**
excludes engine load, so we can size the 512-item job without extrapolating
the full wall time including cold startup.

## Scale after smoke only

First batch, one GPU, 512 train items, resumable and sharded:
    
    ... sample_distill_teacher_vllm.py \
      --items ... --merged ... --teacher ... \
      --output /workspace/o-lens-qwen35-4b/oracle/l26-vllm-train-shard0-v1 \
      --split train --start 0 --count 512 --n-shards 4 --shard 0 \
      --n-samples 64 --max-new-tokens 100 --chunk-items 2

Replicate for shards 1..3 only when there are four independent GPUs,
giving EACH its own output directory. Run NNOMP against a shard directory
to consume its items: source manifest includes its exact start and count.

Restart identical commands to skip completed items. Never merge per-shard
outputs into one directory without a distinct manifest/assembly step.

CAUTION: 512 items x 64 candidates is 32,768 sequences. GPU time still
depends on actual throughput; there is no wall-time promise. Check
sample quality, repetition, engine crashes, EOS handling and real
generated-token throughput before scaling. Preserve all original
HF outputs for comparative diagnosis. This stage does NOT implement
NNOMP, bullet assembly, student SFT, or RL.
