# L26 Lens Workout v1 — twelve fixed, inspectable activations

An instrument test, not an interpretability-quality score. This evaluates the
first Qwen3.5-4B single-layer distilled O-Lens LoRA before more training.

## Protocol

- Source: **val** true-residual bank, filtered by the already-frozen
  `no_unigram` assembly (the same 64-item validation split the student
  saw for CE measurement). Not an independent final test.
- Stratification: sort accepted validation rows by frozen NNOMP joint FVE,
  split into three equal-size groups (terciles), seed 42 to sample **four**
  rows per group. Selection and provenance persist in `manifest.json`.
- Every sampled true L26 vector is unit-normalized × **alpha=8000** at the
  same verified chat-templated `concepts_raw` injection slot for both
  adapters. Greedy decoding, max 160 generated tokens.
- Readouts: frozen 4-bullet NNOMP text target; **parent** 500-step AO-SFT
  LoRA under the *student* prompt; **student** step-50 LoRA under the identical
  prompt and activation; **student shuffled-activation control**, where the
  injected vector is a different selected validation item with no fixed points.
- Optional `--include-parent-native` adds a separately labeled parent
  generation with its own original plain-text prompt, for ecological context.
  That arm is *not* a like-for-like weight comparison.
- Auto diagnostics measure **bullet formatting only**: bullet count, whether
  there are 4 bullets, number of unique bullet lines, token cap, EOS. They do
  NOT measure concept accuracy, semantic complementarity, faithfulness, or
  reconstruction.

## Verify the CPU-only selection (no model loaded)

```bash
cd /root/o-lens-training
git pull --ff-only
.venv/bin/python -m unittest discover -s tests -p 'test_lens_workout.py' -v

ROOT=/workspace/o-lens-qwen35-4b/oracle
.venv/bin/python -u oracle/lens_workout.py \
  --selection "$ROOT/l26-vllm-nnomp-val64-v1" \
  --val "$ROOT/l26-bullets-val64-no_unigram-v1" \
  --parent "$ROOT/l26-ao-sft-500-v1/checkpoints/step_000500" \
  --student "$ROOT/l26-student-no-unigram-50step-v1/checkpoints/step_000050" \
  --output "$ROOT/l26-lens-workout12-v1" \
  --dry-run
```

This validates checkpoint contracts, adapter hashes, target SHA, exact
student prompt token sequence/slot, target membership in selected NNOMP
rows, and shows the FVE-stratified chosen indices and two frozen targets.

## Generate the three comparison arms

In tmux, use the **same command without** `--dry-run`:

```bash
cd /root/o-lens-training
set -o pipefail
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
ROOT=/workspace/o-lens-qwen35-4b/oracle

.venv/bin/python -u oracle/lens_workout.py \
  --selection "$ROOT/l26-vllm-nnomp-val64-v1" \
  --val "$ROOT/l26-bullets-val64-no_unigram-v1" \
  --parent "$ROOT/l26-ao-sft-500-v1/checkpoints/step_000500" \
  --student "$ROOT/l26-student-no-unigram-50step-v1/checkpoints/step_000050" \
  --output "$ROOT/l26-lens-workout12-v1" \
  2>&1 | tee /root/l26-lens-workout12.log
```

The base Qwen3.5-4B is loaded once; both frozen LoRAs are named adapters
switched per readout. Exactly **three generations per item** are sampled
greedily, so 36 total (plus an optional native-parent arm). The code writes
one atomic `item_NNNNNN.json` after each finished item to /workspace;
rerunning the identical command skips validated completed items. Any altered
selection, checkpoint, injection or decoding parameter needs a *fresh*
versioned output directory.

Completed output:

- `manifest.json`: exact inputs, hashes, item indices, shuffle mapping,
  FVE strata, decoding and prompt contract.
- `item_NNNNNN.json`: original 4-bullet target + each arm's actual token IDs
  and decoded text.
- `report.json`: complete machine-readable joined data + formatting metrics.
- `report.md`: portable side-by-side narrative.
- `report.html`: self-contained responsive inspection page, escaped text,
  no external scripts/assets. Copy it to your Mac or host on a static site.
- `COMPLETE`: written only after all selected rows and reports are saved.

## How to evaluate the telescope

Review each activation for:

1. **Structural learning:** does the student produce several parseable bullets
   under the shared prompt, while the parent mostly continues text?
2. **Activation sensitivity:** is the student's correct-activation response
   materially different from its shuffled-activation response? Failure here
   is a red flag for generic bullet formatting.
3. **Complementarity:** do bullets capture *different* plausible aspects or
   repeat one fragment four times?
4. **Text artifacts:** do obvious NNOMP prefix truncations bleed into the
   student's generations?
5. **FVE dependence:** is there an apparent qualitative trend from low to
   high NNOMP FVE? Beware 12-item cherry-picking and subjective judgment.

Treat the NNOMP targets as **construction artifacts**, not objective ground
truth of what the activation means. The fixed validation split contributed
to checkpoint selection. A later evaluation should reserve independent
rollouts and use controls such as shuffled/zero vectors more formally.

## Next direction

Once it passes the first workout, repeat on a separately reserved batch,
add user-provided activations, and turn the static report into a hosted
interactive lens instrument with model/layer selectors, prompts,
alternative hypotheses, and downloadable provenance. That's the bridge
to a broader lens suite and eventual Neuronpedia integration.
