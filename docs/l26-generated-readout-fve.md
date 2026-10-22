# L26 generated-readout FVE: frozen workout scorer

`oracle/score_generated_readouts.py` scores saved workout generations. It does
not generate, train, alter assembly filters, or modify source artifacts.
The first experiment is the existing 12-item validation workout, not an
independent test. GPU scoring has not been run by the PR author; the commands
below establish the first measured baseline on the user's A100.

## Metric and comparison arms

For each parsed concept, reuse `select_distill_nnomp.embed_candidates` and
`legacy/train_ar.build_ar`: prepend the exact tokenized `[Layer 26]` tag,
then the concept token IDs (`add_special_tokens=False`). No EOS is appended;
no chat template, generation, silent truncation, or alpha scaling is applied.
The trained Qwen AR pools the last non-padding token, then applies its saved
LayerNorm and regression head. It is loaded once per invocation with pending
items, set to eval, and gradients disabled. Unique phrases within each item
are embedded in batches; duplicate positions remain in the scored basis.

With the **original** checkpoint whitener and mean:

```
x = (true_residual - mean) @ W
a_i = normalize((AR(bullet_i) - mean) @ W)
c = NNLS(a, x), constrained c >= 0
FVE = 1 - ||x - sum_i(c_i*a_i)||² / ||x||²
```

The true residual comes directly from the SHA-validated validation target bank;
it is not the normalized/scaled injection vector. Only atoms are normalized.
The scorer reuses the existing 80-iteration projected-gradient NNLS, with step
size `1/trace(Gram)`, in float32 on CPU. This preserves the upstream/local
approximate solver rather than switching to an exact solver. Prefix fits are
independent refits in displayed bullet order; increments are not clamped.
Very correlated atoms can produce small negative increments from finite solver
convergence. Nonfinite inputs and zero target energy are fatal errors, not
zero-score examples. Zero atom directions remain zero and are counted invalid.

| Output | Meaning |
|---|---|
| `construction_fve` | Stored NNOMP selection score over exact candidate token prefixes |
| `nnomp_text` | Assembled frozen four-bullet target, retokenized and rescored |
| `parent_shared` | Parent output under the shared student prompt |
| `student_shared` | Student output under the same prompt and target activation |
| `student_shuffled` | Shuffled-injection output scored against the **original target** |
| `student_shuffled.donor_diagnostic` | Same text against donor activation, separately labeled |
| `parent_native` (if present) | Optional workout arm, kept separate |

Assembly strips leading punctuation and sorts selected prefixes by length.
Consequently `construction_fve` and `nnomp_text.fve` need not match. Neither is
independent semantic ground truth. Every text arm includes a first-bullet
baseline, joint FVE after each prefix, increments, nonnegative coefficients,
parsed/eligible/scored counts, exact distinct text count, direction rank
(absolute tolerance 1e-5), mean pairwise cosine, and near-duplicate pair count
(cosine >= 0.999). Rank and cosine are geometric diagnostics, not semantic
concept counts.

## Parsing and truncation policy

Parser version: `dash_multiline_min2tokens_first4_v1`.

- Strip line-edge whitespace. Start concepts at `- ` (a bare `-` starts an
  empty concept which can receive continuation lines).
- Ignore preamble; append every subsequent non-marker line to the current
  concept, preserving internal newlines. No special scaffold/tag removal.
- Drop empty concepts and concepts with fewer than **two AR tokenizer tokens**,
  excluding the layer tag. Take the first four remaining concepts.
- Keep duplicates, and do not refill the four slots with later unique concepts.
- No paragraph fallback. Malformed/zero-bullet readouts have FVE 0 and
  `valid=false`; they remain in all-item means, with invalid counts visible.
- Use the saved EOS-trimmed text and verify it by decoding saved token IDs.
  Keep a final partial bullet when it passes the parser, even when the response
  hit its generation cap; preserve `eos`, `hit_cap`, and token count per arm.
  `hit_cap` is the original length flag and can coexist with EOS at the final
  allowed token. This scorer does not invent termination or repair text.

## Upstream comparison

Reviewed upstream `agu18dec/olens_draft` main at
[`a32590085a6070d75f950155d3c2b3db3eec97ef`](https://github.com/agu18dec/olens_draft/tree/a32590085a6070d75f950155d3c2b3db3eec97ef):

- [`docs/rl.md`](https://github.com/agu18dec/olens_draft/blob/a32590085a6070d75f950155d3c2b3db3eec97ef/docs/rl.md)
  documents multiline bullets, a two-token minimum, and frozen joint FVE.
- [`rl_reward.py`](https://github.com/agu18dec/olens_draft/blob/a32590085a6070d75f950155d3c2b3db3eec97ef/src/oracle_lens/pipeline/rl_reward.py)
  currently implements a different `split_bullets`: standalone marker lines,
  scaffolding cleanup, and a whole-prose fallback. **We follow the documented
  multiline convention and the requested strict failure handling**, not that
  helper byte-for-byte. Scores must not be described as parser-identical to it.
- [`nnomp.py`](https://github.com/agu18dec/olens_draft/blob/a32590085a6070d75f950155d3c2b3db3eec97ef/src/oracle_lens/core/nnomp.py)
  supplies the same 80-step nonnegative refit convention already used locally.
- [`marginal_bullet.py`](https://github.com/agu18dec/olens_draft/blob/a32590085a6070d75f950155d3c2b3db3eec97ef/scripts/rl/checks/marginal_bullet.py)
  motivates cumulative first-k FVE; here order is generated/assembled text order,
  not original NNOMP selection order.
- [`docs/pipeline.md`, Stage 8](https://github.com/agu18dec/olens_draft/blob/a32590085a6070d75f950155d3c2b3db3eec97ef/docs/pipeline.md)
  distinguishes true-residual distillation and the frozen reconstruction basis.

Qwen adaptation: reuse the existing single-layer, textual-layer-tag AR rather
than upstream's multilayer backbone hooks and learned layer embedding. Do not
substitute the injection transform (unit norm times 8000) for metric whitening.
No diversity reward, leave-one-out reward, contrastive subtraction, or RL is
introduced.

## Provenance and resume

The scorer checks the workout/assembly/student/parent lineage, selected rows,
strata and donor permutation, source SHA-256 values, original whitener hash,
AR config hash against the selection manifest, text/token agreement, and
checkpoint completion markers. It fingerprints all checkpoint files (including
adapter, head/training state, config, whitener), tokenizer files, relevant Python
sources, source artifacts, package versions, Python, and batch size. Hashing the
checkpoint can take time on network storage; it does not load the model.

**Historical limitation:** selection did not hash the AR adapter or regression
head. Config and whitener identity can be checked historically, but adapter/head
byte identity at selection time cannot be proven retroactively. The scorer
explicitly records `historical_ar_weights_verified=false`; full checkpoint
bytes are pinned from the first scoring invocation onward. The existing loader
also does not pin a base-model revision. `runtime.json` records the resolved
base commit (or null if unavailable), CUDA runtime and GPU, and rejects changes
on resume. This is not a claim of original-training base revision verification.

Output manifests and item checksums must match to resume. Each item is written
by atomic rename; stale unique staging files never block reruns. An OS lock
prevents simultaneous writers. `--max-new-items` is deliberately excluded from
the fingerprint, so a smoke run resumes into the full run. Completed reruns
regenerate summaries without loading the AR. Do not change source files,
checkpoint bytes, tokenizer, batch size, or environment between smoke and full
runs; choose a new output directory for a changed contract.

Outputs: `manifest.json`, `runtime.json`, per-item `item_*.json`, `report.json`,
`report.md`, and a final `COMPLETE` marker. Partial reports are labeled partial.
The Markdown reports distributions by accepted-NNOMP FVE stratum plus paired
student-minus-parent and student-minus-shuffled differences. JSON also includes
energy-pooled FVE, which differs from the unweighted item mean.

## RunPod: CPU checks, two-item smoke, then all twelve

After merging the PR manually:

```bash
cd /root/o-lens-training
git pull --ff-only
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export TMPDIR=/root/tmp
mkdir -p "$TMPDIR"

.venv/bin/python -m unittest discover -s tests -p 'test_generated_readouts.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_distill_nnomp.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_lens_workout.py' -v

score_args=(
  --workout /workspace/o-lens-qwen35-4b/oracle/l26-lens-workout12-v1
  --ar-checkpoint /workspace/o-lens-qwen35-4b/ar/l26_200k_all32/checkpoints/step_001424
  --output /workspace/o-lens-qwen35-4b/oracle/l26-lens-workout12-fve-v1
  --batch-size 16
)

# CPU provenance + tokenizer validation; no model loaded or output created.
.venv/bin/python -u oracle/score_generated_readouts.py "${score_args[@]}" --dry-run

# First two items in the frozen workout order, not a representative sample.
.venv/bin/python -u oracle/score_generated_readouts.py "${score_args[@]}" --max-new-items 2
cat /workspace/o-lens-qwen35-4b/oracle/l26-lens-workout12-fve-v1/report.md

# Resume the same output: preserve those two rows, score the remaining ten.
.venv/bin/python -u oracle/score_generated_readouts.py "${score_args[@]}"
cat /workspace/o-lens-qwen35-4b/oracle/l26-lens-workout12-fve-v1/report.md
```

Inspect invalid counts, cap hits, target-text scores versus stored construction
scores, first-bullet versus joint FVE, and paired correct-versus-shuffled results.
No threshold here declares semantic faithfulness. Keep the original greedy
160-token baseline even if later decoding controls improve it.

All environments, downloads, package caches and temporary build work stay under
`/root`. Only completed/partial resumable research artifacts go to `/workspace`.
No installation or vLLM environment change is needed for this script.

## Extending beyond the workout

`parse_bullets` and `score_vectors` are independent of workout size. The CLI
currently deliberately validates the existing workout schema and its exact
stratified selection, so it cannot silently relabel another evaluation as the
frozen 12-item baseline. For the full fixed validation set, generate a separate
saved-readout dataset with stable item IDs, true-residual bank SHA, explicit donor
IDs, and a new manifest/schema; add a validated input adapter feeding the same
scoring functions. Do not replace or reselect this workout. Reserve an untouched
independent test set before expanding training; these accepted validation strata
are not representative of every true-activation item.

## CPU verification

Synthetic tests cover multiline/CJK/duplicate parsing, actual tokenizer-token
minimum, analytic orthogonal/negative/zero cases, independent active-set NNLS
reference, whitening, nonfinite and shape errors, source/text/result corruption,
interruption/resume, a no-model completed rerun, and original-target versus donor
scoring. No GPU measurements are claimed by these tests.
