# L26 decoding comparison: fixed validation pilot

Question: can inexpensive decoding controls reduce token cost and repetitive
output while preserving activation-specific frozen-AR reconstruction?
Repetition alone is not a failure of the lens. A token limit is a legitimate
operating choice; the experiment measures the resulting FVE/cost tradeoff.

## Frozen starting observation (2026-10-22)

User-supplied completed report from
`/workspace/o-lens-qwen35-4b/oracle/l26-lens-workout12-fve-rebuilt-v1`:

| Arm | Mean FVE | First bullet FVE | Valid | Cap hits |
|---|---:|---:|---:|---:|
| NNOMP assembled target text | 0.2556 | 0.1296 | 12/12 | 0 |
| Parent, shared prompt | 0.0001 | 0.0000 | 4/12 | 1 |
| Student, correct activation | 0.1682 | 0.1306 | 12/12 | 12 |
| Student, shuffled activation | 0.0003 | 0.0001 | 12/12 | 12 |

Student correct-activation means by low/mid/high accepted-NNOMP stratum:
0.0916 / 0.1486 / 0.2646. All 12 correct-activation scores exceeded all 12 shuffled
scores (minimum correct 0.0344, maximum shuffled 0.0011). Rounded paired mean
student-minus-shuffled difference was 0.1680. These values were pasted by the
operator, not independently fetched from the pod. Preserve its complete JSON,
manifest and runtime records on the network volume.

## Design committed before new generations

Use the exact same checkpoint, prompt, injection slot, unit-residual alpha=8000,
12 target IDs, strata, donor permutation, AR and whitener. The unit is an
activation item; this is an exploratory paired pilot on selected validation
examples, not a representative or independent test set. No new training.

| Profile | Decoding | Token cap | Runs per item per injection |
|---|---|---:|---:|
| greedy160 | Greedy, repetition penalty 1.0 | 160 | 1 |
| greedy64 | Identical greedy, shorter cap | 64 | 1 |
| rep110 | Greedy, repetition penalty 1.1 | 160 | 1 |
| sample070 | Temperature 0.7, top-p 0.9, top-k disabled | 160 | 3 (seeds 11/22/33) |
| stop4 | Greedy, stop when fifth bullet marker begins | 160 | 1 |

Each profile runs under both correct and shuffled injection: **168 new
completions**, at most 24,576 generated tokens. Old parent/student/shuffled
outputs are copied byte-for-byte as nested output records in the new workout;
original source directories are untouched. Fresh greedy160 is a concurrent
baseline and checks exact token agreement with old greedy after rebuilding the
pod environment. If these differ, use the fresh paired baseline and inspect the
runtime record; do not attribute environment drift to a decoding intervention.

Run order is deterministically randomized within each item. Every job resets its
seed using a stable SHA-derived item/profile value, so interrupted runs do not
change sampling. Correct/shuffled conditions share a seed. The three sampled
readouts are averaged **within each item** before comparing profiles; never
select best-of-three by FVE or count 36 readouts as 36 independent activations.
There are no significance or generalization claims at this stage.

Primary readouts: paired FVE change versus current greedy160, correct-versus-
shuffled FVE, and generated token count. Secondary: valid fraction, scored and
unique bullet counts, incremental gain beyond the first bullet, cap frequency,
EOS and explicit stopping reasons. Summaries include strata, per-item values,
and seed-level values in the underlying score report. Token counts include EOS
and the fifth-marker detection overhead; they are not wall-clock benchmarks.

`stop4` leaves the first four bullet bodies unchanged and stops after observing
an initial fifth marker. The marker remains in the saved raw generation but is
empty/too-short or beyond the scorer's four-bullet limit. This controls waste;
it does not create diversity. It counts raw markers, not eligible parsed
concepts, so short/empty early bullets may leave fewer than four scored bullets.
The global 160 cap still applies. A shorter token cap can cut the fourth bullet
and change its score; quantify that cost rather than assume it is harmless.

Repetition penalty uses the standard generation implementation. With our
`inputs_embeds` call the token history begins at generated tokens; the injected
vector is not a penalized token. Penalty may also suppress useful repeated
words or bullet markers. All profiles explicitly use one beam and disable
no-repeat-ngram constraints. Model generation config and resolved revision are
recorded, along with adapters, tokenizer, source/code hashes and packages.

## Run on the rebuilt pod after manual PR merge

```bash
cd /root/o-lens-training
git pull --ff-only
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export TMPDIR=/root/tmp
mkdir -p "$TMPDIR"

.venv/bin/python -m unittest discover -s tests -p 'test_decoding_comparison.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_generated_readouts.py' -v

LAB=/workspace/o-lens-qwen35-4b
DEC="$LAB/oracle/l26-decoding-comp-v1"
FVE="$LAB/oracle/l26-decoding-comp-fve-v1"
gen_args=(
  --workout "$LAB/oracle/l26-lens-workout12-v1"
  --ar-checkpoint "$LAB/ar/l26_200k_all32/checkpoints/step_001424"
  --output "$DEC"
)
.venv/bin/python -u oracle/compare_decoding.py generate "${gen_args[@]}" --dry-run
.venv/bin/python -u oracle/compare_decoding.py generate "${gen_args[@]}" --max-new-items 2
# After the two-item generation smoke succeeds, resume the remaining ten:
.venv/bin/python -u oracle/compare_decoding.py generate "${gen_args[@]}"

score_args=(
  --workout "$DEC"
  --ar-checkpoint "$LAB/ar/l26_200k_all32/checkpoints/step_001424"
  --output "$FVE"
  --batch-size 16
)
.venv/bin/python -u oracle/score_generated_readouts.py "${score_args[@]}" --max-new-items 2
.venv/bin/python -u oracle/score_generated_readouts.py "${score_args[@]}"
.venv/bin/python -u oracle/compare_decoding.py summarize --workout "$DEC" --scores "$FVE"
```

Generation and scoring are separate processes, so the oracle and AR are never
resident together. Generation loads the student once per invocation and commits
one full item at a time (an interruption can lose at most that item's 14 jobs).
An OS lock, row checksums and source/runtime manifests reject mixed resumes.
Scoring uses the unchanged frozen parser and NNLS metric; the only scorer
extension adds donor diagnostics for the new named shuffled arms. All new arms
are always scored against the original target. No scorer outputs from previous
code versions should be resumed under this changed source fingerprint; use the
new FVE directory above.

The scoring CLI requires a complete generated workout, so finish generation
before beginning the scoring smoke. Final comparison files are
`$FVE/decoding_summary.md` and `decoding_summary.json`; detailed per-seed outputs
and FVE are in the generated workout and scorer's per-item JSON.

No GPU job was run during implementation. CPU tests cannot certify the Qwen
CUDA generation path; the two-item generation smoke is the first runtime check.
Choose an operating point from the FVE/token-cost tradeoff, and verify any
apparent improvement on the larger fixed validation set before more tuning.

## What the literature supports

- Holtzman et al., *The Curious Case of Neural Text Degeneration* (ICLR 2020),
  https://arxiv.org/abs/1904.09751: decoding alone changes repetition and text
  quality in a fixed model. This supports a decoding intervention, not a universal
  small-versus-large scaling law.
- Xu et al., *Learning to Break the Loop* (2022),
  https://arxiv.org/abs/2206.02369: sentence repetition can reinforce its own
  probability, especially under greedy/maximization decoding.
- Zhang, Das and Zhang, *Verbosity != Veracity* (2024),
  https://arxiv.org/abs/2411.07858: verbose compensation appears across evaluated
  models and relates to uncertainty; it does not simply disappear with capability.
- Lin, Hilton and Evans, *TruthfulQA* (ACL 2022),
  https://aclanthology.org/2022.acl-long.229/: original experiments found larger
  models less truthful on misconception-eliciting questions. This is a specific
  counterexample to size alone fixing hallucination, not evidence that larger
  modern models generally hallucinate more.

Our study has no model-size intervention and cannot establish that 4B size
causes repetition. Small-data distillation, prompt/injection distribution and
free-running decoding are alternative explanations. Likewise frozen-AR FVE is
not a factuality/hallucination measurement.

Design assistance used the experimental-design skill in Scientific Agent Skills:
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
https://doi.org/10.48550/arXiv.2609.00065 (current arXiv record checked, v2).
