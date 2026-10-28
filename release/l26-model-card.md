---
base_model: Qwen/Qwen3.5-4B
library_name: peft
tags:
- interpretability
- activation-oracle
- lora
- experimental
---

# Qwen3.5-4B O-Lenses

Experimental activation-to-text LoRA adapters for **Qwen/Qwen3.5-4B**.
L26 is available; future layers will have separate directories and their own contracts.
These are exploratory readouts, not faithful explanations of everything a model represents.

## Use

Default: `l26/expanded`. Matched small-data comparator: `l26/control`.
Both are fixed step-1000 checkpoints; neither was selected by best validation score.
Base model weights and the activation reconstructor (AR) are not bundled.

Download this repository, then:
```bash
python read_vector.py --release /path/to/download --vector activation.safetensors
```
The safetensors file must contain `activation`: a raw, finite, nonzero float vector
of shape `[2560]`. CUDA is required. The reader checks adapter weights after loading.

**L26 means the output of decoder block index 26 (zero-based), before final model
normalization**, at the token immediately preceding the continuation of interest.
The reader replaces one prompt-token embedding with `8000 * h / ||h||`.
Use the saved tokenizer, exact prompt IDs and slot in each `contract.json`.
Do not whiten the input. Default decoding: greedy, 64 new tokens.

## Method and provenance

Adapted from [Agam Bhatia's O-Lens pipeline](https://github.com/agu18dec/olens_draft).
A frozen L26 AR embeds candidate text; whitened NNOMP selects four candidates
against each true residual, producing bullet targets. Students warm-start from
our 500-step AO SFT teacher and train on true residuals, LoRA-only, for 1,000
optimizer steps (LR 5e-5; batch 1 × accumulation 4; seed 42).
Control uses 374 targets; expanded uses 2,928. Both use the same validation set.
Assembly uses `no_unigram`: repeated 2-/3-grams are filtered, unigram repetition is allowed.
This 4B lineage differs from Agam's 27B recipe, including its teacher prompt and injection scale.

Code: [bubskee/o-lens-training](https://github.com/bubskee/o-lens-training).
Per-variant provenance records source metadata and adapter/target hashes.
`SHA256SUMS` covers the exported payload. Historical NNOMP did not record AR
adapter/head hashes; later scoring froze full checkpoint hashes. No claim of
verified historical AR-weight identity is made.

## Exploratory results

Twelve repeatedly inspected, stratified validation items; **not an independent test**.
Frozen-AR joint FVE measures reconstruction, not semantic truth.

| Metric | Control | Expanded |
|---|---:|---:|
| Greedy FVE (headroom rescore) | 0.1993 | 0.1987 |
| Mean sampled FVE | 0.1971 | 0.2012 |
| AR-selected best of 8 | 0.2285 | 0.2399 |
| Shuffled greedy FVE (approximately) | 0.0002 | 0.0002 |

Sampling: T=0.7, top-p=0.9, 64-token cap; seeds averaged within item.
Best-of-eight is scorer-assisted selection, not ordinary inference performance.
See `l26/*/evaluation.json` for exported aggregate results.

Readouts often recover local continuation/topic information, but repeat,
mix languages, truncate, and invent plausible details. Higher FVE does not
guarantee a more faithful explanation. Four bullets need not express four
independent concepts. Arbitrary direction vectors and other layers are unvalidated inputs.

Compute support: BlueDot Impact.
