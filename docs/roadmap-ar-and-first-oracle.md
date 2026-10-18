# Full-depth AR release + first L26 oracle

## Deliverables (independent)

1. A complete 32-layer Qwen3.5-4B AR collection, preserving every layer's metrics including negative or near-zero FVE.
2. A usable L26 activation-to-phrase oracle pilot, evaluated against the frozen L26 AR and simple controls.
3. Reproducible code, model cards, provenance and limitations for both.

## Existing completed inputs

- Shared full-depth corpus: /workspace/o-lens-qwen35-4b/data/olens_200k_all32, 201460 pairs per layer, 90 complete manifests, 2880 layer-shard files.
- Reusable training scripts: legacy/harvest_ar.py and legacy/train_ar.py.
- Completed ARs: layers 2, 5, 8, 11, 14, 17, 20, 23, 26, 28, 30 (historical), 31.
- L26: 1424 steps, held-out FVE 13.513%, MSE 0.877809, cosine 0.3504.
- Full-depth trained sweep: /workspace/o-lens-qwen35-4b/ar/sweep10_all32.
- L26: /workspace/o-lens-qwen35-4b/ar/l26_200k_all32.

## AR track: finish remaining 20 layers

- Pending layers: 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 29.
- Sequential single-A100 jobs. Preserve original two-epoch, microbatch-32 / accumulation-8 / eval-every-25 / LoRA-r16 recipe and train-only whitening.
- Unique output directories with resumable checkpoints; keep each layer's run config, environment, whitener statistics, final metrics, and checkpoint hash manifest.
- Never automatically retry on an unexplained scientific/numerical error. No performance-based layer exclusion.
- Final release must label AR checkpoints as reconstructors, not fully trained oracles. Present evaluation metrics and cross-layer caveats (truncation depth differs).

## Oracle track: L26 first

- Recover/reuse a public implementation as a reference. Source candidates:
  - https://huggingface.co/andyx10/oracle-lens-qwen3-4b (code/training, artifacts/provenance)
  - https://transformer-circuits.pub/2026/workspace/index.html (Appendix A.9.2)
- Check applicability to Qwen3.5-4B and the saved L26 AR geometry, residual hook and whitener before adapting anything.
- Begin with a minimal activation-conditioned inverter SFT on the existing aligned L26 span/activation pairs.
- Add deterministic inference smoke tests: activation injection, prompt formatting, saved adapter load, decoding, and no train/test leakage.
- Evaluate held-out descriptions via frozen AR reconstruction scoring and controls (random activation, shuffled activation-to-target, generic phrases), with clearly separated train/val/test.
- Add multi-concept distillation and RL only after the inverter can actually learn and produce interpretable phrases.
- Do not conflate AR FVE with oracle explanation fidelity. Report potential reward gaming and template-degeneracy failure modes.

## Stop/go gates

AR production can proceed while oracle software is developed. The L26 oracle pilot can proceed without waiting for remaining ARs. Publish a meaningful partial oracle only if the code, weights and evaluation contract are sufficiently reproducible; publish all ARs regardless of FVE, with transparent metrics.

## Provenance constraints

Never modify existing completed checkpoint directories. Persistent data/checkpoints on RunPod volume; package and HF caches on ephemeral pod disk. Explicitly version and hash datasets, checkpoint families, model revisions, environment and prompt/injection conventions. Support BlueDot Impact grant acknowledgment in released model cards.
