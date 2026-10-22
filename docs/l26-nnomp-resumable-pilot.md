# L26 vLLM teacher → resumable NNOMP selection

The teacher sampling banks are complete on the persistent network volume:

- Train: `/workspace/o-lens-qwen35-4b/oracle/l26-vllm-train512-v1` (512 items, 64 samples/item)
- Validation: `/workspace/o-lens-qwen35-4b/oracle/l26-vllm-val64-v1` (64 items, 64 samples/item)
- True-activation bank: `/workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1`

**Run with the `.venv` Python environment**, not `/root/olens-vllm-venv`. Frozen AR inference is GPU-bound. The source, target, and whitener contracts remain unchanged.

## 32-item pilot, resumable into the complete train run

```bash
cd /root/o-lens-training
set -o pipefail

.venv/bin/python -u oracle/select_distill_nnomp.py \
  --samples /workspace/o-lens-qwen35-4b/oracle/l26-vllm-train512-v1 \
  --items /workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1 \
  --ar-checkpoint /workspace/o-lens-qwen35-4b/ar/l26_200k_all32/checkpoints/step_001424 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-vllm-nnomp-train512-v1 \
  --batch-size 32 --max-new-items 32 \
  2>&1 | tee /root/l26-nnomp-train512-pilot.log
```

The program prints `NNOMP PARTIAL 32/512` and a `STRICT FILTER AUDIT` showing assembly rejection reasons for the completed items. This does **not** write `COMPLETE`; the downstream assembly script rightly refuses incomplete selection.

When the quality audit is satisfactory, run the **same command and output** but omit `--max-new-items 32`. Already completed rows are checked and skipped; the remaining rows are computed. The partial cap is not part of the run contract, so removing it is safe.

On interruption, rerun the command with the same `--samples`, `--items`, `--ar-checkpoint`, `--output`, `--batch-size`, `--start`, `--count`, and `--min-gain` options.

## Artifacts and provenance

- `selection_manifest.json`: frozen input hashes, AR whitener and run-config hashes, code hash, and selection settings
- `selection_rows/item_NNNNNN.json`: atomic per-item checkpoints, including source teacher-sample hash
- `report.json`: diagnostic partial report, or full legacy-compatible selection report
- `COMPLETE`: written **only when all requested items have been selected or explicitly skipped**

The script refuses to reuse an output directory with a differing selection manifest. Do not delete/overwrite partial artifacts to force a restart; use a new versioned directory if the contract needs to change.

Once `COMPLETE` exists, use `oracle/assemble_distill_bullets.py` to produce the final student targets, retaining the original upstream filters.

## Important interpretation

The baseline assembly's `ngram_loop` condition treats any repeated 1-, 2-, or 3-gram (at least three occurrences) as a loop. Ordinary English function words such as `a` can trip this filter. The printed strict audit **does not modify this filter**. Record its rejection distribution before considering any separately declared counterfactual rules; never silently overwrite the baseline.
