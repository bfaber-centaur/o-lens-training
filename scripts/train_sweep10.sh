#!/usr/bin/env bash
# Sequential 10-layer AR training sweep. Safe restart; no destructive overwrites.
set -euo pipefail

ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA="${AR_DATA:-/workspace/o-lens-qwen35-4b/data/olens_200k_all32}"
OUT="${AR_OUTPUT:-/workspace/o-lens-qwen35-4b/ar/sweep10_all32}"
LAYERS=(2 5 8 11 14 17 20 23 28 31)

if [[ ! -f "$ROOT/legacy/train_ar.py" ]]; then echo "Missing trainer in $ROOT" >&2; exit 1; fi
if [[ ! -f "$DATA/progress.json" ]]; then echo "Missing harvest at $DATA" >&2; exit 1; fi
python - "$DATA/progress.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["pairs"]>=200000 and d["complete_shards"]>=90, d
print("Harvest preflight:", d["pairs"], "pairs;", d["complete_shards"], "shards")
PY
mkdir -p "$OUT"
echo "Starting 10-layer sweep at $(date -Is)"
for layer in "${LAYERS[@]}"; do
  dest="$OUT/l${layer}"
  if [[ -f "$dest/metrics.json" ]]; then
    echo "SKIP L${layer}: completed metrics exist at $dest/metrics.json"
    continue
  fi
  if [[ ! -d "$DATA/pairs" ]] || ! compgen -G "$DATA/pairs/pairs_l${layer}_*.safetensors" >/dev/null; then
    echo "Missing L${layer} pair files" >&2; exit 1
  fi
  resume=()
  if [[ -e "$dest/run_config.json" ]]; then
    if [[ -f "$dest/latest.json" ]]; then
      resume=(--resume auto)
      echo "RESUME L${layer}"
    else
      echo "L${layer} has a partial run but no completed checkpoint: $dest; manual inspection needed" >&2
      exit 1
    fi
  fi
  echo "START L${layer} at $(date -Is)"
  python -u "$ROOT/legacy/train_ar.py" \
    --model Qwen/Qwen3.5-4B \
    --layer "$layer" \
    --pair-dir "$DATA/pairs" \
    --rollout-dir "$DATA/rollouts" \
    --output "$dest" \
    --epochs 2 --micro-batch 32 --grad-accum 8 \
    --eval-every 25 --lr 0.0001 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --keep-checkpoints 2 \
    --data-note "Full-depth 201460-pair V3 corpus; sequential ten-layer sweep" \
    "${resume[@]}" 2>&1 | tee -a "$OUT/l${layer}.log"
  if [[ ! -s "$dest/metrics.json" ]]; then
    echo "L${layer}: trainer exited without final metrics; stopping" >&2; exit 1
  fi
  echo "DONE L${layer} at $(date -Is)"
done
echo "SWEEP COMPLETE at $(date -Is)"
