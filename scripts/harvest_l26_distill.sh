#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/root/o-lens-training/.venv/bin/python
LAB=/workspace/o-lens-qwen35-4b
PARENT=$LAB/oracle/l26-ao-sft-500-v1/checkpoints/step_000500
revision=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("revision") or "")' "$PARENT/meta.json")
extra=()
if [[ -n $revision ]]; then extra+=(--revision "$revision"); fi
case "${1:-}" in
  preflight) target=256; extra+=(--dry-run) ;;
  smoke) target=256 ;;
  full) target=8192 ;;
  *) echo 'Usage: bash scripts/harvest_l26_distill.sh {preflight|smoke|full}' >&2; exit 2 ;;
esac
# One pair per accepted rollout; fresh pool is dedicated to distillation, not AR.
"$PY" -u legacy/harvest_ar.py --model Qwen/Qwen3.5-4B --layers 26 \
  --output "$LAB/data/l26-distill-fresh8192-v1" \
  --exclude-rollouts "$LAB/data/olens_200k_all32/rollouts" \
  --target-pairs "$target" --shard-pairs 256 --max-spans 1 --span-min 1 --span-max 32 \
  --gen-batch 16 --replay-batch 1 --max-new-tokens 1536 \
  --temperature 1.0 --top-p 1.0 --seed 43 \
  --data-note 'Dedicated L26 distillation harvest; excluded old prompts; never AR training' \
  "${extra[@]}"
