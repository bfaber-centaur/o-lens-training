#!/usr/bin/env bash
# Run from the repository on the pod. Artifacts persist; environments stay on /root.
set -euo pipefail
cd "$(dirname "$0")/.."
LAB=/workspace/o-lens-qwen35-4b
PY=/root/o-lens-training/.venv/bin/python
VPY=/root/olens-vllm-venv/bin/python
PARENT=$LAB/oracle/l26-ao-sft-500-v1/checkpoints/step_000500
AR=$LAB/ar/l26_200k_all32/checkpoints/step_001424
BASE=$LAB/oracle/l26-distill-gt-items-v1
OLD=$LAB/oracle/l26-bullets-train512-no-unigram-v1
VAL=$LAB/oracle/l26-bullets-val64-no_unigram-v1
SELECTION=$LAB/oracle/l26-vllm-nnomp-val64-v1
FRESH_DATA=$LAB/data/l26-distill-fresh8192-v1
ITEMS=$LAB/oracle/l26-distill-items4096-v1
SAMPLES=$LAB/oracle/l26-teacher-extra3584-v1
NNOMP=$LAB/oracle/l26-nnomp-extra3584-v1
FRESH=$LAB/oracle/l26-bullets-extra3584-v1
COMBINED=$LAB/oracle/l26-bullets-train4096-v1
MERGED=/root/olens-merged/l26-step500
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TMPDIR=/root/tmp
mkdir -p "$TMPDIR"
prepare=(oracle/prepare_distill_items.py --data "$LAB/data/olens_200k_all32"
  --teacher "$PARENT" --base-items "$BASE" --output "$ITEMS"
  --n-train 4096 --reserve-test 256 --seed 42 --fresh-data "$FRESH_DATA")
case "${1:-}" in
  preflight) "$PY" "${prepare[@]}" --dry-run ;;
  prepare) "$PY" "${prepare[@]}" ;;
  merge-teacher) "$PY" oracle/merge_teacher_vllm.py --teacher "$PARENT" --output "$MERGED" ;;
  sample)
    "$VPY" oracle/sample_distill_teacher_vllm.py --teacher "$PARENT" --items "$ITEMS" \
      --merged "$MERGED" --output "$SAMPLES" --split train --start 512 --count 3584 \
      --n-samples 64 --max-new-tokens 100 --chunk-items 2 --max-num-seqs 128 --seed 0 ;;
  select)
    "$PY" oracle/select_distill_nnomp.py --samples "$SAMPLES" --items "$ITEMS" \
      --ar-checkpoint "$AR" --output "$NNOMP" ;;
  assemble)
    "$PY" oracle/assemble_distill_bullets.py --selection "$NNOMP" --items "$ITEMS" \
      --output "$FRESH" --min-picks 4 --filter-policy no_unigram
    "$PY" oracle/combine_distill_targets.py --original "$OLD" --fresh "$FRESH" \
      --items "$ITEMS" --output "$COMBINED" ;;
  train-control|train-expanded|check-control|check-expanded)
    arm=${1##*-}; train=$OLD
    if [[ $arm == expanded ]]; then train=$COMBINED; fi
    extra=()
    if [[ $1 == check-* ]]; then extra+=(--dry-run); fi
    if [[ ${2:-} == --resume ]]; then extra+=(--resume); fi
    "$PY" oracle/train_distill_student.py --train "$train" --val "$VAL" \
      --init-from "$PARENT" --output "$LAB/oracle/l26-data-${arm}-1000-v1" \
      --max-steps 1000 --save-every 250 --batch-size 1 --grad-accum 4 --lr 5e-5 --seed 42 "${extra[@]}" ;;
  evaluate)
    for arm in control expanded; do
      workout=$LAB/oracle/l26-data-${arm}-1000-workout-v1
      "$PY" oracle/lens_workout.py --selection "$SELECTION" --val "$VAL" --parent "$PARENT" \
        --student "$LAB/oracle/l26-data-${arm}-1000-v1/checkpoints/step_001000" \
        --output "$workout" --per-bin 4 --seed 42 --max-new-tokens 64
      "$PY" oracle/score_generated_readouts.py --workout "$workout" --ar-checkpoint "$AR" \
        --output "$LAB/oracle/l26-data-${arm}-1000-fve-v1"
    done ;;
  *) echo 'Usage: bash scripts/run_l26_data_expansion.sh {preflight|prepare|merge-teacher|sample|select|assemble|check-control|check-expanded|train-control|train-expanded|evaluate} [--resume]' >&2; exit 2 ;;
esac
