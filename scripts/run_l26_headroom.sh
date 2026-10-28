#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/root/o-lens-training/.venv/bin/python
LAB=/workspace/o-lens-qwen35-4b
AR=$LAB/ar/l26_200k_all32/checkpoints/step_001424
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export PYTHONUNBUFFERED=1
export TMPDIR=/root/tmp
mkdir -p "$TMPDIR" "$LAB/logs"
# Validate both source workouts before spending GPU time.
for arm in control expanded; do
  "$PY" oracle/l26_sampling_headroom.py generate \
    --workout "$LAB/oracle/l26-data-${arm}-1000-workout-v1" \
    --ar-checkpoint "$AR" --output "$LAB/oracle/l26-headroom-${arm}-v1" --dry-run
done
for arm in control expanded; do
  workout=$LAB/oracle/l26-headroom-${arm}-v1
  scores=$LAB/oracle/l26-headroom-${arm}-fve-v1
  "$PY" oracle/l26_sampling_headroom.py generate \
    --workout "$LAB/oracle/l26-data-${arm}-1000-workout-v1" \
    --ar-checkpoint "$AR" --output "$workout"
  "$PY" oracle/l26_sampling_headroom.py score \
    --workout "$workout" --ar-checkpoint "$AR" --output "$scores"
  "$PY" oracle/l26_sampling_headroom.py summarize --workout "$workout" --scores "$scores"
done
