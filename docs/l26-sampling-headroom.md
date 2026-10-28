# L26 sampling headroom (bounded diagnostic)

Decision: characterize existing control and expanded step-1000 students before L30.
Keep the twelve selected validation items; this is not independent generalization evidence.
Do not train, regenerate targets, or tune the decoding configuration on these results.

Per student: eight samples per item, 64-token cap, temperature 0.7, top-p 0.9,
top-k disabled, no repetition penalty. Seed labels 11 through 88 in increments of
11 are hashed with item/profile IDs by the existing generator. No new shuffled
generations: saved shuffled controls and saved greedy are retained and rescored.
Total 192 new generations, at most 12,288 generated tokens.

The wrapper uses the existing hash-bound, per-item resumable generator and scorer.
Complementarity uses joint FVE minus the strongest individual scored bullet, not
minus the first bullet. All invalid zero scores count in averages. Report per-item
mean and best-of-eight, then average over items. Best-of-eight excludes greedy and
is selection using the same scorer, not semantic truth or deployment performance.
Reports include every item's greedy, fixed seed-11 sample, and scorer-selected best.

Run from the pod checkout:
```bash
set -o pipefail
mkdir -p /workspace/o-lens-qwen35-4b/logs
bash scripts/run_l26_headroom.sh 2>&1 | tee -a /workspace/o-lens-qwen35-4b/logs/l26-headroom.log
```

Rerun the same command after interruption; complete items are reused. Do not change
code, packages or source artifacts mid-run. Outputs are
`oracle/l26-headroom-{control,expanded}-fve-v1/headroom_summary.{md,json}`
under the persistent project root.

Local validation: stdlib tests cover item averaging, invalid inclusion, best-of-eight
excluding greedy, deterministic ties and empty input. GPU execution remains a pod check.
