# L26 oracle activation dependence gate

After completing docs/l26-oracle-pilot.md, evaluate the frozen adapter against
the original **exact** oracle validation rows. This gate compares teacher-forced
cross-entropy for correct activations, a deterministic derangement of validation
activations, and zero activations. It also samples 3 greedy continuations in
each condition.

```bash
cd /root/o-lens-training
source .venv/bin/activate
python -u oracle/eval_gt_sft.py \
  --checkpoint /workspace/o-lens-qwen35-4b/oracle/l26-gt-sft-pilot-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/eval-l26-gt-sft-v1 \
  --sample-count 3 \
  2>&1 | tee /root/eval-l26-gt-sft-v1.log
```

The evaluator requires a COMPLETE marker, loads saved prompt/source/transform/
scale/revision from checkpoint metadata, reconstructs the exact validation
indices and checks hashes, refuses output overwrites, and saves report.json.
Older pilot metadata did not include max_train and max_val; its original
defaults (2048/256) are used, and hashes must match or evaluation aborts.
No original checkpoints are modified.

Do not conclude faithfulness from better correct than shuffled CE alone. This
is a **pilot validation** check, not a truly untouched test. If scores are
indistinguishable, the oracle may ignore the activation. A correct-activation
advantage supports dependence but does not establish semantic interpretation.
Generation is greedy and short; inspect it for repetition, common-phrase
degeneracy, and prompt-template imitation. Do not launch RL until this gate
and saved adapter loading succeed.
