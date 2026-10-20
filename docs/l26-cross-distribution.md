# L26 cross-distribution transfer matrix

Evaluation-only comparison. Uses both completed SFT adapters against the same
256 oracle-validation rows and token labels. The same checkpoint injection
contract (prompt, soft-token slot, unit normalization, alpha) is enforced.
The only deliberate variation is vector source:

| Trained on / Tested on | Real target activation | Frozen AR(span) |
| --- | --- | --- |
| GT-SFT 100-step extension | correct / shuffled / zero CE | correct / shuffled / zero CE |
| AO-1 20-step pilot | correct / shuffled / zero CE | correct / shuffled / zero CE |

Both models use unequal training budgets: report that prominently and avoid
ranking objectives until their budgets are matched. Data are oracle-validation
rows, not independent final test items. Teacher-forced CE is NOT explanation
faithfulness. This experiment is designed to measure transfer of activation
conditioning across the two distributions only.

From a pod with the working uv environment and mounted shared volume:

```bash
cd /root/o-lens-training
git pull --ff-only
source .venv/bin/activate
python -u oracle/eval_cross_distribution.py \
  --gt-checkpoint /workspace/o-lens-qwen35-4b/oracle/l26-gt-sft-extend100-v1 \
  --ao-checkpoint /workspace/o-lens-qwen35-4b/oracle/l26-ao1-20-v1 \
  --arout /workspace/o-lens-qwen35-4b/oracle/l26-arout-v1 \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-cross-distribution-v1 \
  2>&1 | tee /root/l26-cross-distribution-v1.log
```

Output is report.json with all twelve CE conditions, token counts and matched
correct-minus-shuffled gaps. Script refuses existing output, missing completion
markers, input cache/hash mismatch or different validation index hashes. Each
model loads and evaluates separately to avoid fitting both in GPU memory.

After checking the report, use `nvidia-smi` and ensure other work is saved,
then **stop/terminate the RunPod pod in the console to stop billing**.
Halting training or closing SSH does not stop GPU rental. Network-volume
artifacts persist independently of disposable pod storage.

Note: First GPU smoke test is pending. Do not present the metrics as final
scientific evidence without independent held-out evaluation.
