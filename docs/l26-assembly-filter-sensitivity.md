# L26 distillation: assembly-policy sensitivity

## Why

An exploratory 32-item train NNOMP pilot of Qwen3.5-4B found:

| First rejection reason | Upstream 1/2/3-gram rule | Exploratory 2/3-gram rule |
| --- | ---: | ---: |
| Accepted | 11 | 23 |
| ngram_loop | 15 | 1 |
| bullet_collision | 1 | 3 |
| n_sel!=4 | 3 | 3 |
| min_units<2 | 2 | 2 |

The baseline's repeated-unigram hits include common words or CJK units (e.g., `the`, `a`, `的`). First-rejection reason counts are conditional on filter order: two rows formerly labeled ngram_loop were later rejected for bullet collisions. These pilot outcomes are **exploratory**; 32 items do not establish generalization.

## Policies

- `--filter-policy upstream` (default): byte-for-byte equivalent filter logic to the existing assembler and Agam's source; reject any 1-, 2-, or 3-gram recurring ≥3 times within any pick.
- `--filter-policy no_unigram` (experimental): leave the rest of the filter unchanged, but reject only 2-/3-grams recurring ≥3 times. This admits legitimate repeated function words but can also admit some genuine single-word stuttering; inspect newly admitted text manually.

No changes to candidate generation, frozen AR, target vectors, whiteners, NNOMP selections, k=4, or all other assembly rules. Each output stores its policy name, an upstream counterfactual audit, and `newly_accepted_vs_upstream`. The **output directory must be unique** per variant.

## After the NNOMP run finishes

The selection directory must contain `COMPLETE`; the 32-item partial report cannot be assembled into a student dataset.

```bash
cd /root/o-lens-training
S=/workspace/o-lens-qwen35-4b/oracle/l26-vllm-nnomp-train512-v1
I=/workspace/o-lens-qwen35-4b/oracle/l26-distill-gt-items-v1

.venv/bin/python oracle/assemble_distill_bullets.py \
  --selection "$S" --items "$I" --filter-policy upstream \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-bullets-train512-upstream-v1

.venv/bin/python oracle/assemble_distill_bullets.py \
  --selection "$S" --items "$I" --filter-policy no_unigram \
  --output /workspace/o-lens-qwen35-4b/oracle/l26-bullets-train512-no-unigram-v1
```

Repeat separately for validation once NNOMP has completed a distinct validation-split selection run. Do not tune the filter on evaluation results, or treat the resulting datasets as independently selected holdouts.

## Reporting

Report the original upstream acceptance separately from the exploratory no-unigram variant. Compare accepted-item overlap, selected-target text quality, NNOMP FVE distributions, and downstream student metrics before claiming a methodological improvement. Do not equate the ~72% exploratory pilot acceptance with upstream's ~70% on a different architecture/dataset.
