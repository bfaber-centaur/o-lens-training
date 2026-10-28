# L26 HF release v0.1

Target: `bubskee/qwen3.5-4b-o-lenses`. Expanded is the default exploratory
adapter; control is a matched comparator, not discarded. Keep greedy64.
L30 and subsequent layers will have their own folders/contracts. No L26
retraining is planned. Pain-axis examples are reserved for external L30 diagnostics.

Public payload is allowlisted: student adapters, saved tokenizers, exact prompt
contracts, compact evaluations, provenance, reader, environment package pins,
and checksums. No merged base, AR weights, optimizer state, raw prompts,
training texts, validation readout texts, or activations are uploaded.

After merging and pulling:
```bash
cd /root/o-lens-training
export TMPDIR=/root/tmp
mkdir -p "$TMPDIR"
.venv/bin/python scripts/export_l26_hf.py \
  --output /workspace/o-lens-qwen35-4b/releases/hf-l26-v0.1 --smoke
```

The smoke invokes the standalone packaged reader on one saved expanded validation
vector and requires an exact decoded greedy match; kernel drift may cause a
failure requiring inspection. It checks adapter weight liveness as well.
The exporter verifies source/evaluation hashes and copied file checksums.
GPU smoke must pass on the pod; it was not run by the remote author.

Publish the verified folder using an HF token with write access to bubskee
(`HF_TOKEN` or existing local authentication; do not paste a token into chat):
```bash
.venv/bin/python scripts/export_l26_hf.py \
  --output /workspace/o-lens-qwen35-4b/releases/hf-l26-v0.1 \
  --upload-only --publish
```
Publication creates a public model repo and a single payload commit.
An existing nonempty repository is refused; inspect rather than overwrite.
Upload-only rechecks every payload checksum. It does not re-run GPU inference.
On success it prints the immutable HF commit URL.
Keep the exported folder on /workspace.

Summary: greedy control/expanded FVE .1993/.1987; mean sampling .1971/.2012;
best-of-eight .2285/.2399; sample caps .7708/.6667, on twelve selected validation
items. Expanded best-of-eight wins 9/12. Sampling offers alternatives but
AR score does not certify semantic fidelity. Source readouts show meaningful
local topic/continuation recovery alongside repetition, language mixing, and
unsupported elaboration. Neither checkpoint is independently validated.
