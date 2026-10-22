"""Audit and namespace a dedicated fresh harvest before expansion selection."""
import json
from pathlib import Path
import torch
from harvest_exclusions import audit_fresh, file_digest
from harvest_ar import validate_completed_shards
from train_ar import load_corpus


def load_fresh(root, original, ns, model, layer, revision=None):
    root = Path(root)
    config = json.loads((root / 'run_config.json').read_text())['contract']
    if config['model_id'] != model or config['layers'] != [layer] or config['max_spans'] != 1:
        raise ValueError('Fresh harvest must match model/layer and use one span per rollout')
    env = json.loads((root / 'environment.json').read_text())
    if revision and env.get('model_revision') != revision:
        raise ValueError('Fresh harvest model revision differs from teacher')
    manifests, _ = validate_completed_shards(root/'pairs',root/'rollouts',root/'manifests',[layer])
    for directory, pattern, expected in (
        ('rollouts','rollouts_*.jsonl',{f'rollouts_{i:05d}.jsonl' for i in manifests}),
        ('pairs',f'pairs_l{layer}_*.safetensors',{f'pairs_l{layer}_{i:05d}.safetensors' for i in manifests})):
        if {p.name for p in (root/directory).glob(pattern)} != expected:
            raise ValueError('Uncommitted files in fresh harvest; finish/resume harvesting first')
    audit = audit_fresh(ns.rollout_dir, root/'rollouts')
    if config.get('exclude_rollout_sha256') != audit['original_rollout_sha256']:
        raise ValueError('Fresh harvest exclusions do not match the original corpus')
    fresh_ns = type(ns)(**vars(ns))
    fresh_ns.pair_dir = root/'pairs'
    fresh_ns.rollout_dir = root/'rollouts'
    corpus = load_corpus(fresh_ns)
    if corpus.d_model != original.d_model:
        raise ValueError('Fresh activation width mismatch')
    # These rollouts have never been supplied to AR training. The whole new pool
    # is eligible for distillation; its own 85/15 partition reserves test items.
    corpus.is_val = torch.ones_like(corpus.is_val)
    corpus.is_train = torch.zeros_like(corpus.is_train)
    shard_offset = int(original.shard_ids.max()) + 1
    corpus.shard_ids += shard_offset
    audit.update(schema='fresh_distill_corpus_v1', path=str(root.resolve()),
        shard_offset=shard_offset, pair_row_offset=len(original.activations),
        config_sha256=file_digest(root/'run_config.json'),
        model_revision=env.get('model_revision'),
        original_pair_sha256={str(p):file_digest(p) for p in original.pair_paths})
    return corpus, audit
