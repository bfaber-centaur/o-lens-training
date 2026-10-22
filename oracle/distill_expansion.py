"""Hash-bound expansion lineage; fixed validation and rollout-level exclusions."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def bank(path, expected=None):
    path = Path(path)
    meta = json.loads((path / 'meta.json').read_text())
    actual = digest(path / 'items.safetensors')
    if not (path / 'COMPLETE').is_file() or actual != meta['items_sha256'] or (expected and actual != expected):
        raise ValueError('Incomplete or changed item bank')
    return meta, load_file(str(path / 'items.safetensors'))


def groups(data, split):
    return set(zip(data[split + '_shards'].tolist(), data[split + '_rollouts'].tolist()))


def verify_expansion(path, expected, base_sha):
    meta, data = bank(path, expected)
    lineage = meta.get('expansion', {})
    if lineage.get('base_sha256') != base_sha:
        raise ValueError('Expansion does not descend from fixed validation bank')
    oldmeta, old = bank(lineage['base_items'], base_sha)
    for key, value in old.items():
        if not torch.equal(data[key][:len(value)], value):
            raise ValueError('Original item prefix or fixed validation changed: ' + key)
    for split in ('train', 'val', 'test'):
        if len(groups(data, split)) != len(data[split + '_vec']):
            raise ValueError('Repeated rollout within split')
    if any(groups(data, a) & groups(data, b) for a, b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Expansion rollout leakage')
    if len(data['val_vec']) != len(old['val_vec']):
        raise ValueError('Fixed validation size changed')
    return meta, data, old


def verify_targets(meta, data, raw, split):
    idx = data['src_item_idx'].long()
    if len(set(idx.tolist())) != len(idx) or (idx < 0).any() or (idx >= len(raw[split + '_vec'])).any():
        raise ValueError('Invalid target source indices')
    if not torch.equal(data['src_pair_row'], raw[split + '_pair_rows'][idx]) or not torch.equal(data['vec'], raw[split + '_vec'][idx]):
        raise ValueError('Target source mapping changed')
