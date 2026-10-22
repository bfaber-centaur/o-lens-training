#!/usr/bin/env python3
"""Combine original targets with fresh expansion targets, preserving source indices."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
try:
    from .distill_expansion import digest, verify_expansion, verify_targets
    from .train_distill_student import load_targets
except ImportError:
    from distill_expansion import digest, verify_expansion, verify_targets
    from train_distill_student import load_targets


def combine(original, fresh, items, output):
    if output.exists():
        raise ValueError('Refusing existing output')
    om, old = load_targets(original, 'train')
    fm, new = load_targets(fresh, 'train')
    _, expanded, base = verify_expansion(items, fm['items_sha256'], om['items_sha256'])
    for k in ('model','layer','teacher','filter_policy','min_picks','ar_checkpoint','prompt_mode'):
        if om[k] != fm[k]:
            raise ValueError('Target policy mismatch: ' + k)
    verify_targets(om, old, base, 'train')
    verify_targets(fm, new, expanded, 'train')
    if (new['src_item_idx'] < len(base['train_vec'])).any():
        raise ValueError('Fresh targets include original prefix')
    result = {k:torch.cat((old[k], new[k])) for k in ('vec','target_ids','src_pair_row','src_item_idx')}
    result['offsets'] = torch.cat((old['offsets'], new['offsets'][1:] + len(old['target_ids'])))
    verify_targets(fm, result, expanded, 'train')
    meta = dict(fm)
    meta.update(items_path=str(items.resolve()), kept_rows=len(result['vec']),
        input_rows=om['input_rows'] + fm['input_rows'],
        components=[{'path':str(p.resolve()), 'sha256':m['targets_sha256']} for p,m in ((original,om),(fresh,fm))])
    # Assembly-specific counts belong to each component, not the combined bank.
    for k in ('reject_counts','upstream_reject_counts','newly_accepted_vs_upstream','selection_sha256'):
        meta.pop(k, None)
    output.mkdir(parents=True)
    save_file(result, str(output / 'targets.safetensors'))
    meta['targets_sha256'] = digest(output / 'targets.safetensors')
    (output / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    (output / 'COMPLETE').write_text('ok\n')
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('original','fresh','items','output'):
        p.add_argument('--'+arg, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(combine(a.original, a.fresh, a.items, a.output), indent=2))


if __name__ == '__main__':
    main()
