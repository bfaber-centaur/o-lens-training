#!/usr/bin/env python3
"""Frozen-AR joint FVE for saved L26 workouts. Never generates or trains."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'oracle'))
from select_distill_nnomp import nnls, embed_candidates, sha256

SCHEMA = 'l26_generated_readout_fve_v1'
PARSER = 'dash_multiline_min2tokens_first4_v1'


def parse_bullets(text, tokenize):
    """Only '- ' at a stripped line start; attach continuation, ignore preamble.

    Preserve internal newlines and duplicates. Drop empty/<2 AR-text-token
    concepts, then take the first four. No prose fallback or scaffold rewriting.
    """
    raw, current = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('- ' ) or line == '-':
            if current is not None:
                raw.append('\n'.join(current).strip())
            current = [line[1:].lstrip()]
        elif current is not None:
            current.append(line)
    if current is not None:
        raw.append('\n'.join(current).strip())
    usable = [b for b in raw if b and len(tokenize(b)) >= 2]
    bullets = usable[:4]
    return {'bullets': bullets, 'parsed_count': len(raw),
            'eligible_count': len(usable), 'scored_count': len(bullets),
            'dropped_short': len(raw)-len(usable), 'beyond_cap': max(0,len(usable)-4),
            'unique_text_count': len(set(bullets)),
            'status': 'ok' if bullets else 'no_valid_bullets'}


def geometry(raw, true, mean, whitener):
    """Original AR centering/whitening; only concept directions are normalized."""
    if (true.ndim != 1 or mean.shape != true.shape or
        whitener.shape != (len(true),len(true)) or raw.ndim != 2 or
        raw.shape[1] != len(true)):
        raise ValueError('AR geometry shape mismatch')
    if not all(torch.isfinite(x).all() for x in (raw,true,mean,whitener)):
        raise ValueError('Nonfinite AR geometry')
    target = (true.float()-mean.float()) @ whitener.float()
    if target.square().sum() < 1e-12:
        raise ValueError('Zero whitened target')
    atoms = (raw.float()-mean.float()) @ whitener.float()
    valid = atoms.norm(dim=1) > 1e-9
    return F.normalize(atoms,dim=1,eps=1e-9), target, valid


def score_vectors(raw, true, mean, whitener):
    dirs, target, valid = geometry(raw,true,mean,whitener)
    # Keep zero atoms in their original positions; they cannot reconstruct anything.
    dirs = dirs * valid[:,None]
    prefix, coeffs = [], []
    for n in range(1,len(dirs)+1):
        w, fve = nnls(dirs[:n],target)
        prefix.append(fve)
        coeffs = w.tolist()
    fve = prefix[-1] if prefix else 0.0
    active = dirs[valid]
    cos = active @ active.T
    pairs = cos[torch.triu(torch.ones_like(cos,dtype=torch.bool),diagonal=1)]
    return {'fve': fve, 'first_bullet_fve': prefix[0] if prefix else 0.0,
            'prefix_fve': prefix, 'incremental_fve': [b-a for a,b in zip([0.]+prefix,prefix)],
            'coefficients': coeffs, 'valid_direction_count': int(valid.sum()),
            'valid': bool(valid.any()),
            'status': 'ok' if valid.any() else 'no_valid_directions',
            'direction_rank': int(torch.linalg.matrix_rank(active,atol=1e-5,rtol=0)) if len(active) else 0,
            'mean_pairwise_cosine': float(pairs.mean()) if pairs.numel() else None,
            'near_duplicate_pairs_cos_ge_0999': int((pairs>=.999).sum()),
            'target_energy': float(target.square().sum()),
            'residual_energy': (1-fve)*float(target.square().sum())}


def atomic_json(path, obj):
    """Unique staging file + atomic rename, so an interrupted temp never blocks resume."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def hash_tree(path):
    files = sorted(p for p in path.rglob('*') if p.is_file())
    require(bool(files),f'Empty artifact directory: {path}')
    return {str(p.relative_to(path)):sha256(p) for p in files}


def validate_inputs(workout, checkpoint):
    from lens_workout import check_lineage, select_rows, shuffle_plan
    require((workout/'COMPLETE').is_file(),'Workout incomplete')
    m = read_json(workout/'manifest.json')
    require(m.get('schema')=='l26_lens_workout_v1' and m.get('layer')==26
            and m.get('source')=='true_GT_residual','Workout contract mismatch')
    selection, val, parent, student = [Path(m[k]) for k in ('selection','val','parent','student')]
    vm, bank, sel, _, _, psha, ssha = check_lineage(selection,val,parent,student)
    require(m['selection_sha256']==sha256(selection/'report.json')
            and m['val_targets_sha256']==vm['targets_sha256']
            and m['parent_adapter_sha256']==psha and m['student_adapter_sha256']==ssha
            and m['student_run_meta_sha256']==sha256(student.parent.parent/'meta.json'),
            'Workout source SHA mismatch')
    require(Path(sel['ar_checkpoint']).resolve()==checkpoint.resolve()
            and Path(vm['ar_checkpoint']).resolve()==checkpoint.resolve(), 'Frozen AR path mismatch')
    require((checkpoint/'COMPLETE').is_file(),'AR checkpoint incomplete')
    cfg = read_json(checkpoint/'run_config.json')
    require(cfg['model_id']==m['model'] and cfg['target_layer']==26
            and cfg['d_model']==bank['vec'].shape[1], 'AR model/layer/width mismatch')
    white = checkpoint/'whitener_l26.safetensors'
    require(sha256(white)==cfg['whitener_sha256'],'Original AR whitener SHA mismatch')
    # Old selection artifacts did not hash AR weights. Validate what they did pin,
    # and snapshot *all* checkpoint bytes now; never claim historical weight identity.
    selection_manifest = selection/'selection_manifest.json'
    require(selection_manifest.is_file(),'Missing frozen selection manifest')
    sm = read_json(selection_manifest)
    require(sel['selection_run_manifest_sha256']==sha256(selection_manifest)
            and sm['ar_run_config_sha256']==sha256(checkpoint/'run_config.json')
            and sm['ar_whitener_sha256']==sha256(white), 'Selection AR provenance mismatch')
    loc = {int(i):j for j,i in enumerate(bank['src_item_idx'].tolist())}
    require(len(loc)==len(bank['src_item_idx']),'Duplicate target IDs')
    choices = select_rows(sel,bank,m['per_bin'],m['seed'])
    permutation = shuffle_plan(len(choices),m['seed'])
    for j,c in enumerate(choices):
        c.update(target_row=loc[c['item_idx']],
                 src_pair_row=int(bank['src_pair_row'][loc[c['item_idx']]]),
                 shuffled_from_item_idx=choices[permutation[j]]['item_idx'])
    require(choices==m['selected'],'Workout selection/shuffle plan mismatch')
    rows, files = [], {}
    for choice in choices:
        path = workout/f"item_{choice['item_idx']:06d}.json"
        r = read_json(path)
        require(all(r.get(k)==v for k,v in choice.items()),'Workout item metadata mismatch')
        require(set(r['outputs'])==set(m['arms']) and
                {'parent_shared','student_shared','student_shuffled'}.issubset(r['outputs']),
                'Workout arm mismatch')
        for out in r['outputs'].values():
            require(isinstance(out['text'],str) and isinstance(out['ids'],list)
                    and all(type(i) is int and i>=0 for i in out['ids'])
                    and out['tokens']==len(out['ids']) and type(out['eos']) is bool
                    and type(out['hit_cap']) is bool, 'Malformed saved generation')
        rows.append(r)
        files[str(path.resolve())]=sha256(path)
    for p in (workout/'manifest.json',val/'meta.json',val/'targets.safetensors',
              selection/'report.json',selection_manifest,parent/'meta.json',
              student.parent.parent/'meta.json'):
        files[str(p.resolve())]=sha256(p)
    return m, vm, bank, sel, cfg, rows, files


def verify_texts(rows, bank, tok):
    for r in rows:
        j = r['target_row']
        a,b = bank['offsets'][j:j+2].tolist()
        require(tok.decode(bank['target_ids'][a:b].tolist(),skip_special_tokens=True)==r['target'],
                'Frozen target text/tokenizer mismatch')
        for out in r['outputs'].values():
            require(tok.decode(out['ids'],skip_special_tokens=True)==out['text'],
                    'Generated text/token IDs mismatch')


def summarize(rows):
    groups = {}
    for stratum in ('all','low','mid','high'):
        subset = [r for r in rows if stratum=='all' or r['stratum']==stratum]
        if not subset:
            continue
        arms = sorted(subset[0]['scores'])
        groups[stratum] = {}
        for arm in arms:
            values = [r['scores'][arm] for r in subset]
            fves = torch.tensor([v['fve'] for v in values],dtype=torch.float64)
            groups[stratum][arm] = {
                'n':len(values), 'valid':sum(v['valid'] for v in values),
                'mean':float(fves.mean()),'min':float(fves.min()),
                'q25':float(fves.quantile(.25)), 'median':float(fves.median()) if len(fves)==1 else float(fves.quantile(.5)),
                'q75':float(fves.quantile(.75)), 'max':float(fves.max()),
                'mean_first_bullet_fve':sum(v['first_bullet_fve'] for v in values)/len(values),
                'pooled_fve':1-sum(v['residual_energy'] for v in values)/sum(v['target_energy'] for v in values),
                'mean_scored_bullets':sum(v['parser']['scored_count'] for v in values)/len(values),
                'hit_cap':sum(v.get('hit_cap',False) for v in values)}
    paired = {}
    for name, left, right in (('student_minus_parent','student_shared','parent_shared'),
                              ('student_minus_shuffled','student_shared','student_shuffled')):
        ds = [r['scores'][left]['fve']-r['scores'][right]['fve'] for r in rows]
        paired[name] = {'n':len(ds),'mean':sum(ds)/len(ds)}
    return {'groups':groups,'paired_differences':paired}


def markdown(report):
    lines = ['# Generated-readout FVE','',
             f"Scored {len(report['rows'])}/{report['expected_items']} items; "
             f"{'complete' if report['complete'] else 'PARTIAL'}.",'',
             'Historical AR adapter/head hashes were not recorded by NNOMP; config and whitener are verified, '
             'and full checkpoint hashes are frozen at scoring time. See manifest.json and runtime.json.','',
             'Frozen AR reconstruction on selected validation items; not semantic ground truth. '
             'Zero-bullet/zero-direction failures contribute FVE 0 to all-item means and are counted invalid. '
             'Shuffled text is scored against the original target. Donor scores are separate diagnostics.','',
             '| Stratum | Arm | n / valid | Mean | Q25 | Median | Q75 | Min–max | First bullet | Cap hits |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s,arms in report['summary']['groups'].items():
        for arm,v in arms.items():
            lines.append(f"| {s} | {arm} | {v['n']} / {v['valid']} | {v['mean']:.4f} | "
                         f"{v['q25']:.4f} | {v['median']:.4f} | {v['q75']:.4f} | "
                         f"{v['min']:.4f}–{v['max']:.4f} | {v['mean_first_bullet_fve']:.4f} | {v['hit_cap']} |")
    lines += ['', '## Paired differences','']
    for name,v in report['summary']['paired_differences'].items():
        lines.append(f"- {name}: mean {v['mean']:.4f} (n={v['n']}).")
    lines += ['', '## Construction reference versus text readout','',
              'Stored NNOMP used exact candidate token prefixes before assembly cleanup. '
              '`nnomp_text` retokenizes the assembled target under the same scorer as generated text. '
              'These are separate measurements, neither independent ground truth.','',
              '| Item | Stratum | Stored construction FVE | Rescored target text FVE |',
              '|---:|---|---:|---:|']
    for r in report['rows']:
        lines.append(f"| {r['item_idx']} | {r['stratum']} | {r['construction_fve']:.4f} | {r['scores']['nnomp_text']['fve']:.4f} |")
    return '\n'.join(lines)+'\n'


def saved_row(path, fingerprint, item_idx):
    obj = read_json(path)
    require(obj.get('fingerprint')==fingerprint and obj.get('item_idx')==item_idx,
            'Saved score belongs to different run/item')
    payload = obj['result']
    digest = hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(obj.get('result_sha256')==digest, 'Saved score checksum mismatch')
    return payload


def write_row(path, fingerprint, result):
    digest = hashlib.sha256(json.dumps(result,sort_keys=True,allow_nan=False).encode()).hexdigest()
    atomic_json(path, {'fingerprint':fingerprint,'item_idx':result['item_idx'],
                       'result_sha256':digest,'result':result})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workout',type=Path,required=True)
    p.add_argument('--ar-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--max-new-items',type=int)
    p.add_argument('--dry-run',action='store_true')
    a = p.parse_args()
    require(a.batch_size>0 and (a.max_new_items is None or a.max_new_items>0),'Invalid batch/item limit')
    m,vm,bank,sel,cfg,rows,files = validate_inputs(a.workout,a.ar_checkpoint)
    # Prevent writing into or above any immutable source artifact tree.
    sources = [a.workout,a.ar_checkpoint,*[Path(m[k]) for k in ('selection','val','parent','student')]]
    out = a.output.resolve()
    for src in sources:
        src = src.resolve()
        require(out!=src and out not in src.parents and src not in out.parents,'Output overlaps source artifacts')
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    tok_path = Path(vm['teacher'])/'tokenizer'
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    require(tok.pad_token_id is not None,'Tokenizer has no padding/EOS ID')
    verify_texts(rows,bank,tok)
    white = load_file(str(a.ar_checkpoint/'whitener_l26.safetensors'))
    mean, W = white['mean'].float(),white['whitener'].float()
    for r in rows:
        geometry(torch.empty(0,len(mean)),bank['vec'][r['target_row']],mean,W)
    source_hashes = {str(path.relative_to(ROOT)):sha256(path) for path in
                     sorted(list((ROOT/'oracle').glob('*.py'))+list((ROOT/'legacy').glob('*.py')))}
    manifest = {'schema':SCHEMA,'parser':PARSER,'nnls_iterations':80,'batch_size':a.batch_size,
                'workout':str(a.workout.resolve()),'ar_checkpoint':str(a.ar_checkpoint.resolve()),
                'source_files':files,'code_sha256':source_hashes,
                'checkpoint_sha256':hash_tree(a.ar_checkpoint), 'tokenizer_sha256':hash_tree(tok_path),
                'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','peft','safetensors')},
                'python':sys.version,'historical_ar_weights_verified':False,
                'historical_ar_note':'Selection pinned config/whitener, not adapter/head bytes; weights fingerprinted at scoring time.'}
    fingerprint = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    if a.dry_run:
        print(json.dumps({'validated_items':len(rows),'fingerprint':fingerprint,'manifest':manifest},indent=2))
        return
    existed = out.exists()
    out.mkdir(parents=True,exist_ok=True)
    # Nonblocking OS lock releases on process death, including interrupted runs.
    import fcntl
    with (out/'.scoring.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'manifest.json').exists():
            require(read_json(out/'manifest.json')==manifest,
                    'Existing output manifest differs; choose a new output')
        else:
            require(not existed,'Existing output lacks manifest; choose a new output')
            atomic_json(out/'manifest.json',manifest)
        run(a,m,bank,sel,rows,mean,W,tok,cfg,out,fingerprint)


def run(a,m,bank,sel,rows,mean,W,tok,cfg,out,fingerprint):
    saved, pending = {}, []
    for r in rows:
        path = out/f"item_{r['item_idx']:06d}.json"
        if path.exists():
            saved[r['item_idx']] = saved_row(path,fingerprint,r['item_idx'])
        else:
            pending.append(r)
    require(not (out/'COMPLETE').exists() or not pending,'COMPLETE output has missing item rows')
    if a.max_new_items is not None:
        pending = pending[:a.max_new_items]
    loc = {int(i):j for j,i in enumerate(bank['src_item_idx'].tolist())}
    construction = {r['item_idx']:r['omp_fve'] for r in sel['rows']}
    if pending:
        require(torch.cuda.is_available(),'CUDA required for frozen AR inference')
        from train_ar import build_ar
        config = argparse.Namespace(model=cfg['model_id'],gpu=0,layer=26,
                                    cache_dir='/root/.cache/huggingface/hub')
        ar = build_ar(config,tok,mean,len(mean),torch.device('cuda:0'),a.ar_checkpoint).eval()
        ar.requires_grad_(False)
        base_config = ar.peft_model.get_base_model().config
        runtime = {'base_model_resolved_commit':getattr(base_config,'_commit_hash',None),
                   'cuda_runtime':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name(0)}
        if (out/'runtime.json').exists():
            require(read_json(out/'runtime.json')==runtime,'AR runtime/base revision changed on resume')
        else:
            atomic_json(out/'runtime.json',runtime)
        tag = torch.tensor(tok('[Layer 26]',add_special_tokens=False).input_ids,dtype=torch.long)
        for r in pending:
            texts = {'nnomp_text':r['target'],**{k:v['text'] for k,v in r['outputs'].items()}}
            parsed = {k:parse_bullets(v,lambda text: tok(text,add_special_tokens=False).input_ids)
                      for k,v in texts.items()}
            phrases = list(dict.fromkeys(b for v in parsed.values() for b in v['bullets']))
            token_rows = [{'ids':tok(b,add_special_tokens=False).input_ids} for b in phrases]
            require(all(v['ids'] for v in token_rows),'Nonempty bullet tokenizes to empty input')
            # No EOS, chat wrapping, silent truncation, or 8000/unit injection scaling.
            with torch.inference_mode():
                vectors = embed_candidates(ar,token_rows,tag,tok.pad_token_id,a.batch_size) if phrases else torch.empty(0,len(mean))
            lookup = dict(zip(phrases,vectors))
            true = bank['vec'][r['target_row']]
            scores = {}
            for arm,parsed_arm in parsed.items():
                raw = torch.stack([lookup[b] for b in parsed_arm['bullets']]) if parsed_arm['bullets'] else torch.empty(0,len(mean))
                scores[arm] = {**score_vectors(raw,true,mean,W),'parser':parsed_arm}
                if arm in r['outputs']:
                    scores[arm].update({k:r['outputs'][arm][k] for k in ('eos','hit_cap','tokens')})
                if arm=='student_shuffled':
                    donor = bank['vec'][loc[r['shuffled_from_item_idx']]]
                    scores[arm]['donor_diagnostic'] = score_vectors(raw,donor,mean,W)
            result = {'item_idx':r['item_idx'],'stratum':r['stratum'],
                      'src_pair_row':r['src_pair_row'],'shuffled_from_item_idx':r['shuffled_from_item_idx'],
                      'construction_fve':construction[r['item_idx']],'scores':scores}
            write_row(out/f"item_{r['item_idx']:06d}.json",fingerprint,result)
            saved[r['item_idx']] = result
            print('SCORED',r['item_idx'],{k:round(v['fve'],4) for k,v in scores.items()},flush=True)
    results = [saved[r['item_idx']] for r in rows if r['item_idx'] in saved]
    report = {'schema':SCHEMA,'fingerprint':fingerprint,'expected_items':len(rows),
              'complete':len(results)==len(rows),'rows':results,'summary':summarize(results)}
    atomic_json(out/'report.json',report)
    # Markdown is derived from atomic report.json and can always be regenerated.
    (out/'report.md').write_text(markdown(report))
    if report['complete']:
        (out/'COMPLETE').write_text(fingerprint+'\n')
    print('COMPLETE' if report['complete'] else 'PARTIAL',out/'report.md',flush=True)


if __name__=='__main__':
    main()
