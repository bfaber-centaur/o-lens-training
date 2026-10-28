#!/usr/bin/env python3
"""Paired decoding pilot on frozen L26 workout items; generate, then summarize FVE."""
import argparse
import copy
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'oracle'))
from lens_workout import injected_embeddings, trim_eos, bullet_stats
from train_distill_student import render_concepts_prompt
from score_generated_readouts import (atomic_json, read_json, require, sha256,
                                      hash_tree, validate_inputs, verify_texts)

PROFILES = {
    'greedy160': {'max_new_tokens':160},
    'greedy64': {'max_new_tokens':64},
    'rep110': {'max_new_tokens':160,'repetition_penalty':1.1},
    **{f'sample070_s{s}':{'max_new_tokens':160,'do_sample':True,
                         'temperature':.7,'top_p':.9,'top_k':0,'seed':s} for s in (11,22,33)},
    'stop4': {'max_new_tokens':160,'stop_four':True},
}
CONDITIONS = ('correct','shuffled')


def fifth_marker(text):
    # Match the scorer's stripped '- ' / bare '-' markers. Stop as the fifth
    # marker appears; four completed bullet bodies are retained without edits.
    return len(re.findall(r'^\s*-(?:[ \t]|$)',text,re.MULTILINE)) >= 5


class StopFour(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self,input_ids,scores,**kwargs):
        return torch.tensor([fifth_marker(self.tokenizer.decode(ids.tolist(),skip_special_tokens=True))
                             for ids in input_ids],dtype=torch.bool,device=input_ids.device)


def stable_seed(item, profile):
    # Same seed for correct/donor condition, independent of scheduling/resume.
    key = f"decoding-v1:{item}:{profile}:{PROFILES[profile].get('seed',0)}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big')


def schedule(item):
    jobs = [(name,condition) for name in PROFILES for condition in CONDITIONS]
    random.Random(f'decoding-order-v1:{item}').shuffle(jobs)
    return jobs


@torch.inference_mode()
def decode(model,tok,prompt,slot,vector,alpha,profile,seed):
    cfg = dict(PROFILES[profile])
    stop = cfg.pop('stop_four',False)
    cfg.pop('seed',None)
    cap = cfg['max_new_tokens']
    cfg.setdefault('do_sample',False)
    # Explicitly avoid inherited beam search/repetition constraints.
    cfg.update(num_beams=1,num_return_sequences=1,no_repeat_ngram_size=0,
               repetition_penalty=cfg.get('repetition_penalty',1.0))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pe = injected_embeddings(model,prompt,slot,vector,alpha)
    kw = {'stopping_criteria':StoppingCriteriaList([StopFour(tok)])} if stop else {}
    seq = model.generate(inputs_embeds=pe,
        attention_mask=torch.ones(pe.shape[:2],dtype=torch.long,device=pe.device),
        eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True,**cfg,**kw)
    require(seq.ndim==2 and seq.shape[0]==1 and 1<=seq.shape[1]<=cap,
            'Unexpected generated sequence shape/prompt inclusion')
    ids,eos = trim_eos(seq[0].tolist(),tok.eos_token_id)
    text = tok.decode(ids,skip_special_tokens=True)
    return {'ids':ids,'text':text,'tokens':len(ids),'eos':eos,
            'hit_cap':seq.shape[1]==cap,'structure':bullet_stats(text),
            'generated_tokens_including_eos':seq.shape[1],
            'stop_reason':'eos' if eos else ('fifth_marker' if stop and fifth_marker(text) else 'token_cap'),
            'seed':seed,'profile':profile}


def row_hash(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False).encode()).hexdigest()


def load_row(path,contract,original,arms):
    r = read_json(path)
    checksum = r.pop('comparison_row_sha256',None)
    require(checksum==row_hash(r),'Comparison row checksum mismatch')
    require(r.get('comparison_contract')==contract,'Comparison row contract mismatch')
    require(all(r.get(k)==v for k,v in original.items() if k!='outputs'),
            'Comparison row source mismatch')
    require(set(r['outputs'])==set(arms),'Comparison row arms mismatch')
    require(all(r['outputs'][k]==v for k,v in original['outputs'].items()),
            'Original greedy baseline changed')
    return r


def generate(a):
    require(a.max_new_items is None or a.max_new_items>0,'Invalid item limit')
    m,vm,bank,sel,cfg,originals,files = validate_inputs(a.workout,a.ar_checkpoint)
    require('comparison' not in m,'Use the original workout, not a previous comparison')
    out = a.output.resolve()
    for source in (a.workout,a.ar_checkpoint,*[Path(m[k]) for k in ('selection','val','parent','student')]):
        source = source.resolve()
        require(out!=source and source not in out.parents and out not in source.parents,'Output overlaps source')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok_path = Path(m['parent'])/'tokenizer'
    tok = AutoTokenizer.from_pretrained(tok_path)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    require(tok.eos_token_id is not None,'Tokenizer lacks EOS')
    verify_texts(originals,bank,tok)
    prompt,slot,_ = render_concepts_prompt(tok,26)
    require(sha256_bytes(prompt.numpy().tobytes())==m['prompt_sha256'] and slot==m['prompt_slot']
            and m['alpha']==8000,'Frozen prompt/injection mismatch')
    import importlib.metadata
    manifest = copy.deepcopy(m)
    manifest['arms'] += [f'decode_{name}_{condition}' for name in PROFILES for condition in CONDITIONS]
    manifest['comparison'] = {'schema':'l26_decoding_v1','profiles':PROFILES,
        'original_workout':str(a.workout.resolve()),'source_sha256':files,
        'tokenizer_sha256':hash_tree(tok_path),
        'student_lora_sha256':hash_tree(Path(m['student'])/'lora'),
        'source_code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'oracle').glob('*.py'))},
        'packages':{k:importlib.metadata.version(k) for k in ('torch','transformers','peft')},
        'python':sys.version,'jobs':{str(r['item_idx']):schedule(r['item_idx']) for r in originals},
        'unit':'activation item; seeds are nested repeats, not additional items'}
    # Normalize tuple schedules to JSON-compatible lists before resume equality.
    manifest = json.loads(json.dumps(manifest))
    contract = row_hash(manifest)
    if a.dry_run:
        print(json.dumps({'items':len(originals),'new_generations':len(originals)*len(PROFILES)*len(CONDITIONS),
                          'contract':contract,'profiles':PROFILES},indent=2))
        return
    existed = out.exists()
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'.generation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'manifest.json').exists():
            require(read_json(out/'manifest.json')==manifest,'Changed comparison manifest; choose fresh output')
        else:
            require(not existed,'Existing output lacks manifest')
            atomic_json(out/'manifest.json',manifest)
        pending = []
        for r in originals:
            path = out/f"item_{r['item_idx']:06d}.json"
            if path.exists(): load_row(path,contract,r,manifest['arms'])
            else: pending.append(r)
        require(not (out/'COMPLETE').exists() or not pending,'COMPLETE has missing rows')
        jobs = pending[:a.max_new_items] if a.max_new_items else pending
        if jobs:
            require(torch.cuda.is_available(),'CUDA required')
            pm = read_json(Path(m['parent'])/'meta.json')
            base = AutoModelForCausalLM.from_pretrained(m['model'],revision=pm.get('revision'),dtype=torch.bfloat16).cuda()
            model = PeftModel.from_pretrained(base,Path(m['student'])/'lora',is_trainable=False).eval()
            model.requires_grad_(False)
            runtime = {'base_commit':getattr(base.config,'_commit_hash',None),
                       'generation_config':model.generation_config.to_dict(),
                       'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}
            runtime = json.loads(json.dumps(runtime))
            if (out/'runtime.json').exists(): require(read_json(out/'runtime.json')==runtime,'Runtime changed')
            else: atomic_json(out/'runtime.json',runtime)
            prompt = prompt.cuda()
            loc = {int(i):j for j,i in enumerate(bank['src_item_idx'].tolist())}
            for original in jobs:
                r = copy.deepcopy(original)
                for profile,condition in schedule(r['item_idx']):
                    idx = r['item_idx'] if condition=='correct' else r['shuffled_from_item_idx']
                    arm = f'decode_{profile}_{condition}'
                    r['outputs'][arm] = decode(model,tok,prompt,slot,bank['vec'][loc[idx]].cuda(),
                                                m['alpha'],profile,stable_seed(r['item_idx'],profile))
                    print(r['item_idx'],arm,r['outputs'][arm]['tokens'],r['outputs'][arm]['stop_reason'],flush=True)
                r['comparison_contract'] = contract
                r['comparison_row_sha256'] = row_hash(r)
                atomic_json(out/f"item_{r['item_idx']:06d}.json",r)
        complete = all((out/f"item_{r['item_idx']:06d}.json").exists() for r in originals)
        if complete: (out/'COMPLETE').write_text(contract+'\n')
        print('COMPLETE' if complete else 'PARTIAL',out,flush=True)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def summary_rows(scored, generated):
    """Average seeds within each item first; never use best-of-three selection."""
    def mean(xs): return sum(xs)/len(xs)
    groups = {k:[k] for k in ('greedy160','greedy64','rep110','stop4')}
    groups['sample070'] = [f'sample070_s{s}' for s in (11,22,33)]
    output = []
    for stratum in ('all','low','mid','high'):
        rows = [r for r in scored if stratum=='all' or r['stratum']==stratum]
        if not rows: continue
        for profile,names in groups.items():
            for condition in CONDITIONS:
                per_item = []
                for r in rows:
                    arms = [f'decode_{name}_{condition}' for name in names]
                    scores = [r['scores'][arm] for arm in arms]
                    gens = [generated[r['item_idx']]['outputs'][arm] for arm in arms]
                    ref = r['scores'][f'decode_greedy160_{condition}']['fve']
                    per_item.append({'item_idx':r['item_idx'],
                        'fve':mean([s['fve'] for s in scores]),
                        'delta_greedy':mean([s['fve'] for s in scores])-ref,
                        'correct_minus_shuffled':mean([r['scores'][f'decode_{name}_correct']['fve']-
                                                      r['scores'][f'decode_{name}_shuffled']['fve'] for name in names]),
                        'tokens':mean([g['generated_tokens_including_eos'] for g in gens]),
                        'valid_fraction':mean([float(s['valid']) for s in scores]),
                        'unique_bullets':mean([s['parser']['unique_text_count'] for s in scores]),
                        'scored_bullets':mean([s['parser']['scored_count'] for s in scores]),
                        'later_bullet_gain':mean([s['fve']-s['first_bullet_fve'] for s in scores]),
                        'cap_fraction':mean([float(g['hit_cap']) for g in gens])})
                output.append({'stratum':stratum,'profile':profile,'condition':condition,
                               'n_items':len(rows),'samples_per_item':len(names),
                               **{k:mean([v[k] for v in per_item]) for k in per_item[0] if k!='item_idx'},
                               'per_item':per_item})
    return output


def summarize(a):
    report = read_json(a.scores/'report.json')
    m = read_json(a.workout/'manifest.json')
    require(report['complete'] and (a.workout/'COMPLETE').is_file(),'Incomplete comparison/scoring')
    sm = read_json(a.scores/'manifest.json')
    require(Path(sm['workout']).resolve()==a.workout.resolve() and
            sm['source_files'][str((a.workout/'manifest.json').resolve())]==sha256(a.workout/'manifest.json'),
            'Score/workout lineage mismatch')
    generated = {}
    for r in report['rows']:
        path = a.workout/f"item_{r['item_idx']:06d}.json"
        require(sm['source_files'][str(path.resolve())]==sha256(path),'Generation changed after scoring')
        generated[r['item_idx']] = read_json(path)
    require(len(generated)==len(m['selected']),'Missing scored items')
    rows = summary_rows(report['rows'],generated)
    agreement = {condition:sum(g['outputs'][f'decode_greedy160_{condition}']['ids']==
                               g['outputs']['student_shared' if condition=='correct' else 'student_shuffled']['ids']
                               for g in generated.values()) for condition in CONDITIONS}
    payload = {'schema':'l26_decoding_summary_v1','score_report_sha256':sha256(a.scores/'report.json'),
               'greedy_original_exact_id_matches':agreement,'rows':rows}
    atomic_json(a.scores/'decoding_summary.json',payload)
    lines = ['# Paired decoding pilot','',f'Original/current greedy exact token matches: {agreement}.','',
             '12 selected validation items; seeds averaged within item, never best-of-three. '
             'Differences use contemporaneous greedy. Token counts include EOS and stopping-marker overhead.','',
             '| Stratum | Profile | Injection | n | FVE | Δ greedy | Tokens | Valid fraction | Later-bullet gain | Cap fraction |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['stratum']} | {r['profile']} | {r['condition']} | {r['n_items']} | {r['fve']:.4f} | "
                     f"{r['delta_greedy']:+.4f} | {r['tokens']:.1f} | {r['valid_fraction']:.3f} | "
                     f"{r['later_bullet_gain']:.4f} | {r['cap_fraction']:.3f} |")
    (a.scores/'decoding_summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command',required=True)
    g = sub.add_parser('generate')
    g.add_argument('--workout',type=Path,required=True)
    g.add_argument('--ar-checkpoint',type=Path,required=True)
    g.add_argument('--output',type=Path,required=True)
    g.add_argument('--max-new-items',type=int)
    g.add_argument('--dry-run',action='store_true')
    s = sub.add_parser('summarize')
    s.add_argument('--workout',type=Path,required=True)
    s.add_argument('--scores',type=Path,required=True)
    a = p.parse_args()
    (generate if a.command=='generate' else summarize)(a)


if __name__=='__main__': main()
