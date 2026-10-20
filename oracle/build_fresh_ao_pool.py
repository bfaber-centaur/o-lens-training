#!/usr/bin/env python3
"""Fresh, rollout-window AO pool with Agam-style prefix exclusion.

Produces token-id crops only. Frozen AR output generation is a distinct stage.
Fails closed if rollout output_ids are unavailable: do NOT substitute pairs spans.
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

LENGTHS=(2,4,8,16,32,64)
PLACEHOLDER=re.compile(r"\b(?:NAME|EMAIL|PHONE|IP_ADDRESS|CREDIT_CARD)_\d+\b")
def fingerprint(ids):
    return hashlib.blake2b(torch.tensor(ids,dtype=torch.int32).numpy().tobytes(),digest_size=16).digest()

def create_pool(rollouts,pairs,tokenizer,seed=42,val_fraction=.1,max_per_conv=4):
    """Each 64-token window contributes at most one crop; no AR-prefix leakage."""
    rng=random.Random(seed)
    excluded={n:set() for n in LENGTHS}
    for pair in sorted(pairs):
        tensors=load_file(str(pair))
        ids=tensors["span_token_ids"].tolist()
        offsets=tensors["span_offsets"].tolist()
        for lo,hi in zip(offsets[:-1],offsets[1:]):
            for n in LENGTHS:
                if hi-lo>=n:
                    excluded[n].add(fingerprint(ids[lo:lo+n]))
    seen_window=set()
    seen={n:set() for n in LENGTHS}
    records={"train":[],"val":[]}
    counts={"rollouts":0,"windows":0,"duplicate_window":0,"duplicate_crop":0,
            "ar_prefix_excluded":0,"special":0,"placeholder":0}
    special=set(tokenizer.all_special_ids)
    for path in sorted(rollouts):
        with path.open(encoding="utf-8") as f:
            for line in f:
                row=json.loads(line)
                ids=row.get("assistant_ids",row.get("output_ids"))
                if not isinstance(ids,list) or not all(isinstance(t,int) for t in ids):
                    raise ValueError(f"{path}: missing assistant_ids/output_ids token list; cannot construct fresh AO windows")
                counts["rollouts"]+=1
                blocks=list(range(len(ids)//64))
                rng.shuffle(blocks)
                for block in blocks[:max_per_conv]:
                    counts["windows"]+=1
                    window=ids[block*64:(block+1)*64]
                    wh=fingerprint(window)
                    if wh in seen_window:
                        counts["duplicate_window"]+=1
                        continue
                    seen_window.add(wh)
                    if any(t in special for t in window):
                        counts["special"]+=1
                        continue
                    if PLACEHOLDER.search(tokenizer.decode(window)):
                        counts["placeholder"]+=1
                        continue
                    n=rng.choice(LENGTHS)
                    crop=window[:n]
                    h=fingerprint(crop)
                    if h in seen[n]:
                        counts["duplicate_crop"]+=1
                        continue
                    seen[n].add(h)
                    if h in excluded[n]:
                        counts["ar_prefix_excluded"]+=1
                        continue
                    # Stable conversation-level split, not random per window.
                    identity=f'{path.name}:{row.get("rollout_id")}'
                    split="val" if int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8],"big")/2**64 < val_fraction else "train"
                    records[split].append(crop)
    if not all(records.values()):
        raise ValueError("Empty train or val pool")
    return records,counts

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir",type=Path,required=True)
    p.add_argument("--pairs-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",default="Qwen/Qwen3.5-4B")
    p.add_argument("--seed",type=int,default=42)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists")
    rolls=sorted(a.rollout_dir.glob("rollouts_*.jsonl"))
    pairs=sorted(a.pairs_dir.glob("pairs_l26_*.safetensors"))
    if not rolls or not pairs:p.error("Missing rollout or L26 pair shards")
    tokenizer=AutoTokenizer.from_pretrained(a.model)
    records,stats=create_pool(rolls,pairs,tokenizer,a.seed)
    a.output.mkdir(parents=True)
    tensors={}
    meta={"format":"fresh_ao_pool_v1","seed":a.seed,"model":a.model,"lengths":LENGTHS,
          "window":64,"max_per_conversation":4,"counts":stats,
          "source_rollouts":[x.name for x in rolls],"source_pairs":[x.name for x in pairs]}
    for split,crops in records.items():
        padded=torch.zeros((len(crops),64),dtype=torch.int32)
        lengths=torch.tensor([len(c) for c in crops],dtype=torch.int32)
        for j,c in enumerate(crops):padded[j,:len(c)]=torch.tensor(c,dtype=torch.int32)
        tensors[split+"_ids"]=padded
        tensors[split+"_lengths"]=lengths
        meta[split+"_rows"]=len(crops)
    target=a.output/"pool.safetensors"
    save_file(tensors,str(target))
    meta["pool_sha256"]=hashlib.sha256(target.read_bytes()).hexdigest()
    (a.output/"meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    (a.output/"COMPLETE").write_text("ok\n")
    print(json.dumps(meta,indent=2))
if __name__=="__main__":main()
