#!/usr/bin/env python3
"""Precompute frozen single-layer AR outputs for fresh AO token-id crops."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file,save_file
from torch.utils.data import DataLoader,Dataset
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"legacy"))
from train_ar import build_ar, ARCollator

class Crops(Dataset):
    def __init__(self,ids,lengths,tag):
        self.ids=ids;self.lengths=lengths;self.tag=tag
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        # Same tuple schema expected by ARCollator.
        crop=self.ids[i,:int(self.lengths[i])].long()
        return {"input_ids":torch.cat((self.tag,crop)), "target":torch.zeros(2560)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--layer",type=int,default=26)
    a=p.parse_args()
    if a.output.exists():p.error("Output already exists")
    if not (a.pool/"COMPLETE").is_file() or not (a.checkpoint/"COMPLETE").is_file():
        p.error("Incomplete pool or frozen AR")
    meta=json.loads((a.pool/"meta.json").read_text())
    artifact=a.pool/"pool.safetensors"
    if hashlib.sha256(artifact.read_bytes()).hexdigest()!=meta["pool_sha256"]:
        p.error("Pool checksum mismatch")
    if meta["format"]!="fresh_ao_pool_v1":p.error("Unrecognized pool contract")
    cfg=json.loads((a.checkpoint/"run_config.json").read_text())
    if cfg["target_layer"]!=a.layer or cfg["model_id"]!=meta["model"]:
        p.error("AR checkpoint model or layer mismatch")
    if meta.get("layer",26)!=a.layer:
        p.error("AO pool exclusion layer mismatch")
    tensors=load_file(str(artifact))
    tokenizer=AutoTokenizer.from_pretrained(meta["model"])
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    tag=torch.tensor(tokenizer(f"[Layer {a.layer:02d}]",add_special_tokens=False).input_ids,dtype=torch.long)
    args=argparse.Namespace(model=meta["model"],gpu=0,layer=a.layer,
            cache_dir="/root/.cache/huggingface/hub",lora_r=16,lora_alpha=32,lora_dropout=.05)
    ar=build_ar(args,tokenizer,torch.zeros(2560),2560,torch.device("cuda"),a.checkpoint).eval()
    out={}
    with torch.inference_mode():
        for split in ("train","val"):
            ds=Crops(tensors[split+"_ids"],tensors[split+"_lengths"],tag)
            loader=DataLoader(ds,batch_size=a.batch_size,shuffle=False,
                              collate_fn=ARCollator(tokenizer.pad_token_id))
            preds=[]
            for batch in loader:
                preds.append(ar(batch["input_ids"].cuda(),batch["attention_mask"].cuda(),batch["lengths"].cuda()).cpu())
            out[split]=torch.cat(preds).contiguous()
    a.output.mkdir(parents=True)
    fn=a.output/"arout.safetensors"
    save_file(out,str(fn))
    result={"format":"fresh_ao_arout_v1","model":meta["model"],"layer":a.layer,
            "pool":str(a.pool),"pool_sha256":meta["pool_sha256"],
            "checkpoint":str(a.checkpoint),
            "checkpoint_sha256":{str(x.relative_to(a.checkpoint)):hashlib.sha256(x.read_bytes()).hexdigest()
                for x in sorted(a.checkpoint.rglob("*")) if x.is_file() and x.name != "state.pt"},
            "arout_sha256":hashlib.sha256(fn.read_bytes()).hexdigest(),
            "train_rows":len(out["train"]),"val_rows":len(out["val"])}
    (a.output/"meta.json").write_text(json.dumps(result,indent=2)+"\n")
    (a.output/"COMPLETE").write_text("ok\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
