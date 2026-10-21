#!/usr/bin/env python3
"""Choose up to four nonnegative complementary L26 candidate directions.

Agam-style within-item NNOMP: prefix candidates -> frozen AR ->
center/whiten with frozen AR whitener -> unit atoms -> greedy positive
residual correlation, refitting nonnegative coefficients after each pick.
Queries are TRUE residuals, never AR outputs. No shrink: upstream retired
shortest-prefix shrink from the production distillation data path.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
sys.path.insert(0, str(ROOT / "oracle"))
from train_ar import ARCollator, build_ar  # noqa: E402
from prepare_distill_items import degenerate  # noqa: E402

PREFIXES = (2, 4, 8, 16, 32, 64)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def candidates(samples, tokenizer, lengths=PREFIXES):
    """Unique nondegenerate token prefixes, retaining their sample provenance."""
    out, seen = [], set()
    for si, sample in enumerate(samples):
        ids = sample["ids"]
        if not isinstance(ids, list) or not all(isinstance(t, int) for t in ids):
            raise ValueError("Teacher sample lacks integer token IDs")
        for n in lengths:
            if len(ids) < n:
                continue
            key = tuple(ids[:n])
            if key in seen:
                continue
            seen.add(key)
            text = tokenizer.decode(list(key), skip_special_tokens=True)
            if degenerate(text):
                continue
            out.append({"ids": list(key), "len": n, "src": si, "text": text})
    return out


def nnls(dirs, target, iters=80):
    """Projected gradient NNLS using 1/trace(Gram), as in Agam's nnomp.py."""
    gram = dirs @ dirs.T
    rhs = dirs @ target
    rate = 1.0 / gram.diag().sum().clamp_min(1e-12)
    weights = torch.zeros(len(dirs), dtype=target.dtype, device=target.device)
    for _ in range(iters):
        weights = (weights - rate * (gram @ weights - rhs)).clamp_min(0)
    recon = weights @ dirs
    fve = 1 - (target - recon).square().sum() / target.square().sum().clamp_min(1e-12)
    return weights, float(fve)


def nnomp(target, directions, k=4, min_gain=1e-3):
    """Within-item nonnegative OMP with NNLS refit and gain stopping."""
    if target.ndim != 1 or directions.ndim != 2 or directions.shape[1] != len(target):
        raise ValueError("NNOMP geometry mismatch")
    if not torch.isfinite(directions).all() or not torch.isfinite(target).all():
        raise ValueError("Non-finite NNOMP inputs")
    if target.square().sum() < 1e-12:
        raise ValueError("Zero-norm target")
    selected, coeffs, residual, prev = [], None, target.clone(), 0.0
    for _ in range(min(k, len(directions))):
        scores = directions @ residual
        if selected:
            scores[selected] = -torch.inf
        idx = int(scores.argmax())
        if float(scores[idx]) <= 0:
            break
        trial = selected + [idx]
        w, fve = nnls(directions[trial], target)
        if fve - prev <= min_gain:
            break
        selected, coeffs, prev = trial, w, fve
        residual = target - w @ directions[trial]
    return selected, [] if coeffs is None else coeffs.tolist(), prev


class ARInput(Dataset):
    def __init__(self, rows, tag):
        self.rows, self.tag = rows, tag
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        return {"input_ids": torch.cat((self.tag, torch.tensor(self.rows[i]["ids"],dtype=torch.long))),
                "target":torch.zeros(2560)}


@torch.inference_mode()
def embed_candidates(ar, rows, tag, pad_id, batch_size):
    ds=ARInput(rows,tag)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,collate_fn=ARCollator(pad_id))
    vec=[]
    for b in loader:
        v=ar(b["input_ids"].cuda(),b["attention_mask"].cuda(),b["lengths"].cuda())
        vec.append(v.float().cpu())
    return torch.cat(vec)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples",type=Path,required=True)
    p.add_argument("--items",type=Path,required=True)
    p.add_argument("--ar-checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--min-gain",type=float,default=.001)
    p.add_argument("--dry-run",action="store_true",help="Validate source and count candidates without GPU")
    a=p.parse_args()
    if not 0 < a.batch_size or a.output.exists():
        p.error("Invalid batch size or output already exists")
    if not (a.items/"COMPLETE").exists() or not (a.ar_checkpoint/"COMPLETE").exists():
        p.error("Source items or frozen AR incomplete")
    smeta=json.loads((a.samples/"manifest.json").read_text())
    imeta=json.loads((a.items/"meta.json").read_text())
    cfg=json.loads((a.ar_checkpoint/"run_config.json").read_text())
    if (smeta["format"] != "gt_distill_teacher_samples_v1"
        or smeta["items_sha256"] != imeta["items_sha256"]
        or Path(smeta["items_path"]).resolve() != a.items.resolve()
        or smeta["model"] != cfg["model_id"]
        or smeta["layer"] != cfg["target_layer"]
        or smeta["layer"] != imeta["layer"]
        or smeta["split"] not in ("train","val")):
        p.error("Teacher, item, or AR contract mismatch")
    if sha256(a.items/"items.safetensors") != imeta["items_sha256"]:
        p.error("GT items checksum mismatch")
    split=smeta["split"]
    source=load_file(str(a.items/"items.safetensors"))
    tokenizer=AutoTokenizer.from_pretrained(smeta["checkpoint"]+"/tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    paths=[a.samples/f"item_{i:06d}.json"
           for i in range(smeta["start"],smeta["start"]+smeta["count"])]
    all_rows=[]
    for file in paths:
        if not file.is_file():
            p.error(f"Missing teacher item: {file}")
        d=json.loads(file.read_text())
        i=int(file.stem.split("_")[-1])
        if (d.get("item_idx")!=i or d.get("split")!=split
            or d.get("pair_row")!=int(source[split+"_pair_rows"][i])
            or len(d.get("samples",[]))!=smeta["samples_per_item"]):
            p.error(f"Inconsistent sample artifact: {file}")
        all_rows.append((i,candidates(d["samples"],tokenizer)))
    report={"schema":"l26_nnomp_smoke_v1","sample_manifest_sha256":sha256(a.samples/"manifest.json"),
            "items_sha256":imeta["items_sha256"],"ar_checkpoint":str(a.ar_checkpoint),
            "layer":smeta["layer"],"split":split,"k":4,"min_gain":a.min_gain,
            "source":"true_GT_residual","rows":[]}
    for i,rows in all_rows:
        report["rows"].append({"item_idx":i,"n_candidates":len(rows)})
    print("CANDIDATES",json.dumps(report["rows"]),flush=True)
    if a.dry_run:
        print("DRY RUN: no GPU, no output",flush=True)
        return
    if not torch.cuda.is_available():
        p.error("CUDA required for frozen AR inference")
    white_path=a.ar_checkpoint/f"whitener_l{smeta['layer']}.safetensors"
    w=load_file(str(white_path))
    mean=w["mean"].float().cuda()
    W=w["whitener"].float().cuda()
    if W.shape != (len(mean),len(mean)):
        raise RuntimeError("Frozen AR whitener dimension mismatch")
    tag=torch.tensor(tokenizer(f"[Layer {smeta['layer']:02d}]",add_special_tokens=False).input_ids,dtype=torch.long)
    config=argparse.Namespace(model=smeta["model"],gpu=0,layer=smeta["layer"],
          cache_dir="/root/.cache/huggingface/hub",lora_r=16,lora_alpha=32,lora_dropout=.05)
    ar=build_ar(config,tokenizer,torch.zeros(len(mean)),len(mean),torch.device("cuda"),a.ar_checkpoint).eval()
    for item, (i,rows) in zip(report["rows"],all_rows):
        if len(rows)<4:
            item["skipped"]="fewer than 4 candidate prefixes"
            continue
        raw=embed_candidates(ar,rows,tag,tokenizer.pad_token_id,a.batch_size).cuda()
        # AR whitener fit on AR training residuals; do not refit on distill items.
        atom_w=F.normalize((raw-mean)@W,dim=1,eps=1e-9)
        true=source[split+"_vec"][i].float().cuda()
        target=(true-mean)@W
        sel, coef, fve=nnomp(target,atom_w,k=4,min_gain=a.min_gain)
        item["omp_fve"]=fve
        item["coefficients"]=coef
        item["omp_sel"]=[{"text":rows[j]["text"],"ids":rows[j]["ids"],
                          "len":rows[j]["len"],"src":rows[j]["src"]}
                         for j in sel]
        print(f"item={i} candidates={len(rows)} picks={len(sel)} joint_fve={fve:.4f}",flush=True)
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    (a.output/"COMPLETE").write_text("ok\n")
    print("NNOMP COMPLETE",a.output/"report.json",flush=True)


if __name__=="__main__":
    main()
