#!/usr/bin/env python3
"""Cross-evaluate GT-SFT and AO-1 adapters against real-h and AR(span) vectors.

Same frozen oracle-validation indices, same target tokens, same injector and
shuffle for all cells. No new training, no alteration of prior checkpoints.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "oracle"))
from eval_gt_sft import validation_indices, derangement, scores
from train_gt_sft import SpanDataset, collate, load_corpus, PROMPT

SOURCES = {
    "real_h": "true post-block activations from AR-held-out rollouts",
    "ar_out": "frozen_AR_of_span_on_AR_heldout_rollouts",
}

@torch.inference_mode()
def evaluate(model, prompts, tokenizer, ds, order, sources, alpha, batch_size):
    rows = [ds[i] for i in range(len(ds))]
    totals = {}
    for source, vectors in sources.items():
        if len(vectors) != len(rows):
            raise RuntimeError("Vector source length mismatch")
        for kind in ("correct", "shuffled", "zero"):
            totals[f"{source}/{kind}"] = [0.0, 0]
        for lo in range(0, len(rows), batch_size):
            batch = rows[lo:lo+batch_size]
            _, ids, labels, mask = collate(batch, tokenizer.pad_token_id, tokenizer.eos_token_id)
            ids, labels, mask = (t.cuda() for t in (ids, labels, mask))
            v = vectors[lo:lo+len(batch)].cuda()
            shift = vectors[torch.tensor(order[lo:lo+len(batch)])].cuda()
            for kind, x in (("correct", v), ("shuffled", shift), ("zero", torch.zeros_like(v))):
                loss, n = scores(model, prompts, x, ids, labels, mask, alpha)
                rec = totals[f"{source}/{kind}"]
                rec[0] += loss
                rec[1] += n
    return {k: {"token_ce": val/n, "tokens": n} for k, (val,n) in totals.items() if n > 0}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-checkpoint", type=Path, required=True)
    p.add_argument("--ao-checkpoint", type=Path, required=True)
    p.add_argument("--arout", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("/workspace/o-lens-qwen35-4b/data/olens_200k_all32"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    if a.batch_size <= 0 or a.output.exists():
        p.error("Invalid batch size or existing output directory")
    metas = {}
    for name, checkpoint in (("gt", a.gt_checkpoint), ("ao", a.ao_checkpoint)):
        if not (checkpoint/"COMPLETE").is_file():
            p.error(f"Checkpoint not complete: {checkpoint}")
        meta = json.loads((checkpoint/"meta.json").read_text())
        if meta.get("source") != SOURCES["real_h" if name=="gt" else "ar_out"]:
            p.error(f"Wrong injection source for {name}")
        if meta.get("transform") != "unit" or meta.get("prompt") != PROMPT or meta.get("prompt_slot") != "last token of rendered prompt":
            p.error(f"Injection contract mismatch: {name}")
        metas[name] = meta
    gt, ao = metas["gt"], metas["ao"]
    for key in ("model", "layer", "seed", "alpha", "partition", "train_indices_sha256", "val_indices_sha256"):
        if gt.get(key) != ao.get(key):
            p.error(f"Models are not comparable: {key}")
    for key in ("max_train", "max_val", "max_span_tokens"):
        if gt["training"].get(key, {"max_train":2048,"max_val":256}.get(key)) != ao["training"].get(key, {"max_train":2048,"max_val":256}.get(key)):
            p.error(f"Data contract differs: {key}")
    data_args = argparse.Namespace(layer=gt["layer"], pair_dir=a.data/"pairs",
        rollout_dir=a.data/"rollouts", pair_glob=None, rollout_glob="rollouts_*.jsonl",
        split_seed=42, val_frac=0.10)
    corpus = load_corpus(data_args)
    indices = validation_indices(corpus, gt)
    if validation_indices(corpus, ao) != indices:
        raise RuntimeError("Different validation indices")
    ds = SpanDataset(corpus, indices, gt["training"]["max_span_tokens"])
    cache_meta = json.loads((a.arout/"meta.json").read_text())
    if not (a.arout/"COMPLETE").is_file() or cache_meta["arout_sha256"] != ao["arout_sha256"]:
        p.error("AR-output cache incomplete or mismatched")
    tensor_file = a.arout/"arout.safetensors"
    if hashlib.sha256(tensor_file.read_bytes()).hexdigest() != cache_meta["arout_sha256"]:
        p.error("AR-output file SHA mismatch")
    if cache_meta["splits"]["val"]["indices_sha256"] != gt["val_indices_sha256"]:
        p.error("AR-output cache indexes don't match")
    ar_out = load_file(str(tensor_file))["val"]
    real_h = torch.stack([ds[j][0] for j in range(len(ds))])
    if ar_out.shape != real_h.shape:
        raise RuntimeError("Input source tensor shapes differ")
    sources = {"real_h": real_h, "ar_out": ar_out}
    order = derangement(len(ds), gt["seed"])
    report = {"status": "pilot_validation_only_not_untouched_final",
        "val_rows":len(ds), "alpha":gt["alpha"], "shuffle":"identical fixed circular shift per source",
        "checkpoints":{"gt":str(a.gt_checkpoint),"ao":str(a.ao_checkpoint)},
        "input_sources":list(sources), "models":{}}
    for name, checkpoint in (("gt", a.gt_checkpoint), ("ao", a.ao_checkpoint)):
        meta = metas[name]
        tokenizer = AutoTokenizer.from_pretrained(checkpoint/"tokenizer")
        model = AutoModelForCausalLM.from_pretrained(meta["model"], revision=meta["revision"],
                                                     dtype=torch.bfloat16, device_map="cuda")
        model = PeftModel.from_pretrained(model, checkpoint/"lora", is_trainable=False).eval()
        prompts = torch.tensor(tokenizer(PROMPT, add_special_tokens=False).input_ids,device="cuda")
        result = evaluate(model, prompts, tokenizer, ds, order, sources, meta["alpha"], a.batch_size)
        report["models"][name] = result
        print(name, json.dumps(result, indent=2), flush=True)
        del model
        torch.cuda.empty_cache()
    for name, data in report["models"].items():
        for source in sources:
            data[source+"/correct_minus_shuffled"] = data[source+"/correct"]["token_ce"]-data[source+"/shuffled"]["token_ce"]
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print("COMPLETE:",a.output/"report.json")

if __name__=="__main__":
    main()
