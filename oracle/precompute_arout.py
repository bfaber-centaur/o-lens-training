#!/usr/bin/env python3
"""Precompute frozen L26 AR(span) representations for the exact oracle pilot split.

Never uses true activations as the oracle's input. For each selected span, apply
the saved AR to '[Layer 26]' + span tokens, exactly as during AR training.
Read original checkpoint; write new self-contained, index-anchored cache.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
from train_ar import (load_corpus, build_ar, ARDataset, ARCollator)
sys.path.insert(0, str(ROOT / "oracle"))
from train_gt_sft import oracle_partition, SPLIT_VERSION


def build_indices(corpus, seed, max_train, max_val):
    rng = random.Random(seed)
    result = {"train": [], "val": []}
    for i in corpus.val_indices.tolist():
        key = oracle_partition(corpus.shard_ids[i], corpus.rollout_ids[i], seed, 0.2)
        result[key].append(i)
    for k, n in (("train", max_train), ("val", max_val)):
        rng.shuffle(result[k])
        result[k] = result[k][:n]
        if not result[k]:
            raise ValueError(f"Empty {k} split")
    a = {(int(corpus.shard_ids[i]), int(corpus.rollout_ids[i])) for i in result["train"]}
    b = {(int(corpus.shard_ids[i]), int(corpus.rollout_ids[i])) for i in result["val"]}
    if not a.isdisjoint(b):
        raise ValueError("Rollout leakage")
    return result


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=Path("/workspace/o-lens-qwen35-4b/ar/l26_200k_all32/checkpoints/step_001424"))
    ap.add_argument("--data", type=Path, default=Path("/workspace/o-lens-qwen35-4b/data/olens_200k_all32"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=2048)
    ap.add_argument("--max-val", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists; never overwrite")
    if not (args.checkpoint / "COMPLETE").is_file():
        ap.error("AR checkpoint lacks COMPLETE marker")
    ckpt_cfg = json.loads((args.checkpoint / "run_config.json").read_text())
    if ckpt_cfg["model_id"] != args.model or ckpt_cfg["target_layer"] != args.layer:
        ap.error("AR checkpoint model/layer mismatch")
    args.output.mkdir(parents=True)
    import argparse as ns
    corpus = load_corpus(ns.Namespace(layer=args.layer, pair_dir=args.data / "pairs",
                                      rollout_dir=args.data / "rollouts", pair_glob=None,
                                      rollout_glob="rollouts_*.jsonl",
                                      split_seed=42, val_frac=0.10))
    splits = build_indices(corpus, args.seed, args.max_train, args.max_val)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="/root/.cache/huggingface/hub")
    tag = tokenizer(f"[Layer {args.layer:02d}]", add_special_tokens=False)["input_ids"]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = ns.Namespace(model=args.model, gpu=0, layer=args.layer,
                       cache_dir="/root/.cache/huggingface/hub", lora_r=16,
                       lora_alpha=32, lora_dropout=0.05)
    ar = build_ar(cfg, tokenizer, torch.zeros(corpus.d_model), corpus.d_model,
                  torch.device("cuda"), args.checkpoint).eval()
    outs = {}
    for split, indices in splits.items():
        ds = ARDataset(torch.tensor(indices), corpus.span_token_ids,
                       corpus.span_offsets, corpus.activations, tag)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=ARCollator(tokenizer.pad_token_id))
        preds = []
        for batch in loader:
            pred = ar(batch["input_ids"].cuda(), batch["attention_mask"].cuda(),
                      batch["lengths"].cuda())
            preds.append(pred.cpu())
        outs[split] = torch.cat(preds, dim=0).contiguous()
        if len(outs[split]) != len(indices) or outs[split].shape[1] != corpus.d_model:
            raise RuntimeError("AR output geometry mismatch")
    save_file({k: v for k,v in outs.items()}, str(args.output / "arout.safetensors"))
    digest = hashlib.sha256((args.output / "arout.safetensors").read_bytes()).hexdigest()
    meta = {"source": "frozen_AR_of_span", "partition": SPLIT_VERSION,
            "model": args.model, "layer": args.layer, "seed": args.seed,
            "checkpoint": str(args.checkpoint),
            "arout_sha256": digest, "max_train": args.max_train, "max_val": args.max_val,
            "splits": {k: {"count": len(v),
                          "indices_sha256": hashlib.sha256(json.dumps(v).encode()).hexdigest()}
                       for k,v in splits.items()}}
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.output / "COMPLETE").write_text("ok\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
