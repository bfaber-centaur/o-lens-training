#!/usr/bin/env python3
"""Evaluate a trained GT soft-token SFT oracle with matched activation controls.

Teacher-forced token NLL on exactly the original frozen oracle validation rows:
correct activation, deterministic derangement, and zero activation.
No training, checkpoint edits or mutation of original data.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "oracle"))
from train_gt_sft import (load_corpus, oracle_partition, SpanDataset,
                          collate, conditioned_batch, PROMPT, SPLIT_VERSION)


def validation_indices(corpus, meta):
    if meta["partition"] != SPLIT_VERSION:
        raise ValueError("Unsupported checkpoint partition contract")
    rng = random.Random(meta["seed"])
    split = {"train": [], "val": []}
    for i in corpus.val_indices.tolist():
        split[oracle_partition(corpus.shard_ids[i], corpus.rollout_ids[i], meta["seed"], 0.2)].append(i)
    for kind, limit in (("train", meta["training"].get("max_train", 2048)),
                        ("val", meta["training"].get("max_val", 256))):
        rng.shuffle(split[kind])
        split[kind] = split[kind][:limit]
        actual = hashlib.sha256(json.dumps(split[kind]).encode()).hexdigest()
        if actual != meta[f"{kind}_indices_sha256"]:
            raise ValueError(f"{kind} source indices differ from checkpoint: {actual}")
    return split["val"]


def derangement(n, seed):
    if n < 2:
        raise ValueError("Need two or more validation examples for shuffle control")
    shift = random.Random(seed + 1_337).randrange(1, n)
    return [(i + shift) % n for i in range(n)]


def scores(model, prompts, vectors, ids, labs, mask, alpha):
    emb, att, labels = conditioned_batch(model, prompts, vectors, ids, labs, mask, alpha)
    logits = model(inputs_embeds=emb, attention_mask=att).logits
    # HF causal-LM objective: token at t is predicted from position t-1.
    logits = logits[:, :-1, :].float()
    target = labels[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           target.reshape(-1), ignore_index=-100, reduction="sum")
    n = int((target != -100).sum().item())
    return float(loss), n


@torch.inference_mode()
def generate_one(model, prompt_ids, vector, alpha, tokenizer, max_tokens):
    device = vector.device
    embed = model.get_input_embeddings()
    p = prompt_ids[None, :]
    pe = embed(p).clone()
    pe[:, -1] = (F.normalize(vector.float(), dim=-1, eps=1e-9) * alpha).to(pe.dtype)
    generated = []
    model.eval()
    for _ in range(max_tokens):
        logits = model(inputs_embeds=pe, use_cache=False).logits[:, -1]
        nxt = torch.argmax(logits, dim=-1)
        if int(nxt.item()) == tokenizer.eos_token_id:
            break
        generated.append(int(nxt.item()))
        pe = torch.cat([pe, embed(nxt[:, None])], dim=1)
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("/workspace/o-lens-qwen35-4b/data/olens_200k_all32"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sample-count", type=int, default=3)
    ap.add_argument("--max-generate-tokens", type=int, default=32)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Output exists: refusing overwrite")
    if not (args.checkpoint / "COMPLETE").is_file():
        ap.error("Incomplete checkpoint")
    if args.batch_size < 1 or args.sample_count < 0 or args.max_generate_tokens < 1:
        ap.error("Invalid eval settings")
    meta = json.loads((args.checkpoint / "meta.json").read_text())
    if (meta.get("prompt") != PROMPT or meta.get("prompt_slot") != "last token of rendered prompt"
        or meta.get("transform") != "unit" or meta.get("source") != "true post-block activations from AR-held-out rollouts"):
        ap.error("Unexpected injection contract; refusing to guess")
    layer = int(meta["layer"])
    data_args = argparse.Namespace(layer=layer, pair_dir=args.data/"pairs", rollout_dir=args.data/"rollouts",
                                   pair_glob=None, rollout_glob="rollouts_*.jsonl",
                                   split_seed=42, val_frac=0.10)
    corpus = load_corpus(data_args)
    val_indices = validation_indices(corpus, meta)
    ds = SpanDataset(corpus, val_indices, meta["training"]["max_span_tokens"])
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint/"tokenizer")
    base = AutoModelForCausalLM.from_pretrained(meta["model"], revision=meta["revision"],
                                                dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.checkpoint/"lora", is_trainable=False).eval()
    # Assert some adapter keys actually loaded and the model includes LoRA layers.
    from peft.tuners.lora import LoraLayer
    if not any(isinstance(m, LoraLayer) for m in model.modules()):
        raise RuntimeError("No live LoRA modules found after adapter load")
    prompts = torch.tensor(tokenizer(meta["prompt"], add_special_tokens=False).input_ids, device="cuda")
    rows = [ds[i] for i in range(len(ds))]
    shuffle = derangement(len(rows), meta["seed"])
    totals = {c: [0.0, 0] for c in ("correct", "shuffled", "zero")}
    for lo in range(0, len(rows), args.batch_size):
        batch = rows[lo:lo+args.batch_size]
        vectors, ids, labs, mask = collate(batch, tokenizer.pad_token_id, tokenizer.eos_token_id)
        vectors, ids, labs, mask = [x.cuda() for x in (vectors, ids, labs, mask)]
        shifted = torch.stack([rows[shuffle[j]][0] for j in range(lo, min(lo+args.batch_size,len(rows)))]).cuda()
        conditions = {"correct": vectors, "shuffled": shifted, "zero": torch.zeros_like(vectors)}
        for name, v in conditions.items():
            total, n = scores(model, prompts, v, ids, labs, mask, float(meta["alpha"]))
            totals[name][0] += total
            totals[name][1] += n
    report = {"checkpoint": str(args.checkpoint),
              "validation_rows": len(rows),
              "metrics": {k: {"token_ce": t/n, "supervised_tokens": n} for k,(t,n) in totals.items()},
              "shuffle": "fixed nonzero circular shift of frozen val rows",
              "evaluation_status": "oracle-validation-only, not untouched final evaluation"}
    report["correct_minus_shuffled_ce"] = report["metrics"]["correct"]["token_ce"] - report["metrics"]["shuffled"]["token_ce"]
    samples = []
    for j in range(min(args.sample_count, len(rows))):
        vec, span = rows[j]
        vec = vec[None, :].cuda()
        samples.append({
            "target": tokenizer.decode(span.tolist(), skip_special_tokens=True),
            "correct": generate_one(model, prompts, vec, meta["alpha"], tokenizer, args.max_generate_tokens),
            "shuffled": generate_one(model, prompts, rows[shuffle[j]][0][None, :].cuda(), meta["alpha"], tokenizer, args.max_generate_tokens),
            "zero": generate_one(model, prompts, torch.zeros_like(vec), meta["alpha"], tokenizer, args.max_generate_tokens),
        })
    report["samples"] = samples
    args.output.mkdir(parents=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
