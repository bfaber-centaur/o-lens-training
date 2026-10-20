#!/usr/bin/env python3
"""Resumable L26 AR-output supervised inversion, before distillation/RL.

Scientific scope: frozen AR(span) vectors -> original text span.
This is AO-1 style AR-output inversion, not distillation or RL.
AR's validation split is used as a *source pool* and re-split by rollout for
oracle train/validation; therefore this is NOT an untouched test of AR quality.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy"))
from train_ar import load_corpus  # noqa: E402

PROMPT = "Read the following representation and reproduce the text:\n"
SPLIT_VERSION = "ar_val_rollout_hash_v1"


def oracle_partition(shard, rollout, seed, val_frac):
    key = f"{int(shard)}:{int(rollout)}:{seed}".encode()
    u = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    return "val" if u < val_frac else "train"


class SpanDataset(Dataset):
    def __init__(self, corpus, indices, max_span_tokens):
        self.corpus = corpus
        self.indices = indices
        self.max_span_tokens = max_span_tokens

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        lo = int(self.corpus.span_offsets[i])
        hi = int(self.corpus.span_offsets[i + 1])
        return self.corpus.activations[i], self.corpus.span_token_ids[lo:hi].long()[:self.max_span_tokens]


def collate(rows, pad_id, eos_id):
    b = len(rows)
    vectors = torch.stack([r[0] for r in rows])
    targets = [torch.cat([r[1], torch.tensor([eos_id])]) for r in rows]
    m = max(map(len, targets))
    ids = torch.full((b, m), pad_id, dtype=torch.long)
    labels = torch.full((b, m), -100, dtype=torch.long)
    mask = torch.zeros((b, m), dtype=torch.long)
    for j, tgt in enumerate(targets):
        n = len(tgt)
        ids[j, :n] = tgt
        labels[j, :n] = tgt
        mask[j, :n] = 1
    return vectors, ids, labels, mask


def conditioned_batch(model, prompt_ids, vector, target_ids, labels, target_mask, alpha):
    """Return inputs_embeds, attention_mask, labels with final prompt slot replaced."""
    embed = model.get_input_embeddings()
    b = vector.shape[0]
    prompt = prompt_ids[None, :].expand(b, -1)
    prompt_embed = embed(prompt).clone()
    assert vector.ndim == 2 and vector.shape[-1] == prompt_embed.shape[-1], "Activation/input-embedding dimension mismatch"
    scaled = F.normalize(vector.float(), dim=-1, eps=1e-9) * alpha
    prompt_embed[:, -1, :] = scaled.to(prompt_embed.dtype)
    inputs = torch.cat([prompt_embed, embed(target_ids)], dim=1)
    attn = torch.cat([torch.ones((b, prompt.shape[1]), device=vector.device, dtype=target_mask.dtype), target_mask], dim=1)
    full_labels = torch.cat([torch.full((b, prompt.shape[1]), -100, device=vector.device, dtype=labels.dtype), labels], dim=1)
    return inputs, attn, full_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--arout", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--data", type=Path, default=Path("/workspace/o-lens-qwen35-4b/data/olens_200k_all32"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-span-tokens", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=12000)
    ap.add_argument("--max-val", type=int, default=1024)
    ap.add_argument("--alpha", type=float, default=8000.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.alpha <= 0:
        ap.error("invalid step/batch/injection scale")
    if args.resume:
        if not (args.output / "latest.json").exists():
            ap.error("Resume requested without valid latest.json")
    elif args.output.exists():
        ap.error(f"output already exists; never overwrite: {args.output}")
    if args.save_every < 1:
        ap.error("--save-every must be positive")
    if not torch.cuda.is_available():
        ap.error("CUDA GPU required")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output.mkdir(parents=True)
    data_args = argparse.Namespace(layer=args.layer, pair_dir=args.data / "pairs",
                                   rollout_dir=args.data / "rollouts", pair_glob=None,
                                   rollout_glob="rollouts_*.jsonl", split_seed=42, val_frac=0.10)
    corpus = load_corpus(data_args)
    if not (args.arout / "COMPLETE").is_file():
        ap.error("Incomplete AR-output cache")
    armeta = json.loads((args.arout / "meta.json").read_text())
    if armeta["model"] != args.model or armeta["layer"] != args.layer or armeta["seed"] != args.seed:
        ap.error("AR-output cache model/layer/seed mismatch")
    if armeta["max_train"] != args.max_train or armeta["max_val"] != args.max_val:
        ap.error("AR-output cache split size mismatch")
    arout_path = args.arout / "arout.safetensors"
    if hashlib.sha256(arout_path.read_bytes()).hexdigest() != armeta["arout_sha256"]:
        ap.error("AR-output file SHA mismatch")
    arout = load_file(str(arout_path))
    rng = random.Random(args.seed)
    split = {"train": [], "val": []}
    # AR held-out rows only; a single rollout cannot cross oracle train/val.
    for i in corpus.val_indices.tolist():
        k = oracle_partition(corpus.shard_ids[i], corpus.rollout_ids[i], args.seed, 0.2)
        split[k].append(i)
    if not split["train"] or not split["val"]:
        raise RuntimeError("oracle split empty")
    for k, limit in [("train", args.max_train), ("val", args.max_val)]:
        rng.shuffle(split[k])
        split[k] = split[k][:limit]
    for part in ("train", "val"):
        digest = hashlib.sha256(json.dumps(split[part]).encode()).hexdigest()
        if digest != armeta["splits"][part]["indices_sha256"]:
            raise RuntimeError(f"{part} split mismatch against AR output cache")
        if tuple(arout[part].shape) != (len(split[part]), corpus.d_model):
            raise RuntimeError(f"{part} AR-output shape mismatch")
    train_rollouts = {(int(corpus.shard_ids[i]), int(corpus.rollout_ids[i])) for i in split["train"]}
    val_rollouts = {(int(corpus.shard_ids[i]), int(corpus.rollout_ids[i])) for i in split["val"]}
    assert train_rollouts.isdisjoint(val_rollouts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False).input_ids
    if not prompt_ids:
        raise RuntimeError("empty prompt")
    device = torch.device("cuda")
    base = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 dtype=torch.bfloat16).to(device)
    base.config.use_cache = False
    if base.get_input_embeddings().weight.shape[1] != corpus.d_model:
        raise RuntimeError("dimension mismatch for direct soft-token injection")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    if args.resume:
        latest = json.loads((args.output / "latest.json").read_text())
        last_path = args.output / latest["checkpoint"]
        if not (last_path / "COMPLETE").is_file():
            ap.error("Incomplete resume checkpoint")
        model = PeftModel.from_pretrained(base, last_path / "lora", is_trainable=True)
    else:
        model = get_peft_model(base, lora)
    model.train()
    prompt_ids = torch.tensor(prompt_ids, device=device)
    collator = lambda rows: collate(rows, tokenizer.pad_token_id, tokenizer.eos_token_id)
    class AROutDataset(SpanDataset):
        def __init__(self, corpus, indices, vectors, max_span_tokens):
            super().__init__(corpus, indices, max_span_tokens)
            self.vectors = vectors
        def __getitem__(self, j):
            _, ids = super().__getitem__(j)
            return self.vectors[j], ids

    train_loader = DataLoader(AROutDataset(corpus, split["train"], arout["train"], args.max_span_tokens),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(AROutDataset(corpus, split["val"], arout["val"], args.max_span_tokens),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    meta = dict(stage="l26_ao1_arout_sft_pilot", layer=args.layer, model=args.model,
                arout_source=str(args.arout), arout_sha256=armeta["arout_sha256"],
                ar_checkpoint=armeta["checkpoint"],
                revision=args.revision or getattr(base.config, "_commit_hash", None),
                prompt=PROMPT, prompt_slot="last token of rendered prompt",
                source="frozen_AR_of_span_on_AR_heldout_rollouts",
                transform="unit", alpha=args.alpha,
                partition=SPLIT_VERSION, seed=args.seed,
                train_indices_sha256=hashlib.sha256(json.dumps(split["train"]).encode()).hexdigest(),
                val_indices_sha256=hashlib.sha256(json.dumps(split["val"]).encode()).hexdigest(),
                train_rows=len(split["train"]), val_rows=len(split["val"]),
                training=dict(max_steps=args.max_steps, batch_size=args.batch_size,
                              grad_accum=args.grad_accum, lr=args.lr, max_span_tokens=args.max_span_tokens,
                              max_train=args.max_train, max_val=args.max_val),
                eval_status="pilot_validation_only_not_independent_test")
    if args.resume:
        prior = json.loads((args.output / "meta.json").read_text())
        if prior != meta:
            ap.error("Resume metadata differs; includes model, corpus, scale, schedule and hashes")
    else:
        (args.output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # Each optimizer step owns its deterministic sample schedule. Interrupted and
    # resumed jobs use identical rows, without fragile DataLoader iterator state.
    train_ds = AROutDataset(corpus, split["train"], arout["train"], args.max_span_tokens)
    def checkpoint(step, metrics):
        final = args.output / "checkpoints" / f"step_{step:06d}"
        staging = args.output / "checkpoints" / f".step_{step:06d}.partial"
        if staging.exists() or final.exists():
            raise RuntimeError("Refusing to overwrite checkpoint")
        staging.mkdir(parents=True)
        model.save_pretrained(staging / "lora")
        tokenizer.save_pretrained(staging / "tokenizer")
        (staging / "meta.json").write_text(json.dumps(meta, indent=2)+"\n")
        torch.save({"optimizer": optim.state_dict(), "step": step,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, staging / "state.pt")
        (staging / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
        (staging / "COMPLETE").write_text("ok\n")
        os.replace(staging, final)
        pointer = args.output / "latest.json.tmp"
        pointer.write_text(json.dumps({"step":step,"checkpoint":str(final.relative_to(args.output))})+"\n")
        os.replace(pointer, args.output / "latest.json")

    start_step = 0
    if args.resume:
        state = torch.load(last_path / "state.pt", map_location="cpu", weights_only=False)
        if state["step"] != latest["step"]:
            ap.error("Checkpoint/pointer step mismatch")
        optim.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step = state["step"]
        if args.max_steps < start_step:
            ap.error("Target max steps less than saved progress")
    def validate():
        model.eval()
        total = 0.0
        tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                vec, ids, labs, mask = (t.to(device) for t in batch)
                emb, attn, labels = conditioned_batch(model, prompt_ids, vec, ids, labs, mask, args.alpha)
                loss = model(inputs_embeds=emb, attention_mask=attn, labels=labels).loss
                n = int((labels != -100).sum())
                total += loss.item()*n
                tokens += n
        if not tokens:
            raise RuntimeError("Validation has zero supervised tokens")
        return {"val_token_ce":total/tokens, "val_supervised_tokens":tokens}
    for step in range(start_step+1, args.max_steps+1):
        model.train()
        optim.zero_grad(set_to_none=True)
        train_ce = 0.0
        for micro in range(args.grad_accum):
            sampler = random.Random(f"{args.seed}:{step}:{micro}")
            chosen = [sampler.randrange(len(train_ds)) for _ in range(args.batch_size)]
            batch = collator([train_ds[i] for i in chosen])
            vec, ids, labs, mask = (t.to(device) for t in batch)
            emb, attn, labels = conditioned_batch(model, prompt_ids, vec, ids, labs, mask, args.alpha)
            loss = model(inputs_embeds=emb, attention_mask=attn, labels=labels).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            (loss / args.grad_accum).backward()
            train_ce += loss.item()/args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        print(f"step={step} train_ce={train_ce:.4f}",flush=True)
        if step % args.save_every == 0 or step == args.max_steps:
            metrics = {"step":step,"train_token_ce":train_ce,**validate()}
            checkpoint(step, metrics)
            print("SAVED",json.dumps(metrics),flush=True)
    tokenizer.save_pretrained(args.output / "tokenizer")
    print("TRAINING COMPLETE", args.output)

if __name__ == "__main__":
    main()
