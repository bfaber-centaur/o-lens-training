#!/usr/bin/env python3
"""
Train an Oracle-Lens activation reconstructor (AR) from harvested span/activation pairs.

This script is the reusable version of the validated 02_train_ar notebook pipeline:
  * deterministic rollout-level train/val split
  * train-only empirical whitening
  * exact token-span inputs prefixed with "[Layer XX]"
  * Qwen truncated through the target layer
  * LoRA over all linear modules
  * LayerNorm -> Linear regression head, initialized to predict the train mean
  * whitened MSE objective
  * held-out MSE / FVE / cosine evaluation
  * gradient accumulation
  * resumable, self-contained checkpoints

Typical use:

    python train_ar.py \
      --layer 30 \
      --pair-dir /workspace/o-lens-qwen35-4b/data/pairs \
      --rollout-dir /workspace/o-lens-qwen35-4b/data/rollouts \
      --output /workspace/o-lens-qwen35-4b/ar/l30_200k \
      --epochs 2

Resume the latest checkpoint:

    python train_ar.py \
      --layer 30 \
      --pair-dir /workspace/o-lens-qwen35-4b/data/pairs \
      --rollout-dir /workspace/o-lens-qwen35-4b/data/rollouts \
      --output /workspace/o-lens-qwen35-4b/ar/l30_200k \
      --epochs 3 \
      --resume auto

Train another layer:

    python train_ar.py \
      --layer 26 \
      --pair-dir /workspace/o-lens-qwen35-4b/data/pairs \
      --rollout-dir /workspace/o-lens-qwen35-4b/data/rollouts \
      --output /workspace/o-lens-qwen35-4b/ar/l26_200k
"""

# Keep disposable caches off persistent /workspace network storage.
import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface/hub"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import hashlib
import json
import math
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------
# Utilities
# ----------------------------

SHARD_RE = re.compile(r"_(\d+)\.(?:safetensors|jsonl)$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a Qwen Oracle-Lens activation reconstructor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layer", type=int, required=True)

    p.add_argument("--pair-dir", required=True)
    p.add_argument("--rollout-dir", required=True)
    p.add_argument(
        "--pair-glob",
        default=None,
        help="Pair-file glob. Defaults to pairs_l{layer}_*.safetensors.",
    )
    p.add_argument("--rollout-glob", default="rollouts_*.jsonl")
    p.add_argument("--output", required=True)

    p.add_argument(
        "--whitener",
        default=None,
        help="Existing whitener. If omitted, reuse output whitener if present, otherwise fit one.",
    )
    p.add_argument(
        "--refit-whitener",
        action="store_true",
        help="Refit even if output/whitener_lXX.safetensors exists.",
    )
    p.add_argument(
        "--whitener-chunk",
        type=int,
        default=8192,
        help="Rows per GPU chunk while fitting covariance.",
    )

    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--micro-batch", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--val-batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-dir", default="/root/.cache/huggingface/hub")
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument(
        "--resume",
        default=None,
        help='Checkpoint directory, or "auto" to use output/latest.json.',
    )
    p.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Keep this many most-recent step checkpoints, plus the best checkpoint.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a fresh run to replace an existing run directory.",
    )
    p.add_argument(
        "--data-note",
        default="",
        help="Freeform provenance note written into run_config.json.",
    )

    args = p.parse_args()

    if not 0.0 < args.val_frac < 1.0:
        p.error("--val-frac must be between 0 and 1")
    if args.micro_batch < 1 or args.grad_accum < 1:
        p.error("--micro-batch and --grad-accum must be >= 1")
    if args.epochs < 1:
        p.error("--epochs must be >= 1")
    if args.eval_every < 1:
        p.error("--eval-every must be >= 1")

    return args


def shard_id(path: Path) -> int:
    m = SHARD_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse shard id from {path}")
    return int(m.group(1))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump_atomic(obj, path: Path):
    """Write JSON without relying on copy2/copystat semantics on network filesystems."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def package_version(name: str):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_for_prompt(prompt: str, split_seed: int, val_frac: float) -> str:
    payload = f"{split_seed}\0{prompt}".encode("utf-8")
    h = hashlib.blake2b(payload, digest_size=8).digest()
    u = int.from_bytes(h, "big") / 2**64
    return "val" if u < val_frac else "train"


def batch_to_device(batch, device):
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


# ----------------------------
# Corpus loading
# ----------------------------

@dataclass
class Corpus:
    activations: torch.Tensor
    span_token_ids: torch.Tensor
    span_offsets: torch.Tensor
    shard_ids: torch.Tensor
    rollout_ids: torch.Tensor
    is_train: torch.Tensor
    is_val: torch.Tensor
    d_model: int
    pair_paths: list
    rollout_paths: list

    @property
    def train_indices(self):
        return torch.where(self.is_train)[0]

    @property
    def val_indices(self):
        return torch.where(self.is_val)[0]


def load_rollout_splits(
    rollout_dir: Path,
    rollout_glob: str,
    split_seed: int,
    val_frac: float,
):
    paths = sorted(rollout_dir.glob(rollout_glob), key=shard_id)
    if not paths:
        raise FileNotFoundError(
            f"No rollout files matching {rollout_dir / rollout_glob}"
        )

    rollout_split = {}
    split_counts = {"train": 0, "val": 0}

    for path in tqdm(paths, desc="rollout splits"):
        sid = shard_id(path)
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                key = (sid, int(row["rollout_id"]))
                split = split_for_prompt(
                    row["prompt"],
                    split_seed=split_seed,
                    val_frac=val_frac,
                )
                rollout_split[key] = split
                split_counts[split] += 1

    return rollout_split, paths, split_counts


def load_corpus(args) -> Corpus:
    pair_dir = Path(args.pair_dir)
    rollout_dir = Path(args.rollout_dir)
    pair_glob = args.pair_glob or f"pairs_l{args.layer}_*.safetensors"

    rollout_split, rollout_paths, rollout_split_counts = load_rollout_splits(
        rollout_dir,
        args.rollout_glob,
        args.split_seed,
        args.val_frac,
    )

    pair_paths = sorted(pair_dir.glob(pair_glob), key=shard_id)
    if not pair_paths:
        raise FileNotFoundError(
            f"No pair files matching {pair_dir / pair_glob}"
        )

    activation_chunks = []
    token_chunks = []
    length_chunks = []
    shard_chunks = []
    rollout_chunks = []

    d_model = None

    for path in tqdm(pair_paths, desc="loading pair shards"):
        sid = shard_id(path)
        x = load_file(str(path))

        required = {
            "activations",
            "span_token_ids",
            "span_offsets",
            "rollout_id",
        }
        missing = required - set(x)
        if missing:
            raise KeyError(f"{path} missing tensors: {sorted(missing)}")

        acts = x["activations"].float()
        if acts.ndim != 2:
            raise ValueError(f"{path}: activations must be rank-2, got {acts.shape}")

        if d_model is None:
            d_model = acts.shape[1]
        elif acts.shape[1] != d_model:
            raise ValueError(
                f"{path}: d_model={acts.shape[1]} differs from prior {d_model}"
            )

        offsets = x["span_offsets"].long()
        lengths = offsets[1:] - offsets[:-1]

        if len(lengths) != len(acts):
            raise ValueError(
                f"{path}: {len(lengths)} spans != {len(acts)} activations"
            )

        activation_chunks.append(acts)
        token_chunks.append(x["span_token_ids"].int())
        length_chunks.append(lengths)
        shard_chunks.append(
            torch.full((len(acts),), sid, dtype=torch.int32)
        )
        rollout_chunks.append(x["rollout_id"].int())

    activations = torch.cat(activation_chunks)
    span_token_ids = torch.cat(token_chunks)
    span_lengths = torch.cat(length_chunks)
    shard_ids = torch.cat(shard_chunks)
    rollout_ids = torch.cat(rollout_chunks)

    span_offsets = torch.zeros(
        len(span_lengths) + 1,
        dtype=torch.int64,
    )
    span_offsets[1:] = torch.cumsum(span_lengths.long(), dim=0)

    if int(span_offsets[-1]) != len(span_token_ids):
        raise ValueError("Global ragged span offsets do not match token array length")

    try:
        is_val = torch.tensor(
            [
                rollout_split[(int(s), int(r))] == "val"
                for s, r in zip(shard_ids.tolist(), rollout_ids.tolist())
            ],
            dtype=torch.bool,
        )
    except KeyError as e:
        raise KeyError(
            f"Pair row references rollout absent from rollout sidecars: {e}"
        ) from e

    is_train = ~is_val

    print(
        f"rollouts: train={rollout_split_counts['train']:,} "
        f"val={rollout_split_counts['val']:,}"
    )
    print(
        f"pairs:    train={int(is_train.sum()):,} "
        f"val={int(is_val.sum()):,} "
        f"total={len(activations):,}"
    )
    print(
        f"pair val fraction: {is_val.float().mean().item():.3f} | "
        f"d_model={d_model}"
    )
    print(
        f"activation RAM: {activations.nbytes / 1024**3:.3f} GiB"
    )

    if not is_train.any() or not is_val.any():
        raise ValueError("Train/val split produced an empty partition")

    return Corpus(
        activations=activations,
        span_token_ids=span_token_ids,
        span_offsets=span_offsets,
        shard_ids=shard_ids,
        rollout_ids=rollout_ids,
        is_train=is_train,
        is_val=is_val,
        d_model=d_model,
        pair_paths=pair_paths,
        rollout_paths=rollout_paths,
    )


# ----------------------------
# Whitener
# ----------------------------

def fit_whitener(
    activations: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    chunk_size: int,
):
    """
    Fit exact empirical covariance in FP64 from sufficient statistics,
    chunked so whitening does not require putting the whole corpus on GPU.
    """
    d_model = activations.shape[1]
    n = len(train_indices)

    # Pass 1: train-only mean.
    sum_x = torch.zeros(d_model, dtype=torch.float64, device=device)

    for start in tqdm(
        range(0, n, chunk_size),
        desc="whitener mean",
    ):
        idx = train_indices[start : start + chunk_size]
        x = activations[idx].to(
            device=device,
            dtype=torch.float64,
            non_blocking=True,
        )
        sum_x += x.sum(dim=0)

    mean = sum_x / n
    del sum_x

    # Pass 2: centered covariance. This mirrors the notebook calculation while
    # staying memory-bounded and avoids subtracting two large second moments.
    cov = torch.zeros(
        (d_model, d_model),
        dtype=torch.float64,
        device=device,
    )

    for start in tqdm(
        range(0, n, chunk_size),
        desc="whitener covariance",
    ):
        idx = train_indices[start : start + chunk_size]
        x = activations[idx].to(
            device=device,
            dtype=torch.float64,
            non_blocking=True,
        )
        x.sub_(mean)
        cov += x.T @ x

    cov /= (n - 1)

    # Symmetrize tiny numerical asymmetry before eigh.
    cov = (cov + cov.T) * 0.5

    evals, evecs = torch.linalg.eigh(cov)

    if not torch.isfinite(evals).all():
        raise FloatingPointError("Non-finite covariance eigenvalues")
    if evals.min().item() <= 0:
        raise ValueError(
            f"Covariance is not positive definite: min eigenvalue={evals.min().item():.6g}"
        )

    inv_sqrt = evals.rsqrt()
    whitener = evecs * inv_sqrt.unsqueeze(0)

    # Algebraic sanity check.
    identity = whitener.T @ cov @ whitener
    max_diag_err = (identity.diag() - 1).abs().max().item()
    offdiag = identity - torch.diag(identity.diag())
    max_offdiag = offdiag.abs().max().item()

    stats = {
        "n_train": int(n),
        "eigen_min": evals.min().item(),
        "eigen_p01": torch.quantile(evals, 0.01).item(),
        "eigen_median": evals.median().item(),
        "eigen_mean": evals.mean().item(),
        "eigen_max": evals.max().item(),
        "condition_number": (evals.max() / evals.min()).item(),
        "max_whitened_diag_error": max_diag_err,
        "max_whitened_offdiag": max_offdiag,
    }

    print(
        "whitener spectrum | "
        f"min={stats['eigen_min']:.6g} "
        f"median={stats['eigen_median']:.6g} "
        f"max={stats['eigen_max']:.6g} "
        f"cond={stats['condition_number']:.1f}"
    )
    print(
        "identity check     | "
        f"max diag err={max_diag_err:.3e} "
        f"max offdiag={max_offdiag:.3e}"
    )

    return (
        mean.float().cpu().contiguous(),
        evals.float().cpu().contiguous(),
        evecs.float().cpu().contiguous(),
        whitener.float().cpu().contiguous(),
        stats,
    )


def prepare_whitener(args, corpus: Corpus, output: Path, device):
    output_whitener = output / f"whitener_l{args.layer}.safetensors"

    resume_dir = resolve_resume_dir(args, output, required=False)
    resume_whitener = (
        resume_dir / f"whitener_l{args.layer}.safetensors"
        if resume_dir is not None
        else None
    )

    if resume_whitener is not None and resume_whitener.exists():
        source = resume_whitener
        print(f"using checkpoint whitener: {source}")
    elif args.whitener:
        source = Path(args.whitener)
        if not source.exists():
            raise FileNotFoundError(source)
        print(f"using explicit whitener: {source}")
    elif output_whitener.exists() and not args.refit_whitener:
        source = output_whitener
        print(f"reusing output whitener: {source}")
    else:
        source = None

    if source is None:
        print("fitting train-only whitener...")
        mean, evals, evecs, whitener, stats = fit_whitener(
            corpus.activations,
            corpus.train_indices,
            device,
            args.whitener_chunk,
        )

        save_file(
            {
                "mean": mean,
                "eigenvalues": evals,
                "eigenvectors": evecs,
                "whitener": whitener,
            },
            str(output_whitener),
            metadata={
                "model_id": args.model,
                "layer": str(args.layer),
                "n_train": str(len(corpus.train_indices)),
                "split_seed": str(args.split_seed),
            },
        )

        json_dump_atomic(
            stats,
            output / "whitener_stats.json",
        )
        source = output_whitener
    else:
        # Make the run self-contained. copyfile avoids copy2/copystat failures
        # on some network-mounted /workspace volumes.
        if source.resolve() != output_whitener.resolve():
            shutil.copyfile(source, output_whitener)
            if sha256_file(source) != sha256_file(output_whitener):
                raise IOError("Whitener copy failed integrity check")
        source = output_whitener

    white = load_file(str(source))
    if set(("mean", "whitener")) - set(white):
        raise KeyError(f"{source} must contain mean and whitener")

    mean = white["mean"].float()
    whitener = white["whitener"].float()

    if mean.shape != (corpus.d_model,):
        raise ValueError(
            f"Whitener mean shape {mean.shape} != ({corpus.d_model},)"
        )
    if whitener.shape != (corpus.d_model, corpus.d_model):
        raise ValueError(
            f"Whitener shape {whitener.shape} != "
            f"({corpus.d_model}, {corpus.d_model})"
        )

    print(f"whitener sha256: {sha256_file(source)}")
    return source, mean.to(device), whitener.to(device)


# ----------------------------
# Dataset / collator
# ----------------------------

class ARDataset(Dataset):
    def __init__(
        self,
        indices,
        span_token_ids,
        span_offsets,
        activations,
        tag_ids,
    ):
        self.indices = indices.tolist()
        self.span_token_ids = span_token_ids
        self.span_offsets = span_offsets
        self.activations = activations
        self.tag = torch.tensor(tag_ids, dtype=torch.long)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = self.indices[j]
        lo = int(self.span_offsets[i])
        hi = int(self.span_offsets[i + 1])

        span_ids = self.span_token_ids[lo:hi].long()
        input_ids = torch.cat((self.tag, span_ids))

        return {
            "input_ids": input_ids,
            "target": self.activations[i],
        }


class ARCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        lengths = torch.tensor(
            [len(x["input_ids"]) for x in batch],
            dtype=torch.long,
        )
        max_len = int(lengths.max())

        ids = torch.full(
            (len(batch), max_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        mask = torch.zeros_like(ids)
        targets = torch.stack([x["target"] for x in batch]).float()

        # Right padding is deliberate: lengths-1 remains the final real token.
        for b, item in enumerate(batch):
            n = len(item["input_ids"])
            ids[b, :n] = item["input_ids"]
            mask[b, :n] = 1

        return {
            "input_ids": ids,
            "attention_mask": mask,
            "lengths": lengths,
            "targets": targets,
        }


# ----------------------------
# Model
# ----------------------------

class ActivationReconstructor(nn.Module):
    def __init__(self, peft_model, target_mean, d_model):
        super().__init__()
        self.peft_model = peft_model

        device = next(
            p.device
            for p in peft_model.parameters()
            if p.device.type != "meta"
        )

        self.norm = nn.LayerNorm(
            d_model,
            dtype=torch.float32,
            device=device,
        )
        self.head = nn.Linear(
            d_model,
            d_model,
            bias=True,
            dtype=torch.float32,
            device=device,
        )

        # Clean null baseline: predict the train-set mean exactly.
        # This intentionally means LoRA gradients are zero on the first update;
        # once the head moves away from zero, gradients flow into the backbone.
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.copy_(target_mean.to(device).float())

    def forward(self, input_ids, attention_mask, lengths):
        causal_lm = self.peft_model.get_base_model()

        # Backbone only: do not compute full vocabulary logits.
        out = causal_lm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state

        last_idx = lengths.to(hidden.device) - 1
        batch_idx = torch.arange(
            hidden.shape[0],
            device=hidden.device,
        )

        pooled = hidden[batch_idx, last_idx].float()
        pooled = self.norm(pooled)
        return self.head(pooled)


def build_ar(
    args,
    tokenizer,
    target_mean,
    d_model,
    device,
    resume_dir: Optional[Path],
):
    gpu_index = args.gpu

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map=gpu_index,
        cache_dir=args.cache_dir,
    )
    base.config.use_cache = False

    if base.config.hidden_size != d_model:
        raise ValueError(
            f"Model hidden size {base.config.hidden_size} != corpus d_model {d_model}"
        )
    if args.layer < 0 or args.layer >= len(base.model.layers):
        raise ValueError(
            f"Target layer {args.layer} outside model depth {len(base.model.layers)}"
        )

    # Single-layer AR only needs blocks 0..target inclusive.
    base.model.layers = nn.ModuleList(
        list(base.model.layers[: args.layer + 1])
    )
    base.config.num_hidden_layers = args.layer + 1
    base.model.config.num_hidden_layers = args.layer + 1

    if resume_dir is not None:
        lora_dir = resume_dir / "lora"
        if not lora_dir.exists():
            raise FileNotFoundError(f"Missing resume adapter: {lora_dir}")

        peft_model = PeftModel.from_pretrained(
            base,
            str(lora_dir),
            is_trainable=True,
        )
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base, lora_cfg)

    # Validated faster regime for short AR spans: no checkpointing.
    peft_model.gradient_checkpointing_disable()
    if hasattr(peft_model, "disable_input_require_grads"):
        peft_model.disable_input_require_grads()

    meta_params = [
        name
        for name, p in peft_model.named_parameters()
        if p.device.type == "meta"
    ]
    if meta_params:
        raise RuntimeError(
            f"Model contains meta tensors after load: {meta_params[:10]}"
        )

    ar = ActivationReconstructor(
        peft_model,
        target_mean,
        d_model,
    )

    if resume_dir is not None:
        state = torch.load(
            resume_dir / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        ar.norm.load_state_dict(state["norm"])
        ar.head.load_state_dict(state["head"])

    trainable = sum(
        p.numel()
        for p in ar.parameters()
        if p.requires_grad
    )
    total = sum(p.numel() for p in ar.parameters())

    print(
        f"AR backbone blocks: {len(base.model.layers)} | "
        f"trainable={trainable:,} / total={total:,} "
        f"({100*trainable/total:.2f}%)"
    )

    return ar


# ----------------------------
# Objective / evaluation
# ----------------------------

def ar_loss(pred, target, whitener):
    err_w = (pred - target) @ whitener
    return err_w.square().mean()


@torch.no_grad()
def evaluate_ar(
    ar,
    loader,
    target_mean,
    whitener,
    d_model,
    device,
):
    ar.eval()

    sum_err2 = 0.0
    sum_target2 = 0.0
    sum_cos = 0.0
    n_examples = 0

    for batch in loader:
        batch = batch_to_device(batch, device)

        pred = ar(
            batch["input_ids"],
            batch["attention_mask"],
            batch["lengths"],
        )
        target = batch["targets"]

        if not torch.isfinite(pred).all():
            raise FloatingPointError(
                "Non-finite AR predictions during validation. "
                "Check GPU/kernel state and padding path."
            )

        pred_w = (pred - target_mean) @ whitener
        target_w = (target - target_mean) @ whitener
        err_w = pred_w - target_w

        sum_err2 += err_w.square().sum().item()
        sum_target2 += target_w.square().sum().item()

        cos = F.cosine_similarity(pred_w, target_w, dim=-1)
        sum_cos += cos.sum().item()
        n_examples += len(target)

    mse = sum_err2 / (n_examples * d_model)
    fve = 1.0 - (sum_err2 / sum_target2)
    cosine = sum_cos / n_examples

    return {
        "mse": float(mse),
        "fve": float(fve),
        "cosine": float(cosine),
    }


# ----------------------------
# Checkpointing / resume
# ----------------------------

def resolve_resume_dir(args, output: Path, required=True):
    if args.resume is None:
        return None

    if args.resume != "auto":
        path = Path(args.resume)
        if required and not path.exists():
            raise FileNotFoundError(path)
        return path if path.exists() else None

    latest_path = output / "latest.json"
    if not latest_path.exists():
        if required:
            raise FileNotFoundError(
                f"--resume auto requested but {latest_path} does not exist"
            )
        return None

    with latest_path.open() as f:
        latest = json.load(f)

    path = Path(latest["checkpoint"])
    if required and not path.exists():
        raise FileNotFoundError(path)
    return path if path.exists() else None


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"])
    torch.cuda.set_rng_state_all(rng["torch_cuda"])


def write_metrics_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_checkpoint(
    *,
    args,
    output: Path,
    ar,
    optimizer,
    scheduler,
    whitener_path: Path,
    run_config_path: Path,
    baseline,
    history,
    global_step: int,
    epoch: int,
    next_micro_step: int,
    best_fve: float,
    best_step: Optional[int],
):
    ckpt_root = output / "checkpoints"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_root / f"step_{global_step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)

    ar.peft_model.save_pretrained(ckpt / "lora")

    torch.save(
        {
            "norm": ar.norm.state_dict(),
            "head": ar.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "next_micro_step": next_micro_step,
            "history": history,
            "baseline": baseline,
            "best_fve": best_fve,
            "best_step": best_step,
            "rng": capture_rng_state(),
        },
        ckpt / "training_state.pt",
    )

    # Self-contained checkpoint. Use copyfile, not copy2, because some
    # persistent network volumes reject timestamp/permission preservation.
    dst_white = ckpt / f"whitener_l{args.layer}.safetensors"
    shutil.copyfile(whitener_path, dst_white)
    if sha256_file(whitener_path) != sha256_file(dst_white):
        raise IOError("Checkpoint whitener copy failed integrity check")

    shutil.copyfile(run_config_path, ckpt / "run_config.json")

    json_dump_atomic(
        {
            "baseline": baseline,
            "history": history,
            "final": history[-1] if history else baseline,
        },
        ckpt / "metrics.json",
    )

    # Marker is written last: presence means checkpoint completed.
    (ckpt / "COMPLETE").write_text("ok\n", encoding="utf-8")

    json_dump_atomic(
        {
            "checkpoint": str(ckpt),
            "global_step": global_step,
        },
        output / "latest.json",
    )

    if best_step is not None:
        json_dump_atomic(
            {
                "checkpoint": str(
                    ckpt_root / f"step_{best_step:06d}"
                ),
                "global_step": best_step,
                "fve": best_fve,
            },
            output / "best.json",
        )

    prune_checkpoints(
        ckpt_root,
        keep=args.keep_checkpoints,
        best_step=best_step,
    )

    return ckpt


def prune_checkpoints(
    ckpt_root: Path,
    keep: int,
    best_step: Optional[int],
):
    if keep < 1:
        return

    dirs = sorted(
        [
            p
            for p in ckpt_root.glob("step_*")
            if p.is_dir() and (p / "COMPLETE").exists()
        ],
        key=lambda p: int(p.name.split("_")[-1]),
    )

    keep_paths = set(dirs[-keep:])
    if best_step is not None:
        keep_paths.add(ckpt_root / f"step_{best_step:06d}")

    for p in dirs:
        if p not in keep_paths:
            shutil.rmtree(p)


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# ----------------------------
# Run metadata
# ----------------------------

def write_run_metadata(args, corpus, output, whitener_path):
    config = {
        "model_id": args.model,
        "target_layer": args.layer,
        "d_model": corpus.d_model,
        "pair_glob": args.pair_glob
        or f"pairs_l{args.layer}_*.safetensors",
        "rollout_glob": args.rollout_glob,
        "pair_dir": str(Path(args.pair_dir)),
        "rollout_dir": str(Path(args.rollout_dir)),
        "pair_shards": [p.name for p in corpus.pair_paths],
        "rollout_shards": [p.name for p in corpus.rollout_paths],
        "train_pairs": int(corpus.is_train.sum()),
        "val_pairs": int(corpus.is_val.sum()),
        "val_frac": args.val_frac,
        "split_seed": args.split_seed,
        "seed": args.seed,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "effective_batch": args.micro_batch * args.grad_accum,
        "val_batch": args.val_batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "eval_every": args.eval_every,
        "max_grad_norm": args.max_grad_norm,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": "all-linear",
        "layer_tag": f"[Layer {args.layer:02d}]",
        "whitener_path": str(whitener_path),
        "whitener_sha256": sha256_file(whitener_path),
        "data_note": args.data_note,
    }

    path = output / "run_config.json"
    json_dump_atomic(config, path)

    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.gpu),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "safetensors": package_version("safetensors"),
        "accelerate": package_version("accelerate"),
    }
    json_dump_atomic(env, output / "environment.json")

    return path


# ----------------------------
# Main training
# ----------------------------

def main():
    args = parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output = Path(args.output)

    resume_dir = None
    if args.resume is not None:
        output.mkdir(parents=True, exist_ok=True)
        resume_dir = resolve_resume_dir(args, output, required=True)
        if not (resume_dir / "COMPLETE").exists():
            raise RuntimeError(f"Resume checkpoint is incomplete: {resume_dir}")
        print(f"resuming from: {resume_dir}")
    else:
        if output.exists() and (output / "run_config.json").exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"{output} already contains a run. "
                    "Use --resume auto, choose a new --output, or pass --overwrite."
                )
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)

    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"model: {args.model}")
    print(f"target layer: {args.layer}")

    # Load data before model so whitener fitting has maximal VRAM.
    corpus = load_corpus(args)

    whitener_path, target_mean, whitener = prepare_whitener(
        args,
        corpus,
        output,
        device,
    )

    # Tokenizer / exact layer-tag contract.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        cache_dir=args.cache_dir,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    layer_tag = f"[Layer {args.layer:02d}]"
    tag_ids = tokenizer(
        layer_tag,
        add_special_tokens=False,
    ).input_ids

    print(
        f"layer tag: {layer_tag!r} | "
        f"{len(tag_ids)} tokens | "
        f"{tokenizer.convert_ids_to_tokens(tag_ids)}"
    )

    train_ds = ARDataset(
        corpus.train_indices,
        corpus.span_token_ids,
        corpus.span_offsets,
        corpus.activations,
        tag_ids,
    )
    val_ds = ARDataset(
        corpus.val_indices,
        corpus.span_token_ids,
        corpus.span_offsets,
        corpus.activations,
        tag_ids,
    )
    collator = ARCollator(tokenizer.pad_token_id)

    run_config_path = write_run_metadata(
        args,
        corpus,
        output,
        whitener_path,
    )

    # Free any temporary CUDA allocations before loading the model.
    gc.collect()
    torch.cuda.empty_cache()

    ar = build_ar(
        args,
        tokenizer,
        target_mean,
        corpus.d_model,
        device,
        resume_dir,
    )

    def make_train_loader(epoch):
        g = torch.Generator()
        g.manual_seed(args.seed + epoch)
        return DataLoader(
            train_ds,
            batch_size=args.micro_batch,
            shuffle=True,
            generator=g,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    # Loader length is epoch-independent.
    probe_loader = make_train_loader(0)
    microbatches_per_epoch = len(probe_loader)
    steps_per_epoch = math.ceil(
        microbatches_per_epoch / args.grad_accum
    )
    total_steps = steps_per_epoch * args.epochs

    optimizer = torch.optim.AdamW(
        [p for p in ar.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            1.0,
            (step + 1) / args.warmup_steps,
        )
        if args.warmup_steps > 0
        else 1.0,
    )

    history = []
    global_step = 0
    start_epoch = 0
    start_micro_step = 0
    best_fve = float("-inf")
    best_step = None

    if resume_dir is not None:
        state = torch.load(
            resume_dir / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        optimizer.load_state_dict(state["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler"])

        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        start_micro_step = int(state["next_micro_step"])
        history = list(state.get("history", []))
        best_fve = float(state.get("best_fve", float("-inf")))
        best_step = state.get("best_step")

        restore_rng_state(state["rng"])

        # A checkpoint at end-of-epoch resumes at the next epoch.
        if start_micro_step >= microbatches_per_epoch:
            start_epoch += 1
            start_micro_step = 0

        baseline = state.get("baseline")
        if baseline is None:
            baseline = evaluate_ar(
                ar,
                val_loader,
                target_mean,
                whitener,
                corpus.d_model,
                device,
            )

        print(
            f"resume state: epoch={start_epoch} "
            f"next_micro={start_micro_step} step={global_step}"
        )
    else:
        baseline = evaluate_ar(
            ar,
            val_loader,
            target_mean,
            whitener,
            corpus.d_model,
            device,
        )
        print(
            "baseline | "
            f"mse={baseline['mse']:.6f} "
            f"FVE={baseline['fve']:.3%} "
            f"cos={baseline['cosine']:.4f}"
        )

        write_metrics_jsonl(
            output / "metrics.jsonl",
            {
                "kind": "baseline",
                "step": 0,
                **baseline,
            },
        )

    if start_epoch >= args.epochs:
        print(
            f"Checkpoint is already at/past requested --epochs={args.epochs}; nothing to do."
        )
        return

    print(
        f"microbatches/epoch={microbatches_per_epoch:,} | "
        f"optimizer steps/epoch={steps_per_epoch:,} | "
        f"requested total steps={total_steps:,}"
    )

    for epoch in range(start_epoch, args.epochs):
        train_loader = make_train_loader(epoch)
        skip_before = start_micro_step if epoch == start_epoch else 0

        ar.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        running_microbatches = 0

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"epoch {epoch + 1}/{args.epochs}",
        )

        for micro_step, batch in pbar:
            if micro_step < skip_before:
                continue

            batch = batch_to_device(batch, device)

            pred = ar(
                batch["input_ids"],
                batch["attention_mask"],
                batch["lengths"],
            )
            loss = ar_loss(
                pred,
                batch["targets"],
                whitener,
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite train loss at epoch={epoch} micro_step={micro_step}"
                )

            # Correct scaling even for the final partial accumulation group.
            group_start = (
                micro_step // args.grad_accum
            ) * args.grad_accum
            group_end = min(
                group_start + args.grad_accum,
                len(train_loader),
            )
            group_size = group_end - group_start

            (loss / group_size).backward()

            running_loss += loss.item()
            running_microbatches += 1

            do_step = (micro_step + 1) == group_end
            if not do_step:
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in ar.parameters() if p.requires_grad],
                args.max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            train_loss = running_loss / running_microbatches
            running_loss = 0.0
            running_microbatches = 0

            pbar.set_postfix(
                {
                    "step": global_step,
                    "loss": f"{train_loss:.3f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                }
            )

            should_eval = (
                global_step % args.eval_every == 0
                or micro_step + 1 == len(train_loader)
            )

            if should_eval:
                metrics = evaluate_ar(
                    ar,
                    val_loader,
                    target_mean,
                    whitener,
                    corpus.d_model,
                    device,
                )

                record = {
                    "kind": "eval",
                    "epoch": epoch,
                    "step": global_step,
                    "next_micro_step": micro_step + 1,
                    "train_loss": float(train_loss),
                    "grad_norm": float(grad_norm),
                    "lr": float(scheduler.get_last_lr()[0]),
                    **metrics,
                }
                history.append(record)
                write_metrics_jsonl(
                    output / "metrics.jsonl",
                    record,
                )

                if metrics["fve"] > best_fve:
                    best_fve = metrics["fve"]
                    best_step = global_step

                print(
                    f"\nstep {global_step:5d} | "
                    f"train {train_loss:.4f} | "
                    f"val mse {metrics['mse']:.4f} | "
                    f"FVE {metrics['fve']:.3%} | "
                    f"cos {metrics['cosine']:.4f}"
                )

                ckpt = save_checkpoint(
                    args=args,
                    output=output,
                    ar=ar,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    whitener_path=whitener_path,
                    run_config_path=run_config_path,
                    baseline=baseline,
                    history=history,
                    global_step=global_step,
                    epoch=epoch,
                    next_micro_step=micro_step + 1,
                    best_fve=best_fve,
                    best_step=best_step,
                )
                print(f"checkpoint: {ckpt}")
                ar.train()

        # Any resumed skip applies only to first resumed epoch.
        start_micro_step = 0

    final = history[-1] if history else baseline

    json_dump_atomic(
        {
            "baseline": baseline,
            "history": history,
            "final": final,
            "best_fve": best_fve,
            "best_step": best_step,
        },
        output / "metrics.json",
    )

    print("\ntraining complete")
    print(
        f"final | mse={final['mse']:.6f} "
        f"FVE={final['fve']:.3%} "
        f"cos={final['cosine']:.4f}"
    )
    print(
        f"best  | step={best_step} "
        f"FVE={best_fve:.3%}"
    )


if __name__ == "__main__":
    main()
