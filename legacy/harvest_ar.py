#!/usr/bin/env python3
"""
Harvest Oracle-Lens activation-reconstructor training pairs from Qwen on-policy rollouts.

This is the reusable V3 version of the validated 01_harvest notebook pipeline.

Data contract
-------------
For each assistant rollout:
  * Generate on-policy assistant text from a WildChat user prompt.
  * Choose up to K mutually-disjoint following-token spans.
  * Span lengths are iid Uniform{span_min, ..., span_max}.
  * K is chosen conservatively so those lengths never need length-dependent
    rejection; this preserves the intended uniform length distribution.
  * For each span beginning at assistant token t, save the post-block-L residual
    at assistant token t-1.
  * The exact same span layout is used for every requested layer.

Output layout
-------------
<output>/
  run_config.json
  environment.json
  progress.json
  pairs/
    pairs_l22_00000.safetensors
    pairs_l26_00000.safetensors
    pairs_l30_00000.safetensors
    ...
  rollouts/
    rollouts_00000.jsonl
    ...
  manifests/
    shard_00000.json
    ...

A shard manifest is written LAST and is the authoritative completeness marker.
On restart, unmarked partial shard files are deleted and regenerated.

The pair shard format is directly compatible with train_ar.py.

Validated A40 defaults
----------------------
  max_new_tokens = 1536
  generation batch = 16
  replay batch = 1
  span cap = 35
  span length = 1..32

Example
-------
Single layer:

    python harvest_ar.py \
      --layers 30 \
      --target-pairs 200000 \
      --output /workspace/o-lens-qwen35-4b/data/olens_200k

Several layers from the same rollouts/spans:

    python harvest_ar.py \
      --layers 22 26 30 \
      --target-pairs 200000 \
      --output /workspace/o-lens-qwen35-4b/data/olens_200k_l22_l26_l30

Then train one layer with:

    python train_ar.py \
      --layer 30 \
      --pair-dir <output>/pairs \
      --rollout-dir <output>/rollouts \
      --output /workspace/o-lens-qwen35-4b/ar/l30_200k \
      --epochs 2
"""

import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface/hub"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .harvest_manifests import scan_manifests, cleanup_unmarked_files
except ImportError:
    from harvest_manifests import scan_manifests, cleanup_unmarked_files


PAIR_RE = re.compile(r"pairs_l(?P<layer>\d+)_(?P<shard>\d{5})\.safetensors$")
ROLLOUT_RE = re.compile(r"rollouts_(?P<shard>\d{5})\.jsonl$")
MANIFEST_RE = re.compile(r"shard_(?P<shard>\d{5})\.json$")


def parse_args():
    p = argparse.ArgumentParser(
        description="Harvest Oracle-Lens AR training pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--layers", nargs="+", type=int, required=True)
    p.add_argument("--output", required=True)

    p.add_argument("--target-pairs", type=int, default=200_000)
    p.add_argument("--shard-pairs", type=int, default=2_048)

    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument("--gen-batch", type=int, default=16)
    p.add_argument("--replay-batch", type=int, default=1)

    p.add_argument("--max-spans", type=int, default=35)
    p.add_argument("--span-min", type=int, default=1)
    p.add_argument("--span-max", type=int, default=32)

    p.add_argument("--dataset", default="allenai/WildChat-1M")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--shuffle-buffer", type=int, default=10_000)
    p.add_argument("--min-prompt-tokens", type=int, default=4)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--min-response-tokens", type=int, default=8)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cache-dir", default="/root/.cache/huggingface/hub")

    p.add_argument("--exclude-rollouts", type=Path, action="append", default=[],
                   help="Reject prompts from these existing rollout directories; hash-bound on resume")
    p.add_argument("--dry-run", action="store_true", help="Validate exclusions/config without GPU or writes")
    p.add_argument("--revision", help="Pinned model/tokenizer revision")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--data-note", default="")

    args = p.parse_args()
    args.layers = sorted(set(args.layers))

    if not args.layers:
        p.error("--layers must contain at least one layer")
    if args.target_pairs < 1:
        p.error("--target-pairs must be >= 1")
    if args.shard_pairs < 1:
        p.error("--shard-pairs must be >= 1")
    if args.gen_batch < 1 or args.replay_batch < 1:
        p.error("--gen-batch and --replay-batch must be >= 1")
    if args.max_new_tokens < 1:
        p.error("--max-new-tokens must be >= 1")
    if args.max_spans < 1:
        p.error("--max-spans must be >= 1")
    if not (1 <= args.span_min <= args.span_max):
        p.error("require 1 <= --span-min <= --span-max")
    if args.min_prompt_tokens < 1:
        p.error("--min-prompt-tokens must be >= 1")
    if args.max_prompt_tokens < args.min_prompt_tokens:
        p.error("--max-prompt-tokens must be >= --min-prompt-tokens")
    if args.min_response_tokens < 1:
        p.error("--min-response-tokens must be >= 1")
    if args.temperature <= 0:
        p.error("--temperature must be > 0")
    if not (0 < args.top_p <= 1):
        p.error("--top-p must be in (0, 1]")

    return args


def json_dump_atomic(obj, path: Path):
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


def prompt_key(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def derived_seed(seed: int, *parts: int) -> int:
    payload = ":".join(str(x) for x in (seed,) + parts).encode("ascii")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "big",
    ) % (2**31 - 1)


def count_pairs(path: Path) -> int:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return int(f.get_slice("activations").get_shape()[0])


def run_contract(args):
    contract = {
        "format_version": 3,
        "span_sampler": "v3_uniform_disjoint_guaranteed_fit",
        "model_id": args.model,
        "layers": args.layers,
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "shuffle_buffer": args.shuffle_buffer,
        "min_prompt_tokens": args.min_prompt_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "min_response_tokens": args.min_response_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_spans": args.max_spans,
        "span_min": args.span_min,
        "span_max": args.span_max,
        "seed": args.seed,
    }

    if getattr(args, "exclusion_hashes", None):
        contract["exclude_rollout_sha256"] = args.exclusion_hashes
        contract["prompt_exclusion_policy"] = "whitespace-normalized-sha256-v1"
    if getattr(args, "revision", None):
        contract["revision"] = args.revision
    return contract


def prepare_output(args):
    root = Path(args.output)

    if args.overwrite and root.exists():
        print(f"OVERWRITE: removing {root}")
        shutil.rmtree(root)

    pair_dir = root / "pairs"
    rollout_dir = root / "rollouts"
    manifest_dir = root / "manifests"

    for p in (root, pair_dir, rollout_dir, manifest_dir):
        p.mkdir(parents=True, exist_ok=True)

    config_path = root / "run_config.json"
    contract = run_contract(args)

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing_contract = existing.get("contract")
        if existing_contract != contract:
            raise RuntimeError(
                "Existing run_config.json has a different data contract.\n"
                f"Existing: {json.dumps(existing_contract, indent=2)}\n"
                f"Requested: {json.dumps(contract, indent=2)}\n"
                "Use a new --output or --overwrite."
            )
        print(f"resuming compatible run: {root}")
    else:
        config = {
            "contract": contract,
            "requested_target_pairs": args.target_pairs,
            "operational_defaults": {
                "gen_batch": args.gen_batch,
                "replay_batch": args.replay_batch,
                "shard_pairs": args.shard_pairs,
                "gpu": args.gpu,
                "cache_dir": args.cache_dir,
            },
            "data_note": args.data_note,
        }
        json_dump_atomic(config, config_path)

    env = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "safetensors": package_version("safetensors"),
        "accelerate": package_version("accelerate"),
    }
    if not (root / "environment.json").exists():
        json_dump_atomic(env, root / "environment.json")

    return root, pair_dir, rollout_dir, manifest_dir




def validate_completed_shards(pair_dir, rollout_dir, manifest_dir, layers):
    manifests = scan_manifests(manifest_dir)

    total_pairs = 0
    total_rollouts = 0
    total_generated_tokens = 0
    span_hist = Counter()

    for sid, manifest in sorted(manifests.items()):
        if manifest.get("layers") != layers:
            raise RuntimeError(
                f"Shard {sid:05d} layer set differs from requested layers"
            )

        rollout_path = rollout_dir / f"rollouts_{sid:05d}.jsonl"
        if not rollout_path.exists():
            raise FileNotFoundError(rollout_path)

        expected_pairs = int(manifest["pairs"])

        for layer in layers:
            path = pair_dir / f"pairs_l{layer}_{sid:05d}.safetensors"
            if not path.exists():
                raise FileNotFoundError(path)
            observed = count_pairs(path)
            if observed != expected_pairs:
                raise RuntimeError(
                    f"{path}: manifest says {expected_pairs} pairs, "
                    f"file contains {observed}"
                )

        total_pairs += expected_pairs
        total_rollouts += int(manifest["rollouts"])
        total_generated_tokens += int(manifest["generated_tokens"])

        for k, v in manifest.get("span_length_hist", {}).items():
            span_hist[int(k)] += int(v)

    return manifests, {
        "pairs": total_pairs,
        "rollouts": total_rollouts,
        "generated_tokens": total_generated_tokens,
        "span_length_hist": span_hist,
    }


def load_used_prompt_keys(rollout_dir: Path, manifests):
    used = set()

    for sid in sorted(manifests):
        path = rollout_dir / f"rollouts_{sid:05d}.jsonl"
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                used.add(prompt_key(row["prompt"]))

    return used


def write_progress(root: Path, totals, args, manifests):
    hist = totals["span_length_hist"]
    n_lengths = sum(hist.values())
    mean_len = (
        sum(k * v for k, v in hist.items()) / n_lengths
        if n_lengths
        else 0.0
    )

    progress = {
        "complete_shards": len(manifests),
        "pairs": totals["pairs"],
        "target_pairs": args.target_pairs,
        "rollouts": totals["rollouts"],
        "generated_tokens": totals["generated_tokens"],
        "mean_span_length": mean_len,
        "span_length_hist": {
            str(i): int(hist.get(i, 0))
            for i in range(args.span_min, args.span_max + 1)
        },
    }
    json_dump_atomic(progress, root / "progress.json")


def extract_user_prompt(row):
    conversation = row.get("conversation") or []

    for msg in conversation:
        if msg.get("role") == "user":
            text = msg.get("content", "").strip()
            if text:
                return text

    return None


def prompt_stream(args, tokenizer, used_keys):
    from datasets import load_dataset
    try:
        from .harvest_exclusions import prompt_digest
    except ImportError:
        from harvest_exclusions import prompt_digest
    normalized_seen = set(getattr(args, "excluded_prompt_digests", ()))
    dataset = load_dataset(
        args.dataset,
        split=args.dataset_split,
        streaming=True,
    )
    dataset = dataset.shuffle(
        seed=args.seed,
        buffer_size=args.shuffle_buffer,
    )

    seen = set()

    for row in dataset:
        prompt = extract_user_prompt(row)
        if prompt is None:
            continue

        key = prompt_key(prompt)

        normalized = prompt_digest(prompt)
        if key in seen or key in used_keys or normalized in normalized_seen:
            continue
        seen.add(key)
        if getattr(args, "exclusion_hashes", None):
            normalized_seen.add(normalized)

        ids = tokenizer(
            prompt,
            add_special_tokens=False,
        ).input_ids

        if not (
            args.min_prompt_tokens
            <= len(ids)
            <= args.max_prompt_tokens
        ):
            continue

        yield prompt


@torch.inference_mode()
def generate_assistant_batch(
    *,
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    min_response_tokens,
    temperature,
    top_p,
    generation_seed,
):
    rendered = []

    for prompt in prompts:
        rendered.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    enc = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(model.device)

    input_width = enc.input_ids.shape[1]

    torch.manual_seed(generation_seed)
    torch.cuda.manual_seed_all(generation_seed)

    generated = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    special_ids = set(tokenizer.all_special_ids)
    results = []

    for i, prompt in enumerate(prompts):
        assistant_ids = generated[i, input_width:].detach().cpu()

        while (
            len(assistant_ids) > 0
            and int(assistant_ids[-1]) in special_ids
        ):
            assistant_ids = assistant_ids[:-1]

        if len(assistant_ids) < min_response_tokens:
            continue

        results.append(
            {
                "prompt": prompt,
                "prompt_text": rendered[i],
                "assistant_ids": assistant_ids,
            }
        )

    return results


def sample_disjoint_spans_v3(
    *,
    assistant_len,
    rng: random.Random,
    max_spans,
    min_span_len,
    max_span_len,
):
    available = assistant_len - 1

    if available < max_span_len:
        return []

    k = min(
        max_spans,
        available // max_span_len,
    )

    if k <= 0:
        return []

    lengths = [
        rng.randint(min_span_len, max_span_len)
        for _ in range(k)
    ]

    total_span_tokens = sum(lengths)
    if total_span_tokens > available:
        raise AssertionError("V3 guaranteed-fit invariant violated")

    leftover = available - total_span_tokens

    gaps = [0] * (k + 1)
    for _ in range(leftover):
        gaps[rng.randrange(k + 1)] += 1

    samples = []
    cursor = 1 + gaps[0]

    for i, span_len in enumerate(lengths):
        start = cursor
        end = start + span_len

        samples.append(
            {
                "start": start,
                "span_len": span_len,
            }
        )

        cursor = end + gaps[i + 1]

    if cursor > assistant_len:
        raise AssertionError("V3 packed spans exceeded assistant length")

    for a, b in zip(samples, samples[1:]):
        if a["start"] + a["span_len"] > b["start"]:
            raise AssertionError("V3 produced overlapping spans")

    starts = [s["start"] for s in samples]
    if len(starts) != len(set(starts)):
        raise AssertionError("V3 produced repeated target positions")

    return samples


def build_replay_batch(model, tokenizer, examples):
    sequences = []
    metadata = []

    for ex in examples:
        prompt_ids = torch.tensor(
            tokenizer(
                ex["prompt_text"],
                add_special_tokens=False,
            ).input_ids,
            dtype=torch.long,
        )

        assistant_ids = ex["assistant_ids"].long()
        full_ids = torch.cat((prompt_ids, assistant_ids))

        sequences.append(full_ids)
        metadata.append(
            {
                "prompt_len": len(prompt_ids),
                "assistant_len": len(assistant_ids),
                "rollout_id": int(ex["rollout_id"]),
            }
        )

    max_len = max(len(x) for x in sequences)
    batch_size = len(sequences)

    input_ids = torch.full(
        (batch_size, max_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )

    for i, seq in enumerate(sequences):
        n = len(seq)
        input_ids[i, :n] = seq
        attention_mask[i, :n] = 1

    return (
        input_ids.to(model.device),
        attention_mask.to(model.device),
        metadata,
    )


def make_layer_hook(captured, layer):
    def hook(module, inputs_, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer] = hidden
    return hook


@torch.inference_mode()
def replay_and_harvest_batch(
    *,
    model,
    tokenizer,
    examples,
    layers,
    args,
    shard_id,
):
    input_ids, attention_mask, metadata = build_replay_batch(
        model,
        tokenizer,
        examples,
    )

    captured = {}
    handles = []

    for layer in layers:
        handles.append(
            model.model.layers[layer].register_forward_hook(
                make_layer_hook(captured, layer)
            )
        )

    try:
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing = set(layers) - set(captured)
    if missing:
        raise RuntimeError(f"Hooks did not capture layers: {sorted(missing)}")

    special_ids = set(tokenizer.all_special_ids)
    rows = []

    for b, meta in enumerate(metadata):
        prompt_len = meta["prompt_len"]
        assistant_len = meta["assistant_len"]
        rollout_id = meta["rollout_id"]

        rng = random.Random(
            derived_seed(
                args.seed,
                shard_id,
                rollout_id,
                0x5350414E,
            )
        )

        specs = sample_disjoint_spans_v3(
            assistant_len=assistant_len,
            rng=rng,
            max_spans=args.max_spans,
            min_span_len=args.span_min,
            max_span_len=args.span_max,
        )

        samples = []

        for spec in specs:
            start = spec["start"]
            span_len = spec["span_len"]

            global_start = prompt_len + start
            global_end = global_start + span_len
            target_pos = global_start - 1

            if target_pos + 1 != global_start:
                raise AssertionError("preceding-activation contract failed")

            span_ids = input_ids[
                b,
                global_start:global_end,
            ].detach().cpu()

            preceding_token_id = int(input_ids[b, target_pos])

            if preceding_token_id in special_ids:
                continue

            if any(int(t) in special_ids for t in span_ids):
                continue

            samples.append(
                {
                    "span_ids": span_ids,
                    "target_pos": target_pos,
                    "preceding_token_id": preceding_token_id,
                    "assistant_span_start": start,
                }
            )

        if not samples:
            continue

        target_positions = torch.tensor(
            [s["target_pos"] for s in samples],
            device=input_ids.device,
        )

        layer_activations = {}
        for layer in layers:
            layer_activations[layer] = (
                captured[layer][b, target_positions]
                .float()
                .cpu()
            )

        for j, sample in enumerate(samples):
            rows.append(
                {
                    "rollout_id": rollout_id,
                    "span_ids": sample["span_ids"],
                    "target_position": sample["target_pos"],
                    "assistant_span_start": sample["assistant_span_start"],
                    "preceding_token_id": sample["preceding_token_id"],
                    "activations": {
                        layer: layer_activations[layer][j]
                        for layer in layers
                    },
                }
            )

    del captured
    return rows


def pack_rows_for_layer(rows, layer):
    if not rows:
        raise ValueError("Cannot pack an empty row list")

    activations = torch.stack(
        [r["activations"][layer] for r in rows]
    ).float().contiguous()

    flat_tokens = []
    offsets = [0]

    for row in rows:
        ids = row["span_ids"].to(torch.int32)
        flat_tokens.append(ids)
        offsets.append(offsets[-1] + len(ids))

    span_token_ids = torch.cat(flat_tokens).contiguous()

    return {
        "activations": activations,
        "span_token_ids": span_token_ids,
        "span_offsets": torch.tensor(
            offsets,
            dtype=torch.int64,
        ).contiguous(),
        "rollout_id": torch.tensor(
            [r["rollout_id"] for r in rows],
            dtype=torch.int32,
        ).contiguous(),
        "target_position": torch.tensor(
            [r["target_position"] for r in rows],
            dtype=torch.int32,
        ).contiguous(),
        "assistant_span_start": torch.tensor(
            [r["assistant_span_start"] for r in rows],
            dtype=torch.int32,
        ).contiguous(),
        "preceding_token_id": torch.tensor(
            [r["preceding_token_id"] for r in rows],
            dtype=torch.int32,
        ).contiguous(),
    }


def validate_rows(rows, rollouts, layers, hidden_size):
    if not rows:
        raise ValueError("Shard produced zero rows")

    rollout_map = {
        int(ex["rollout_id"]): ex
        for ex in rollouts
    }

    by_rollout = defaultdict(list)

    for row in rows:
        if row["rollout_id"] not in rollout_map:
            raise AssertionError("row references absent rollout")

        n = len(row["span_ids"])
        if n < 1:
            raise AssertionError("empty span")

        for layer in layers:
            a = row["activations"][layer]
            if a.shape != (hidden_size,):
                raise AssertionError(
                    f"layer {layer} activation shape {a.shape} != ({hidden_size},)"
                )
            if not torch.isfinite(a).all():
                raise FloatingPointError(
                    f"non-finite activation at layer {layer}"
                )

        start = int(row["assistant_span_start"])
        by_rollout[row["rollout_id"]].append(
            (start, start + n, int(row["target_position"]))
        )

    for rollout_id, intervals in by_rollout.items():
        ordered = sorted(intervals)

        for (a0, a1, _), (b0, b1, _) in zip(
            ordered,
            ordered[1:],
        ):
            if a1 > b0:
                raise AssertionError(
                    f"overlap in rollout {rollout_id}: "
                    f"{(a0, a1)} vs {(b0, b1)}"
                )

        targets = [x[2] for x in intervals]
        if len(targets) != len(set(targets)):
            raise AssertionError(
                f"duplicate target positions in rollout {rollout_id}"
            )


def write_shard_atomic(
    *,
    args,
    shard_id,
    rows,
    rollouts,
    pair_dir,
    rollout_dir,
    manifest_dir,
    model,
    stats,
):
    validate_rows(
        rows,
        rollouts,
        args.layers,
        model.config.hidden_size,
    )

    pair_tmps = {}
    pair_finals = {}

    metadata_common = {
        "model_id": args.model,
        "source_dataset": args.dataset,
        "format_version": "3",
        "span_sampler": "v3_uniform_disjoint_guaranteed_fit",
        "hidden_size": str(model.config.hidden_size),
        "span_min": str(args.span_min),
        "span_max": str(args.span_max),
        "max_spans_per_response": str(args.max_spans),
        "max_new_tokens": str(args.max_new_tokens),
        "seed": str(args.seed),
        "rollout_id_scope": "shard_local",
        "captured_layers": ",".join(map(str, args.layers)),
    }

    for layer in args.layers:
        final = pair_dir / f"pairs_l{layer}_{shard_id:05d}.safetensors"
        tmp = final.with_name(final.name + ".tmp")

        packed = pack_rows_for_layer(rows, layer)

        save_file(
            packed,
            str(tmp),
            metadata={
                **metadata_common,
                "target_layer": str(layer),
            },
        )

        pair_tmps[layer] = tmp
        pair_finals[layer] = final

    rollout_final = rollout_dir / f"rollouts_{shard_id:05d}.jsonl"
    rollout_tmp = rollout_final.with_name(rollout_final.name + ".tmp")

    with rollout_tmp.open("w", encoding="utf-8") as f:
        for ex in rollouts:
            record = {
                "rollout_id": int(ex["rollout_id"]),
                "prompt": ex["prompt"],
                "prompt_text": ex["prompt_text"],
                "assistant_ids": ex["assistant_ids"].tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    span_hist = Counter(len(r["span_ids"]) for r in rows)

    manifest = {
        "shard_id": shard_id,
        "layers": args.layers,
        "pairs": len(rows),
        "rollouts": len(rollouts),
        "generated_tokens": int(stats["generated_tokens"]),
        "generation_seconds": float(stats["generation_seconds"]),
        "replay_seconds": float(stats["replay_seconds"]),
        "span_length_hist": {
            str(i): int(span_hist.get(i, 0))
            for i in range(args.span_min, args.span_max + 1)
        },
        "pair_files": {
            str(layer): pair_finals[layer].name
            for layer in args.layers
        },
        "rollout_file": rollout_final.name,
    }

    manifest_final = manifest_dir / f"shard_{shard_id:05d}.json"
    manifest_tmp = manifest_final.with_name(manifest_final.name + ".tmp")
    with manifest_tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    for layer in args.layers:
        os.replace(pair_tmps[layer], pair_finals[layer])

    os.replace(rollout_tmp, rollout_final)
    os.replace(manifest_tmp, manifest_final)

    return manifest


def produce_one_shard(
    *,
    args,
    shard_id,
    prompts,
    model,
    tokenizer,
):
    rollouts = []
    rows = []

    generation_seconds = 0.0
    replay_seconds = 0.0
    generated_tokens = 0
    generation_batch_idx = 0

    while len(rows) < args.shard_pairs:
        prompt_batch = []

        for _ in range(args.gen_batch):
            try:
                prompt_batch.append(next(prompts))
            except StopIteration:
                break

        if not prompt_batch:
            break

        gen_seed = derived_seed(
            args.seed,
            shard_id,
            generation_batch_idx,
            0x47454E,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        examples = generate_assistant_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompt_batch,
            max_new_tokens=args.max_new_tokens,
            min_response_tokens=args.min_response_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            generation_seed=gen_seed,
        )

        torch.cuda.synchronize()
        generation_seconds += time.perf_counter() - t0
        generated_tokens += sum(len(ex["assistant_ids"]) for ex in examples)

        rollout_base = len(rollouts)
        for j, ex in enumerate(examples):
            ex["rollout_id"] = rollout_base + j

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        for i in range(0, len(examples), args.replay_batch):
            rows.extend(
                replay_and_harvest_batch(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples[i : i + args.replay_batch],
                    layers=args.layers,
                    args=args,
                    shard_id=shard_id,
                )
            )

        torch.cuda.synchronize()
        replay_seconds += time.perf_counter() - t1

        rollouts.extend(examples)
        generation_batch_idx += 1

        if not examples:
            continue

    if not rows:
        raise RuntimeError(
            "Prompt source exhausted or generation produced no harvestable rows"
        )

    return rows, rollouts, {
        "generation_seconds": generation_seconds,
        "replay_seconds": replay_seconds,
        "generated_tokens": generated_tokens,
        "generation_batches": generation_batch_idx,
    }


def main():
    args = parse_args()

    try:
        from .harvest_exclusions import rollout_inventory
    except ImportError:
        from harvest_exclusions import rollout_inventory
    args.excluded_prompt_digests = set()
    args.exclusion_hashes = {}
    for directory in args.exclude_rollouts:
        keys, hashes, _ = rollout_inventory(directory)
        args.excluded_prompt_digests.update(keys)
        args.exclusion_hashes.update(hashes)
    if args.dry_run:
        print(json.dumps({"contract":run_contract(args),
            "excluded_prompts":len(args.excluded_prompt_digests),
            "target_pairs":args.target_pairs}, indent=2))
        return
    torch.cuda.set_device(args.gpu)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    root, pair_dir, rollout_dir, manifest_dir = prepare_output(args)

    cleanup_unmarked_files(
        pair_dir,
        rollout_dir,
        manifest_dir,
        args.layers,
    )

    manifests, totals = validate_completed_shards(
        pair_dir,
        rollout_dir,
        manifest_dir,
        args.layers,
    )

    write_progress(root, totals, args, manifests)

    if totals["pairs"] >= args.target_pairs:
        print(
            f"already complete: {totals['pairs']:,} "
            f">= target {args.target_pairs:,} pairs"
        )
        return

    used_prompt_keys = load_used_prompt_keys(
        rollout_dir,
        manifests,
    )

    if manifests and args.exclude_rollouts:
        keys, _, _ = rollout_inventory(rollout_dir)
        args.excluded_prompt_digests.update(keys)
    next_shard_id = max(manifests, default=-1) + 1

    print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
    print(f"model: {args.model}")
    print(f"layers: {args.layers}")
    print(
        f"resume: {totals['pairs']:,}/{args.target_pairs:,} pairs | "
        f"{len(manifests)} complete shards | "
        f"{len(used_prompt_keys):,} used prompts"
    )
    print(
        f"generation: batch={args.gen_batch} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print(
        f"replay: batch={args.replay_batch} | "
        f"spans<= {args.max_spans}, lengths={args.span_min}..{args.span_max}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        use_fast=True,
        cache_dir=args.cache_dir,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.gpu,
        cache_dir=args.cache_dir,
    )
    model.eval()
    model.config.use_cache = True

    depth = len(model.model.layers)
    bad_layers = [
        layer
        for layer in args.layers
        if layer < 0 or layer >= depth
    ]
    if bad_layers:
        raise ValueError(
            f"Layers {bad_layers} outside model depth 0..{depth-1}"
        )

    env_path = root / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8"))
    revision = getattr(model.config, "_commit_hash", None)
    if env.get("model_revision") and env["model_revision"] != revision:
        raise ValueError("Model revision changed across harvest resume; pin --revision")
    env["model_revision"] = revision
    env.update(
        {
            "gpu": torch.cuda.get_device_name(args.gpu),
            "model_hidden_size": int(model.config.hidden_size),
            "model_num_hidden_layers": int(depth),
        }
    )
    json_dump_atomic(env, env_path)

    prompts = prompt_stream(
        args,
        tokenizer,
        used_prompt_keys,
    )

    pbar = tqdm(
        total=args.target_pairs,
        initial=min(totals["pairs"], args.target_pairs),
        unit="pairs",
        desc="harvest",
    )

    shard_id = next_shard_id

    try:
        while totals["pairs"] < args.target_pairs:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            rows, rollouts, stats = produce_one_shard(
                args=args,
                shard_id=shard_id,
                prompts=prompts,
                model=model,
                tokenizer=tokenizer,
            )

            manifest = write_shard_atomic(
                args=args,
                shard_id=shard_id,
                rows=rows,
                rollouts=rollouts,
                pair_dir=pair_dir,
                rollout_dir=rollout_dir,
                manifest_dir=manifest_dir,
                model=model,
                stats=stats,
            )

            manifests[shard_id] = manifest
            totals["pairs"] += manifest["pairs"]
            totals["rollouts"] += manifest["rollouts"]
            totals["generated_tokens"] += manifest["generated_tokens"]

            for k, v in manifest["span_length_hist"].items():
                totals["span_length_hist"][int(k)] += int(v)

            for ex in rollouts:
                used_prompt_keys.add(prompt_key(ex["prompt"]))

            write_progress(root, totals, args, manifests)

            pbar.update(
                min(
                    manifest["pairs"],
                    args.target_pairs - pbar.n,
                )
            )

            gen_tok_s = (
                manifest["generated_tokens"] / manifest["generation_seconds"]
                if manifest["generation_seconds"] > 0
                else 0.0
            )
            replay_span_s = (
                manifest["pairs"] / manifest["replay_seconds"]
                if manifest["replay_seconds"] > 0
                else 0.0
            )

            peak_alloc = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            peak_reserved = (
                torch.cuda.max_memory_reserved() / 1024**3
            )

            pbar.set_postfix(
                {
                    "shard": f"{shard_id:05d}",
                    "gen": f"{gen_tok_s:.0f} tok/s",
                    "replay": f"{replay_span_s:.0f} span/s",
                }
            )

            print(
                f"\nshard {shard_id:05d} | "
                f"{manifest['rollouts']:,} rollouts | "
                f"{manifest['pairs']:,} pairs | "
                f"gen {manifest['generation_seconds']:.1f}s "
                f"({gen_tok_s:.1f} tok/s) | "
                f"replay {manifest['replay_seconds']:.1f}s "
                f"({replay_span_s:.1f} span/s) | "
                f"VRAM {peak_alloc:.1f}/{peak_reserved:.1f} GB"
            )

            shard_id += 1

    finally:
        pbar.close()

    n_lengths = sum(totals["span_length_hist"].values())
    mean_span_len = (
        sum(k * v for k, v in totals["span_length_hist"].items())
        / n_lengths
        if n_lengths
        else float("nan")
    )

    print()
    print("✓ harvest complete")
    print(f"pairs:            {totals['pairs']:,}")
    print(f"rollouts:         {totals['rollouts']:,}")
    print(f"generated tokens: {totals['generated_tokens']:,}")
    print(f"shards:           {len(manifests):,}")
    print(f"mean span length: {mean_span_len:.3f}")
    print(f"output:           {root}")


if __name__ == "__main__":
    main()

