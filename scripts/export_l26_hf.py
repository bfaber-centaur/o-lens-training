#!/usr/bin/env python3
"""Export a new, allowlisted L26 HF release; optionally publish it."""
import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")

def copy_verified(source, dest):
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected regular source file: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    if sha(source) != sha(dest):
        raise ValueError(f"Copy mismatch: {source}")

def verify_payload(out):
    for line in (out / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        p = Path(relative)
        if p.is_absolute() or ".." in p.parts or sha(out / p) != expected:
            raise ValueError(f"Release checksum mismatch: {relative}")

def publish(out, repo):
    from huggingface_hub import HfApi
    verify_payload(out)
    api = HfApi()  # HF_TOKEN or locally authenticated token; never logged.
    api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
    if api.list_repo_files(repo, repo_type="model") not in ([], [".gitattributes"]):
        raise ValueError("Destination is not empty; refusing to overwrite an existing release")
    result = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(out),
                              commit_message="Release exploratory L26 O-Lens adapters")
    # A separately uploaded checksum file is small enough for exact readback.
    from huggingface_hub import hf_hub_download
    remote = Path(hf_hub_download(repo, "SHA256SUMS", revision=result.oid))
    if sha(remote) != sha(out / "SHA256SUMS"):
        raise ValueError("Remote checksum manifest differs")
    print(f"PUBLISHED https://huggingface.co/{repo}/tree/{result.oid}", flush=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", type=Path, default=Path("/workspace/o-lens-qwen35-4b"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Check packaged reader against one saved expanded greedy readout")
    p.add_argument("--upload-only", action="store_true")
    p.add_argument("--repo", default="bubskee/qwen3.5-4b-o-lenses")
    a = p.parse_args()
    if a.upload_only:
        if not a.publish:
            p.error("--upload-only requires --publish")
        publish(a.output, a.repo)
        return
    if a.output.exists():
        p.error("Export output must be new; use --upload-only for a verified existing export")
    sys.path.insert(0, str(ROOT / "oracle"))
    import torch
    from transformers import AutoTokenizer
    from train_distill_student import render_concepts_prompt
    oracle = a.lab / "oracle"
    # Validate all source contracts before creating any output.
    sources = []
    for variant, count in (("expanded", 2928), ("control", 374)):
        run = oracle / f"l26-data-{variant}-1000-v1"
        cp = run / "checkpoints/step_001000"
        assert (cp / "COMPLETE").is_file(), f"Incomplete checkpoint: {cp}"
        m = read(run / "meta.json")
        assert m["model"] == "Qwen/Qwen3.5-4B" and m["layer"] == 26
        assert m["stage"] == "l26_distill_student_sft_v1"
        assert m["transform"] == "unit" and m["alpha"] == 8000
        assert m["train_rows"] == count and m["training"]["max_steps"] == 1000
        assert m["prompt_mode"] == "concepts_raw"
        assert sha(Path(m["init_from"]) / "lora/adapter_model.safetensors") == m["init_adapter_sha256"]
        train_meta = read(Path(m["train"]) / "meta.json")
        val_meta = read(Path(m["val"]) / "meta.json")
        assert sha(Path(m["train"]) / "targets.safetensors") == m["train_targets_sha256"]
        assert sha(Path(m["val"]) / "targets.safetensors") == m["val_targets_sha256"]
        tok = AutoTokenizer.from_pretrained(cp / "tokenizer")
        ids, slot, char = render_concepts_prompt(tok, 26)
        assert sha_bytes(ids.numpy().tobytes()) == m["rendered_prompt_ids_sha256"]
        assert slot == m["prompt_slot"] and char == m["prompt_char"]
        scored = oracle / f"l26-headroom-{variant}-fve-v1"
        assert (scored / "COMPLETE").is_file()
        headroom = read(scored / "headroom_summary.json")
        assert headroom["score_report_sha256"] == sha(scored / "report.json")
        report, scoring = read(scored / "report.json"), read(scored / "manifest.json")
        assert report["complete"] and len(report["rows"]) == 12
        workout = Path(scoring["workout"])
        generated = read(workout / "manifest.json")
        assert Path(generated["student"]).resolve() == cp.resolve()
        assert generated["student_adapter_sha256"] == sha(cp / "lora/adapter_model.safetensors")
        for filename, digest in scoring["source_files"].items():
            assert sha(filename) == digest, f"Evaluation source changed: {filename}"
        sources.append((variant, run, cp, m, ids.tolist(), train_meta, val_meta,
                        headroom, report, scoring))

    a.output.mkdir(parents=True)
    copy_verified(ROOT / "release/l26-model-card.md", a.output / "README.md")
    copy_verified(ROOT / "release/read_vector.py", a.output / "read_vector.py")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "peft", "safetensors", "accelerate", "huggingface_hub")}
    (a.output / "requirements.txt").write_text(
        "\n".join(f"{k}=={v}" for k,v in versions.items()) + "\n")
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for variant, run, cp, m, ids, tr, va, headroom, report, scoring in sources:
        dest = a.output / "l26" / variant
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            copy_verified(cp / "lora" / filename, dest / "adapter" / filename)
        for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                         "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja"):
            src = cp / "tokenizer" / filename
            if src.is_file():
                copy_verified(src, dest / "tokenizer" / filename)
        assert (dest / "tokenizer/tokenizer_config.json").is_file()
        assert (dest / "tokenizer/tokenizer.json").is_file()
        write(dest / "contract.json", {k:m[k] for k in
            ("model", "revision", "layer", "transform", "alpha", "prompt_mode",
             "prompt", "prompt_slot", "prompt_char", "rendered_prompt_ids_sha256")} |
              {"prompt_ids": ids, "layer_convention": "zero-based decoder block output, before final norm",
               "activation_position": "token immediately preceding target continuation",
               "default_decoding": {"do_sample": False, "max_new_tokens": 64}})
        write(dest / "evaluation.json", {"status": "exploratory selected validation, not independent test",
              "n_items": 12, "sampling": {"n":8, "temperature":0.7, "top_p":0.9, "max_new_tokens":64},
              "headroom_means": headroom["means"], "greedy": report["summary"]["groups"]["all"]["student_shared"],
              "shuffled": report["summary"]["groups"]["all"]["student_shuffled"]})
        write(dest / "provenance.json", {
            "source_code_commit": git, "source_repo": "https://github.com/bubskee/o-lens-training",
            "exporter_sha256": sha(Path(__file__)), "training_metadata": m,
            "checkpoint_metrics": read(cp / "metrics.json"),
            "training_target_metadata": tr, "validation_target_metadata": va,
            "adapter_sha256": sha(cp / "lora/adapter_model.safetensors"),
            "evaluation_manifest": scoring, "packages": versions,
            "torch_cuda": torch.version.cuda,
            "note": "Absolute source paths are provenance only; no raw corpus text or optimizer state is bundled."
        })
    if a.smoke:
        import tempfile
        from safetensors.torch import save_file
        from train_distill_student import load_targets
        workout = oracle / "l26-data-expanded-1000-workout-v1"
        manifest = read(workout / "manifest.json")
        choice = manifest["selected"][0]
        reference = read(workout / f"item_{choice['item_idx']:06d}.json")
        _, bank = load_targets(Path(manifest["val"]), "val")
        with tempfile.TemporaryDirectory(prefix="olens-release-smoke-") as td:
            vector, result = Path(td)/"vector.safetensors", Path(td)/"result.json"
            save_file({"activation": bank["vec"][choice["target_row"]].float().contiguous()}, str(vector))
            subprocess.run([sys.executable, str(a.output/"read_vector.py"),
                            "--release", str(a.output), "--vector", str(vector),
                            "--result", str(result)], check=True)
            if read(result)["text"] != reference["outputs"]["student_shared"]["text"]:
                raise ValueError("Packaged greedy differs from saved readout; inspect before publishing")
        write(a.output/"smoke.json", {"item_idx":choice["item_idx"],
              "variant":"expanded", "saved_greedy_text_exact_match":True})
    payload = sorted(x for x in a.output.rglob("*") if x.is_file())
    (a.output / "SHA256SUMS").write_text(
        "".join(f"{sha(x)}  {x.relative_to(a.output).as_posix()}\n" for x in payload))
    verify_payload(a.output)
    print(f"EXPORT VERIFIED {a.output} ({len(payload)} payload files)", flush=True)
    if a.publish:
        publish(a.output, a.repo)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

if __name__ == "__main__":
    main()
