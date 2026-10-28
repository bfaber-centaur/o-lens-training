#!/usr/bin/env python3
"""CPU-only inventory and hash receipt for the historical L30 lineage."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", type=Path, default=Path("/workspace/o-lens-qwen35-4b"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Receipt exists; choose a new filename")
    cp = a.lab / "ar/l30_200k_v3/checkpoints/step_001412"
    data = a.lab / "data/olens_200k_l22_l26_l30"
    reference = json.loads((Path(__file__).resolve().parents[1] /
        "legacy/artifacts/ar/l30_200k_v3/run_config.json").read_text())
    cfg = json.loads((cp / "run_config.json").read_text())
    for key in ("model_id", "target_layer", "d_model", "split_seed", "val_frac",
                "pair_shards", "rollout_shards", "whitener_sha256"):
        if cfg[key] != reference[key]:
            raise ValueError(f"Historical L30 contract mismatch: {key}")
    files = [cp / name for name in ("COMPLETE", "run_config.json", "training_state.pt",
        "lora/adapter_config.json", "lora/adapter_model.safetensors")]
    whitener = a.lab / "ar/l30_200k_v3/whitener_l30.safetensors"
    files.append(whitener)
    # Explicit names avoid relying on FUSE directory listings for inventory.
    files += [data / "pairs" / name for name in reference["pair_shards"]]
    files += [data / "rollouts" / name for name in reference["rollout_shards"]]
    hashes = {}
    for path in files:
        try:
            hashes[str(path)] = sha(path)
        except OSError as e:
            raise RuntimeError(f"Cannot read required source: {path}") from e
    if hashes[str(whitener)] != reference["whitener_sha256"]:
        raise ValueError("Historical whitener checksum mismatch")
    result = {"status": "sources_readable", "layer": 30, "checkpoint": str(cp),
        "data": str(data), "pair_shards": len(reference["pair_shards"]),
        "rollout_shards": len(reference["rollout_shards"]), "sha256": hashes,
        "note": "Current hashes frozen; historical adapter/head identity not independently verified."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(f"L30 PREFLIGHT OK: {len(hashes)} files; receipt={a.output}")

if __name__ == "__main__":
    main()
