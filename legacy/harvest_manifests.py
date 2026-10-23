"""CPU-only shard commit discovery and cleanup for FUSE-backed harvests."""
import json
import re
from pathlib import Path

PAIR_RE = re.compile(r"pairs_l(?P<layer>\d+)_(?P<shard>\d{5})\.safetensors$")
ROLLOUT_RE = re.compile(r"rollouts_(?P<shard>\d{5})\.jsonl$")
MANIFEST_RE = re.compile(r"shard_(?P<shard>\d{5})\.json$")


def scan_manifests(manifest_dir: Path):
    """Discover committed shards despite incomplete FUSE directory listings.

    Listings and progress are discovery hints only. A directly read manifest
    is the commit record; progress never substitutes for one.
    """
    candidates = set()
    for directory, pattern, regex in (
        (manifest_dir, "shard_*.json", MANIFEST_RE),
        (manifest_dir.parent / "pairs", "pairs_l*_*.safetensors", PAIR_RE),
        (manifest_dir.parent / "rollouts", "rollouts_*.jsonl", ROLLOUT_RE),
    ):
        for path in directory.glob(pattern):
            match = regex.fullmatch(path.name)
            if match:
                candidates.add(int(match.group("shard")))
    progress_path = manifest_dir.parent / "progress.json"
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        progress = {}
    count = progress.get("complete_shards", 0)
    if type(count) is not int or not 0 <= count <= 100000:
        raise ValueError("Invalid committed-shard count in progress.json")
    # This writer starts at zero and commits successive shard IDs.
    candidates.update(range(count))
    manifests = {}
    for sid in sorted(candidates):
        path = manifest_dir / f"shard_{sid:05d}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if sid < count:
                raise RuntimeError(
                    f"Progress records committed shard {sid:05d}, but its manifest "
                    "cannot be read directly. Refusing cleanup; check storage visibility."
                ) from None
            continue
        if record.get("shard_id") != sid:
            raise ValueError(f"Manifest shard identity mismatch: {path}")
        manifests[sid] = record
    return manifests


def cleanup_unmarked_files(pair_dir, rollout_dir, manifest_dir, layers):
    complete = set(scan_manifests(manifest_dir))

    for directory in (pair_dir, rollout_dir, manifest_dir):
        for p in directory.glob("*.tmp"):
            print(f"removing stale temp: {p}")
            p.unlink(missing_ok=True)

    found_ids = set()

    for p in pair_dir.glob("pairs_l*_*.safetensors"):
        m = PAIR_RE.match(p.name)
        if m:
            found_ids.add(int(m.group("shard")))

    for p in rollout_dir.glob("rollouts_*.jsonl"):
        m = ROLLOUT_RE.match(p.name)
        if m:
            found_ids.add(int(m.group("shard")))

    for sid in sorted(found_ids - complete):
        path = manifest_dir / f"shard_{sid:05d}.json"
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        else:
            if record.get("shard_id") != sid:
                raise ValueError(f"Manifest shard identity mismatch: {path}")
            continue
        print(f"removing uncommitted shard {sid:05d}")
        for layer in layers:
            (pair_dir / f"pairs_l{layer}_{sid:05d}.safetensors").unlink(
                missing_ok=True
            )
        (rollout_dir / f"rollouts_{sid:05d}.jsonl").unlink(
            missing_ok=True
        )

