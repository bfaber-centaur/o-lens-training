"""Prompt exclusions shared by harvesting and fresh-bank auditing (CPU only)."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prompt_digest(text):
    # Stronger than legacy exact-text keys: normalize whitespace for fresh runs.
    return hashlib.sha256(' '.join(text.split()).encode('utf-8')).hexdigest()


def rollout_inventory(directory):
    paths = sorted(Path(directory).glob('rollouts_*.jsonl'))
    if not paths:
        raise ValueError(f'No rollout files in {directory}')
    keys = set()
    count = 0
    hashes = {}
    for path in paths:
        hashes[str(path.resolve())] = file_digest(path)
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                    raise ValueError(f'Missing source prompt in {path}')
                keys.add(prompt_digest(row['prompt']))
                count += 1
    return keys, hashes, count


def audit_fresh(old_dir, fresh_dir):
    old, old_hashes, _ = rollout_inventory(old_dir)
    fresh, fresh_hashes, count = rollout_inventory(fresh_dir)
    if old & fresh:
        raise ValueError('Fresh harvest repeats prompts from original harvest')
    if len(fresh) != count:
        raise ValueError('Fresh harvest repeats prompts internally')
    return {'original_rollout_sha256':old_hashes, 'fresh_rollout_sha256':fresh_hashes,
            'fresh_prompts':count, 'prompt_overlap':0}
