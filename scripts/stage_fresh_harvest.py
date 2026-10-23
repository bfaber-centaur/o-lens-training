#!/usr/bin/env python3
"""Stage a completed harvest by exact paths; never enumerate or mutate its source."""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def copy_verified(source, target, attempts=5, delay=2):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.partial')
    for attempt in range(attempts):
        try:
            h = hashlib.sha256()
            with source.open('rb') as src, temp.open('wb') as dst:
                for chunk in iter(lambda: src.read(4 * 1024 * 1024), b''):
                    h.update(chunk)
                    dst.write(chunk)
            expected = h.hexdigest()
            # Independent direct read, plus local readback. No source glob/stat.
            if digest(source) != expected or digest(temp) != expected:
                raise ValueError(f'Content changed or copy mismatch: {source}')
            temp.replace(target)
            return expected
        except FileNotFoundError:
            if attempt + 1 == attempts:
                raise
            print(f'Retrying exact read: {source} ({attempt+1}/{attempts})', flush=True)
            time.sleep(delay)
    raise AssertionError('unreachable')


def pair_rows(path):
    # Read safetensors shape without loading Torch. Full tensor validation is
    # still performed by the existing preparation pipeline after staging.
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f'Truncated safetensors header: {path}')
        size = struct.unpack('<Q', raw)[0]
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError(f'Invalid safetensors header size: {path}')
        header = json.loads(f.read(size))
    shape = header['activations']['shape']
    if len(shape) != 2 or shape[1] != 2560:
        raise ValueError(f'Unexpected L26 activation shape: {path}')
    return shape[0]


def stage(source, output, attempts=5, delay=2):
    source, output = Path(source).absolute(), Path(output).absolute()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Source and destination must be separate trees')
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}

    def copy(relative):
        hashes[relative] = copy_verified(source/relative, output/relative, attempts, delay)
        return output/relative

    progress = json.loads(copy('progress.json').read_text())
    count = progress['complete_shards']
    if type(count) is not int or not 1 <= count <= 100000:
        raise ValueError('Invalid shard count')
    if progress['pairs'] < progress['target_pairs']:
        raise ValueError('Harvest has not reached its recorded target')
    config = json.loads(copy('run_config.json').read_text())
    if config['contract']['layers'] != [26]:
        raise ValueError('This staging command expects L26-only harvest')
    copy('environment.json')
    totals = dict(pairs=0, rollouts=0, generated_tokens=0)
    histogram = {}
    for sid in range(count):
        m = json.loads(copy(f'manifests/shard_{sid:05d}.json').read_text())
        pair = f'pairs_l26_{sid:05d}.safetensors'
        rollout = f'rollouts_{sid:05d}.jsonl'
        if (m['shard_id'] != sid or m['layers'] != [26] or
                m['pair_files'] != {'26':pair} or m['rollout_file'] != rollout):
            raise ValueError(f'Shard identity/filename mismatch: {sid}')
        p = copy('pairs/' + pair)
        if pair_rows(p) != m['pairs']:
            raise ValueError(f'Pair count mismatch: {sid}')
        r = copy('rollouts/' + rollout)
        rows, tokens = 0, 0
        with r.open() as f:
            for line in f:
                record = json.loads(line)
                rows += 1
                tokens += len(record['assistant_ids'])
        if (rows, tokens) != (m['rollouts'], m['generated_tokens']):
            raise ValueError(f'Rollout/token count mismatch: {sid}')
        for k in totals:
            totals[k] += m[k]
        for k,v in m['span_length_hist'].items():
            histogram[k] = histogram.get(k, 0) + v
        print(f'Verified shard {sid+1}/{count}: {m["pairs"]} pairs', flush=True)
    if any(totals[k] != progress[k] for k in totals) or histogram != progress['span_length_hist']:
        raise ValueError('Shard totals differ from saved progress')
    # Source must be quiescent throughout staging.
    for name in ('progress.json', 'run_config.json', 'environment.json'):
        if digest(source/name) != hashes[name]:
            raise ValueError(f'Source changed during staging: {name}')
    report = {'schema':'verified_local_harvest_snapshot_v1', 'source':str(source),
              'files_sha256':hashes, 'totals':totals, 'shards':count}
    (output/'snapshot.json').write_text(json.dumps(report, indent=2)+'\n')
    (output/'STAGED_COMPLETE').write_text('ok\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(stage(a.source, a.output), indent=2))


if __name__ == '__main__':
    main()
