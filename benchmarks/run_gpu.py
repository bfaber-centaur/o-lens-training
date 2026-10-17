#!/usr/bin/env python3
"""Launch isolated, concurrent REAL AR training jobs and record GPU telemetry.

Example:
  python benchmarks/run_gpu.py --label b200-three --layers 22 26 30 --gpu-hourly 5.98
Never modifies legacy training outputs; existing benchmark output is rejected.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA = Path("/workspace/o-lens-qwen35-4b/data/olens_200k_l22_l26_l30")
DEFAULT_OUTPUT = Path("/workspace/o-lens-benchmarks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Unique experiment label, e.g. a100-two-l22-l26")
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--gpu-hourly", type=float, default=None)
    parser.add_argument("--whitener", action="append", default=[],
                        help="LAYER=/absolute/path/to/whitener.safetensors (repeatable)")
    args = parser.parse_args()
    if args.max_steps < 1 or args.eval_every < 1 or args.micro_batch < 1 or args.grad_accum < 1:
        parser.error("steps, eval frequency, batch size and accumulation must be positive")
    if len(set(args.layers)) != len(args.layers):
        parser.error("layers must be unique")
    if not args.label.replace("-", "").replace("_", "").isalnum():
        parser.error("label must contain only letters, digits, hyphens and underscores")
    root = Path(__file__).resolve().parents[1]
    trainer = root / "benchmarks/train_ar_bounded.py"
    if not trainer.is_file():
        parser.error(f"trainer not found: {trainer}")
    pair_dir, rollout_dir = args.data / "pairs", args.data / "rollouts"
    if not rollout_dir.is_dir():
        parser.error(f"missing rollouts: {rollout_dir}")
    for layer in args.layers:
        if not list(pair_dir.glob(f"pairs_l{layer}_*.safetensors")):
            parser.error(f"missing activation pairs for L{layer} in {pair_dir}; harvest before launching")
    whiteners = {}
    for item in args.whitener:
        try:
            key, value = item.split("=", 1)
            layer = int(key)
        except ValueError:
            parser.error(f"invalid --whitener {item!r}; expected LAYER=/path")
        if layer not in args.layers or not Path(value).is_file():
            parser.error(f"missing or unused whitener: {item}")
        whiteners[layer] = value

    out = args.output_root / args.label
    if out.exists():
        parser.error(f"output exists (refusing overwrite): {out}")
    out.mkdir(parents=True)
    manifest = {
        "label": args.label, "layers": args.layers, "max_steps": args.max_steps,
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
        "eval_every": args.eval_every, "data": str(args.data),
        "whiteners": whiteners, "gpu_hourly_usd": args.gpu_hourly,
        "python": sys.version, "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "benchmark_config.json").write_text(json.dumps(manifest, indent=2) + "\n")

    processes = {}
    logs = {}
    monitor = None
    monitor_file = None
    start = time.monotonic()
    interrupted = False

    def stop_children(*_):
        nonlocal interrupted
        interrupted = True
        for p in processes.values():
            if p.poll() is None:
                p.terminate()

    old_int = signal.signal(signal.SIGINT, stop_children)
    old_term = signal.signal(signal.SIGTERM, stop_children)
    try:
        monitor_file = (out / "gpu.csv").open("w")
        monitor = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu",
             "--format=csv", "-l", "2"],
            stdout=monitor_file, stderr=subprocess.DEVNULL)
        for layer in args.layers:
            dest = out / f"l{layer}"
            command = [
                sys.executable, "-u", str(trainer),
                "--layer", str(layer), "--pair-dir", str(pair_dir),
                "--rollout-dir", str(rollout_dir), "--output", str(dest),
                "--micro-batch", str(args.micro_batch),
                "--grad-accum", str(args.grad_accum),
                "--eval-every", str(args.eval_every),
                "--max-steps", str(args.max_steps),
                "--data-note", f"benchmark {args.label}; fresh per-layer run",
            ]
            if layer in whiteners:
                command.extend(["--whitener", whiteners[layer]])
            logs[layer] = (out / f"l{layer}.log").open("w")
            processes[layer] = subprocess.Popen(command, stdout=logs[layer],
                                                 stderr=subprocess.STDOUT,
                                                 start_new_session=True)
            print(f"L{layer} pid={processes[layer].pid} log={out}/l{layer}.log", flush=True)
        failures = {}
        for layer, process in processes.items():
            rc = process.wait()
            failures[layer] = rc
            print(f"L{layer}: exit={rc}", flush=True)
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        if monitor_file:
            monitor_file.close()
        for f in logs.values():
            f.close()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)

    elapsed = time.monotonic() - start
    stats = {"wall_seconds": round(elapsed, 3), "exit_codes": failures,
             "successful_layers": [l for l in args.layers if failures.get(l) == 0],
             "gpu_hourly_usd": args.gpu_hourly,
             "estimated_compute_usd": round(elapsed / 3600 * args.gpu_hourly, 4)
             if args.gpu_hourly is not None else None}
    try:
        with (out / "gpu.csv").open() as f:
            samples = list(csv.DictReader(f))
        for column, key in [("memory.used", "gpu_memory_peak_mib"),
                            ("utilization.gpu", "gpu_utilization_max_pct")]:
            values = []
            for row in samples:
                col = next((k for k in row if column in k), None)
                if col and row[col].strip():
                    try:
                        values.append(float(row[col].strip().split()[0]))
                    except ValueError:
                        pass
            stats[key] = max(values) if values else None
        stats["gpu_samples"] = len(samples)
    except OSError:
        pass
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0 if not interrupted and all(v == 0 for v in failures.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
