#!/usr/bin/env python3
"""Bounded L26 sampling headroom; reuses existing generation/scoring contracts."""
import argparse
import json
import sys
from pathlib import Path

SEEDS = (11, 22, 33, 44, 55, 66, 77, 88)
PROFILES = {
    f"headroom_s{s}": dict(max_new_tokens=64, do_sample=True,
                          temperature=0.7, top_p=0.9, top_k=0, seed=s)
    for s in SEEDS
}
ARMS = [f"decode_{name}_correct" for name in PROFILES]


def aggregate(rows):
    """Each activation has equal weight; best-of-eight excludes greedy."""
    def mean(values):
        return sum(values) / len(values)
    items = []
    for row in rows:
        samples = [row["scores"][arm] for arm in ARMS]
        # Stable tie break follows the prespecified seed order.
        best = max(range(len(samples)), key=lambda j: samples[j]["fve"])
        greedy = row["scores"]["student_shared"]
        items.append({
            "item_idx": row["item_idx"], "stratum": row["stratum"],
            "greedy_fve": greedy["fve"],
            "sample_mean_fve": mean([s["fve"] for s in samples]),
            "best_of_8_fve": samples[best]["fve"], "best_arm": ARMS[best],
            "sample_mean_complementarity": mean([s["complementarity"] for s in samples]),
            "best_complementarity": samples[best]["complementarity"],
            "greedy_complementarity": greedy["complementarity"],
            "valid_fraction": mean([float(s["valid"]) for s in samples]),
            "cap_fraction": mean([float(s["hit_cap"]) for s in samples]),
        })
    if not items:
        raise ValueError("Empty score report")
    keys = [k for k in items[0] if k not in ("item_idx", "stratum", "best_arm")]
    return {"items": items, "means": {k: mean([i[k] for i in items]) for k in keys}}


def summarize(workout, scores):
    import score_generated_readouts as scorer
    report = scorer.read_json(scores / "report.json")
    manifest = scorer.read_json(scores / "manifest.json")
    generated = scorer.read_json(workout / "manifest.json")
    scorer.require(report["complete"] and (scores / "COMPLETE").is_file()
                   and (workout / "COMPLETE").is_file(), "Incomplete diagnostic")
    scorer.require(Path(manifest["workout"]).resolve() == workout.resolve(),
                   "Score/workout mismatch")
    scorer.require(generated["comparison"]["profiles"] == PROFILES,
                   "Unexpected sampling settings")
    for filename, digest in manifest["source_files"].items():
        scorer.require(scorer.sha256(Path(filename)) == digest,
                       f"Source changed after scoring: {filename}")
    scorer.require(len(report["rows"]) == len(generated["selected"]) == 12,
                   "Expected the fixed twelve-item workout")
    result = aggregate(report["rows"])
    result["score_report_sha256"] = scorer.sha256(scores / "report.json")
    scorer.atomic_json(scores / "headroom_summary.json", result)
    lines = [
        "# L26 sampling headroom", "",
        "12 selected validation items, eight samples each; T=0.7, top-p=0.9, cap=64.",
        "Greedy is the saved baseline, rescored here. Best-of-8 is scorer-selected",
        "diagnostic headroom, not ordinary performance or semantic ground truth.",
        "Complementarity = joint FVE minus strongest individual scored bullet FVE.",
        "Invalid samples remain zero in means. Seeds are nested within items.", "",
        "| Measure | Mean |", "|---|---:|",
    ]
    lines += [f"| {key} | {value:.4f} |" for key, value in result["means"].items()]
    # Show every item to avoid selecting only flattering examples.
    for item in result["items"]:
        row = scorer.read_json(workout / f"item_{item['item_idx']:06d}.json")
        lines += ["", f"## Item {item['item_idx']} — {item['stratum']}",
                  f"Sample mean {item['sample_mean_fve']:.4f}; "
                  f"best-of-8 {item['best_of_8_fve']:.4f}.", ""]
        for label, arm in (("Saved greedy", "student_shared"),
                           ("Fixed sample (seed 11)", ARMS[0]),
                           ("Scorer-selected best-of-8", item["best_arm"])):
            lines += [f"### {label}", "", "~~~~text",
                      row["outputs"][arm]["text"], "~~~~", ""]
    (scores / "headroom_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["means"], indent=2))
    print(scores / "headroom_summary.md")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "generate":
        import compare_decoding as comparison
        comparison.PROFILES = PROFILES
        comparison.CONDITIONS = ("correct",)
        # Existing generator retains baseline/shuffled outputs and validates their hashes.
        comparison.main()
    elif command == "score":
        import score_generated_readouts as scorer
        original = scorer.score_vectors
        def with_single_bullets(raw, true, mean, whitener):
            result = original(raw, true, mean, whitener)
            singles = [original(raw[j:j+1], true, mean, whitener)["fve"]
                       for j in range(len(raw))]
            result["single_bullet_fve"] = singles
            result["strongest_bullet_fve"] = max(singles, default=0.0)
            result["complementarity"] = result["fve"] - result["strongest_bullet_fve"]
            return result
        scorer.score_vectors = with_single_bullets
        del sys.argv[1]
        scorer.main()
    elif command == "summarize":
        parser = argparse.ArgumentParser()
        parser.add_argument("--workout", type=Path, required=True)
        parser.add_argument("--scores", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        summarize(args.workout, args.scores)
    else:
        raise SystemExit("Usage: l26_sampling_headroom.py {generate|score|summarize} ...")


if __name__ == "__main__":
    main()
