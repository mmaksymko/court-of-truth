"""Estimate paired metric uncertainty for a candidate versus one baseline mode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from court.experiment.metrics import macro_f1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-mode", choices=("F", "B1", "B2", "B3"), default="F")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-mode", choices=("F", "B1", "B2", "B3"), required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _exact_mcnemar(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    lower = min(candidate_only, baseline_only)
    tail = sum(math.comb(discordant, k) * 0.5**discordant for k in range(lower + 1))
    return min(1.0, 2 * tail)


def main() -> None:
    args = _parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists and will not be overwritten: {args.out}")
    ids = sorted(set(args.ids), key=lambda value: int(value[2:]))
    gold_map = {row["id"]: row["final_verdict"] for row in _rows(args.gold)}
    candidate_map = {
        row["item_id"]: row["label"]
        for row in _rows(args.candidate)
        if row["mode"] == args.candidate_mode and row.get("label")
    }
    baseline_map = {
        row["item_id"]: row["label"]
        for row in _rows(args.baseline)
        if row["mode"] == args.baseline_mode and row.get("label")
    }
    missing = sorted(set(ids) - set(candidate_map) | (set(ids) - set(baseline_map)))
    if missing:
        raise SystemExit(f"missing successful predictions: {', '.join(missing)}")
    gold = np.array([gold_map[item_id] for item_id in ids])
    candidate = np.array([candidate_map[item_id] for item_id in ids])
    baseline = np.array([baseline_map[item_id] for item_id in ids])
    candidate_correct = candidate == gold
    baseline_correct = baseline == gold
    candidate_only = int(np.sum(candidate_correct & ~baseline_correct))
    baseline_only = int(np.sum(~candidate_correct & baseline_correct))

    rng = np.random.default_rng(args.seed)
    accuracy_differences = np.empty(args.resamples)
    macro_f1_differences = np.empty(args.resamples)
    for index in range(args.resamples):
        sample = rng.integers(0, len(ids), len(ids))
        accuracy_differences[index] = np.mean(candidate_correct[sample]) - np.mean(
            baseline_correct[sample]
        )
        macro_f1_differences[index] = macro_f1(
            gold[sample].tolist(), candidate[sample].tolist()
        ) - macro_f1(gold[sample].tolist(), baseline[sample].tolist())

    result = {
        "n": len(ids),
        "candidate_mode": args.candidate_mode,
        "baseline_mode": args.baseline_mode,
        "paired_correctness": {
            "candidate_only_correct": candidate_only,
            "baseline_only_correct": baseline_only,
            "exact_mcnemar_two_sided_p": _exact_mcnemar(candidate_only, baseline_only),
        },
        "differences_candidate_minus_baseline": {
            "accuracy": {
                "estimate": float(np.mean(candidate_correct) - np.mean(baseline_correct)),
                "bootstrap_95_ci": np.quantile(accuracy_differences, (0.025, 0.975)).tolist(),
            },
            "macro_f1": {
                "estimate": macro_f1(gold.tolist(), candidate.tolist())
                - macro_f1(gold.tolist(), baseline.tolist()),
                "bootstrap_95_ci": np.quantile(macro_f1_differences, (0.025, 0.975)).tolist(),
            },
        },
        "bootstrap": {"resamples": args.resamples, "seed": args.seed},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
