"""Create a deterministic, class-balanced calibration/holdout split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

LABELS = ("reliable", "questionable", "unreliable")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--calibration-per-class", type=int, default=8)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--baseline-f-prompt-sha256", required=True)
    parser.add_argument("--baseline-b1-prompt-sha256", required=True)
    parser.add_argument("--candidate-f-prompt-sha256", required=True)
    return parser.parse_args()


def _rank(seed: str, item_id: str) -> str:
    return hashlib.sha256(f"{seed}|{item_id}".encode()).hexdigest()


def main() -> None:
    args = _parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists and will not be overwritten: {args.out}")
    grouped: dict[str, list[str]] = defaultdict(list)
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped[row["final_verdict"]].append(row["id"])
    if set(grouped) != set(LABELS):
        raise SystemExit(f"unexpected gold labels: {sorted(grouped)}")

    calibration: list[str] = []
    holdout: list[str] = []
    for label in LABELS:
        ranked = sorted(grouped[label], key=lambda item_id: _rank(args.seed, item_id))
        calibration += ranked[: args.calibration_per_class]
        holdout += ranked[args.calibration_per_class :]

    payload = {
        "prompt_version": args.prompt_version,
        "purpose": "Improve adversarial evidence synthesis without case-specific rules",
        "split": {
            "method": "SHA-256(seed|item_id), ranked independently within each gold class",
            "seed": args.seed,
            "calibration_per_class": args.calibration_per_class,
            "calibration_ids": sorted(calibration, key=lambda value: int(value[2:])),
            "holdout_ids": sorted(holdout, key=lambda value: int(value[2:])),
        },
        "prompt_hashes": {
            "baseline_F": args.baseline_f_prompt_sha256,
            "baseline_B1": args.baseline_b1_prompt_sha256,
            "candidate_F": args.candidate_f_prompt_sha256,
        },
        "guardrails": [
            "No TC-ID, article, domain, or gold-label-specific instruction may enter the prompt.",
            "Inspect aggregate calibration metrics only before freezing the candidate.",
            "Do not inspect candidate holdout outputs before the prompt is frozen.",
            "After holdout is opened, preserve the result and do not tune this version against it.",
        ],
        "calibration_gate": {
            "candidate_F_accuracy": ">= baseline_F_accuracy",
            "candidate_F_macro_f1": "> baseline_F_macro_f1",
            "candidate_F_vs_B1": "at least one of accuracy or macro_f1 must be higher",
        },
        "holdout_success": {
            "primary": "candidate_F macro_f1 > baseline_B1 macro_f1",
            "secondary": "candidate_F accuracy > baseline_B1 accuracy",
            "safety": "no per-class recall drop greater than one holdout item versus baseline_F",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(calibration)} calibration, {len(holdout)} holdout")


if __name__ == "__main__":
    main()
