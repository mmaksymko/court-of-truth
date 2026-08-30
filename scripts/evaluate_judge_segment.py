"""Compare an immutable judge-only F candidate with frozen F/B1 baselines."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from court.experiment.metrics import LABELS, macro_f1, multiclass_brier


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _metrics(rows: list[dict], gold: dict[str, str], ids: list[str]) -> dict:
    by_id = {row["item_id"]: row for row in rows if row.get("label")}
    missing = sorted(set(ids) - set(by_id))
    if missing:
        raise SystemExit(f"missing successful predictions: {', '.join(missing)}")
    actual = [gold[item_id] for item_id in ids]
    predicted = [by_id[item_id]["label"] for item_id in ids]
    probabilities = [by_id[item_id]["probabilities"] for item_id in ids]
    correct = [prediction == label for prediction, label in zip(predicted, actual, strict=True)]
    recalls = {}
    for label in LABELS:
        members = [index for index, value in enumerate(actual) if value == label]
        recalls[label] = sum(predicted[index] == label for index in members) / len(members)
    return {
        "n": len(ids),
        "accuracy": sum(correct) / len(ids),
        "macro_f1": macro_f1(actual, predicted),
        "brier": multiclass_brier(actual, probabilities),
        "prediction_counts": dict(Counter(predicted)),
        "recall": recalls,
        "correct_by_id": dict(zip(ids, correct, strict=True)),
    }


def main() -> None:
    args = _parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists and will not be overwritten: {args.out}")
    ids = sorted(set(args.ids), key=lambda value: int(value[2:]))
    gold = {row["id"]: row["final_verdict"] for row in _rows(args.gold)}
    baseline_rows = _rows(args.baseline)
    candidate_rows = [row for row in _rows(args.candidate) if row["mode"] == "F"]
    groups = {
        "candidate_F": _metrics(candidate_rows, gold, ids),
        "baseline_F": _metrics([row for row in baseline_rows if row["mode"] == "F"], gold, ids),
        "baseline_B1": _metrics([row for row in baseline_rows if row["mode"] == "B1"], gold, ids),
    }
    candidate_correct = groups["candidate_F"].pop("correct_by_id")
    paired = {}
    for baseline_name in ("baseline_F", "baseline_B1"):
        baseline_correct = groups[baseline_name].pop("correct_by_id")
        paired[baseline_name] = {
            "candidate_only_correct": sum(
                candidate_correct[item_id] and not baseline_correct[item_id] for item_id in ids
            ),
            "baseline_only_correct": sum(
                baseline_correct[item_id] and not candidate_correct[item_id] for item_id in ids
            ),
        }
    result = {"ids": ids, "groups": groups, "paired_correctness": paired}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
