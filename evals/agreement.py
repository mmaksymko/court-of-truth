"""Inter-annotator agreement + arbitration worksheet for the tribunal corpus.

Compares a second annotator's verdicts against the curator gold in cases.jsonl:
Cohen's kappa (3-class), full-agreement rate, per-class agreement and a
confusion matrix, then writes the disagreements for a third-party arbiter.

Run: uv run --no-sync python evals/agreement.py <annotator2.jsonl>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

EVALS = Path(__file__).resolve().parent
LABELS = ("reliable", "questionable", "unreliable")


def _load_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["id"]] = record
    return records


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python evals/agreement.py <annotator2.jsonl>")
    second = _load_jsonl(Path(sys.argv[1]))
    gold = {
        record["id"]: {
            "verdict": record["gold"]["verdict"],
            "rationale": record["gold"]["rationale"],
        }
        for record in _load_jsonl(EVALS / "cases.jsonl").values()
    }

    shared = [item_id for item_id in gold if item_id in second and second[item_id].get("verdict")]
    missing = sorted(set(gold) - set(shared))
    a1 = [gold[item_id]["verdict"] for item_id in shared]
    a2 = [second[item_id]["verdict"] for item_id in shared]

    invalid = sorted({v for v in a2 if v not in LABELS})
    if invalid:
        raise SystemExit(f"annotator 2 used unknown verdicts: {invalid}")

    agree = sum(x == y for x, y in zip(a1, a2, strict=True))
    confusion = defaultdict(int)
    for x, y in zip(a1, a2, strict=True):
        confusion[f"{x}->{y}"] += 1
    disagreements = [
        {
            "id": item_id,
            "gold": gold[item_id]["verdict"],
            "annotator2": second[item_id]["verdict"],
            "gold_rationale": gold[item_id]["rationale"],
            "annotator2_rationale": second[item_id].get("rationale", ""),
        }
        for item_id in shared
        if gold[item_id]["verdict"] != second[item_id]["verdict"]
    ]

    report = {
        "n_compared": len(shared),
        "n_missing_from_annotator2": len(missing),
        "missing_ids": missing,
        "cohen_kappa": round(float(cohen_kappa_score(a1, a2, labels=list(LABELS))), 4),
        "full_agreement": agree,
        "full_agreement_rate": round(agree / len(shared), 4) if shared else 0.0,
        "confusion": dict(sorted(confusion.items())),
        "n_disagreements": len(disagreements),
    }
    (EVALS / "annotation").mkdir(exist_ok=True)
    (EVALS / "annotation" / "disagreements.json").write_text(
        json.dumps(disagreements, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        f"\nwrote {EVALS / 'annotation' / 'disagreements.json'} ({len(disagreements)} to arbitrate)"
    )


if __name__ == "__main__":
    main()
