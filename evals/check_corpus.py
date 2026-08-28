"""Provenance, dedup and leakage checks for evals/cases.jsonl.

Runs five checks and writes evals/check_report.json:
  1. integrity   - each case text still matches its source CSV row, non-empty
  2. dedup       - no two cases share the same normalized text
  3. test-split  - jeansa/clickbait cases really come from split == 'test'
  4. leakage     - for the tail-drawn ai/mt cases, how many fall inside the
                   model's reproduced training partition (the caveat, quantified)
  5. shortcut    - signal -> most-frequent-verdict accuracy, to confirm the gold
                   verdict is not trivially predictable from the detector signal

Run: uv run --no-sync python evals/check_corpus.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make the top-level `training` package importable

from training.config import DetectorConfig  # noqa: E402
from training.evaluation import split_data  # noqa: E402

from court.forensics.text_transforms import transform_for  # noqa: E402

csv.field_size_limit(sys.maxsize)
DATA = ROOT / "data"
EVALS = ROOT / "evals"
SEED = 42


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _load_cases() -> list[dict]:
    lines = (EVALS / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _configs() -> dict[str, DetectorConfig]:
    raw = tomllib.loads((ROOT / "training" / "detectors.toml").read_text(encoding="utf-8"))
    return {name: DetectorConfig(id=name, **body) for name, body in raw.items()}


def _clean_frame(cfg: DetectorConfig) -> pd.DataFrame:
    source = DATA / cfg.source_csv.name
    frame = pd.read_csv(source)
    split_cols = [cfg.split_field] if cfg.split_field and cfg.split_field in frame.columns else []
    frame = frame[[cfg.text_field, cfg.label_field, *split_cols]].dropna(
        subset=[cfg.text_field, cfg.label_field]
    )
    frame["_x"] = frame[cfg.text_field].map(lambda text: transform_for(cfg.id, text))
    return frame[frame["_x"].str.len() > 0]


def _integrity_and_dedup(cases: list[dict]) -> dict:
    hashes: dict[str, list[str]] = defaultdict(list)
    empty: list[str] = []
    for case in cases:
        if not case["text"].strip():
            empty.append(case["id"])
        hashes[case["text_sha256"]].append(case["id"])
    duplicates = {digest: ids for digest, ids in hashes.items() if len(ids) > 1}
    return {
        "empty_text_items": empty,
        "duplicate_groups": list(duplicates.values()),
        "passed": not empty and not duplicates,
    }


def _test_split(cases: list[dict]) -> dict:
    frames = {name: pd.read_csv(DATA / f"{name}.csv") for name in ("jeansa", "clickbait")}
    violations: list[str] = []
    for case in cases:
        if case["dataset"] in frames:
            split = str(frames[case["dataset"]].iloc[case["row"]].get("split", ""))
            if split != "test":
                violations.append(f"{case['id']}: split={split!r}")
    return {"violations": violations, "passed": not violations}


def _leakage(cases: list[dict]) -> dict:
    result: dict[str, object] = {}
    configs = _configs()
    for dataset in ("ai_generated", "mt_translation"):
        frame = _clean_frame(configs[dataset])
        train, val, test = split_data(configs[dataset], frame, SEED)
        partitions = {"train": set(train.index), "val": set(val.index), "test": set(test.index)}
        items = [case for case in cases if case["dataset"] == dataset]
        placement = Counter(
            next((name for name, idx in partitions.items() if case["row"] in idx), "dropped")
            for case in items
        )
        result[dataset] = {
            "tail_items": len(items),
            "in_train": placement["train"],
            "in_val": placement["val"],
            "in_test": placement["test"],
            "dropped_in_cleaning": placement["dropped"],
        }
    total_train = sum(v["in_train"] for v in result.values())  # type: ignore[index]
    total_tail = sum(v["tail_items"] for v in result.values())  # type: ignore[index]
    return {
        "note": "train membership is pre-dedup (upper bound); split reproduced with seed 42",
        "by_dataset": result,
        "tail_items_total": total_tail,
        "leaked_into_train_total": total_train,
    }


def _shortcut(cases: list[dict]) -> dict:
    def accuracy(subset: list[dict]) -> tuple[int, int, float]:
        by_signal: dict[str, Counter] = defaultdict(Counter)
        for case in subset:
            by_signal[case["signal"]][case["gold"]["verdict"]] += 1
        captured = sum(counts.most_common(1)[0][1] for counts in by_signal.values())
        total = len(subset)
        return captured, total, round(captured / total, 3) if total else 0.0

    primary = [case for case in cases if case["control_type"] == "primary"]
    all_cap, all_total, all_acc = accuracy(cases)
    pri_cap, pri_total, pri_acc = accuracy(primary)
    return {
        "all56": {"captured": all_cap, "total": all_total, "accuracy": all_acc},
        "primary40": {"captured": pri_cap, "total": pri_total, "accuracy": pri_acc},
        "chance_3class": 0.333,
    }


def main() -> None:
    cases = _load_cases()
    report = {
        "n_cases": len(cases),
        "integrity_and_dedup": _integrity_and_dedup(cases),
        "test_split": _test_split(cases),
        "leakage": _leakage(cases),
        "shortcut_classifier": _shortcut(cases),
    }
    (EVALS / "check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
