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
from collections.abc import Callable
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


def _sha256(text: str) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cases() -> list[dict]:
    lines = (EVALS / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _load_final_gold() -> dict[str, str]:
    path = EVALS / "annotation" / "human_gold.jsonl"
    if not path.exists():
        return {}
    gold: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            gold[record["id"]] = record["final_verdict"]
    return gold


def _csv_text(dataset: str, index: int) -> str | None:
    """Re-read the article text from the source CSV by 0-based data-row position."""
    source = DATA / f"{dataset}.csv"
    if not source.exists():
        return None
    with source.open(encoding="utf-8") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            if position == index:
                return (row.get("text") or "").strip()
    return None


def _domain(url: str | None) -> str:
    if not url:
        return "(no-url)"
    from urllib.parse import urlsplit  # noqa: PLC0415

    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host or "(no-url)"


def _tokens(text: str) -> set[str]:
    return set(_normalize(text).split())


def _near_duplicates(cases: list[dict], threshold: float = 0.3) -> list[dict]:
    """Flag case pairs with high token-Jaccard overlap (exact-hash dedup misses these)."""
    token_sets = {case["id"]: _tokens(case["text"]) for case in cases}
    ids = [case["id"] for case in cases]
    pairs: list[dict] = []
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            a, b = token_sets[left], token_sets[right]
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard >= threshold:
                pairs.append({"pair": [left, right], "jaccard": round(jaccard, 3)})
    return sorted(pairs, key=lambda item: item["jaccard"], reverse=True)


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
    drift: list[str] = []
    for case in cases:
        if not case["text"].strip():
            empty.append(case["id"])
        hashes[case["text_sha256"]].append(case["id"])
        # Real integrity check: re-read the CSV row and confirm the frozen text still
        # matches. The old check only compared already-stored hashes, so a reordered
        # CSV row would silently swap the text under a TCxx id without failing.
        csv_text = _csv_text(case["dataset"], case["row"])
        if csv_text is None:
            continue
        if _sha256(_normalize(csv_text)) != case["text_sha256"]:
            drift.append(case["id"])
    duplicates = {digest: ids for digest, ids in hashes.items() if len(ids) > 1}
    near_dups = _near_duplicates(cases)
    return {
        "empty_text_items": empty,
        "csv_text_drift_items": drift,
        "duplicate_groups": list(duplicates.values()),
        "near_duplicate_pairs": near_dups,
        "passed": not empty and not duplicates and not drift,
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


def _shortcut(cases: list[dict], gold: dict[str, str] | None = None) -> dict:
    """Most-frequent-verdict baselines for several shortcut features.

    A low signal->verdict number alone is not evidence the corpus is non-trivial:
    the strongest shortcut here is the source domain (domain->verdict beats the
    detector signal), because the mt signal is perfectly collinear with one domain.
    We therefore report domain and signal+domain baselines alongside the signal one,
    against both the curator gold and the frozen final gold.
    """

    def label_of(case: dict) -> str:
        if gold is not None and case["id"] in gold:
            return gold[case["id"]]
        return case["gold"]["verdict"]

    def baseline(subset: list[dict], feature: Callable[[dict], str]) -> dict:
        groups: dict[str, Counter] = defaultdict(Counter)
        for case in subset:
            groups[feature(case)][label_of(case)] += 1
        captured = sum(counts.most_common(1)[0][1] for counts in groups.values())
        total = len(subset)
        return {"captured": captured, "total": total, "accuracy": round(captured / total, 3)}

    primary = [case for case in cases if case["control_type"] == "primary"]
    features = {
        "signal": lambda case: case["signal"],
        "domain": lambda case: _domain(case.get("source_url")),
        "signal+domain": lambda case: f"{case['signal']}|{_domain(case.get('source_url'))}",
    }
    return {
        "gold": "final" if gold else "curator",
        "all_cases": {name: baseline(cases, fn) for name, fn in features.items()},
        "primary": {name: baseline(primary, fn) for name, fn in features.items()},
        "chance_3class": 0.333,
    }


def main() -> None:
    cases = _load_cases()
    final_gold = _load_final_gold()
    report = {
        "n_cases": len(cases),
        "integrity_and_dedup": _integrity_and_dedup(cases),
        "test_split": _test_split(cases),
        "leakage": _leakage(cases),
        "shortcut_classifier_curator": _shortcut(cases),
        "shortcut_classifier_final": _shortcut(cases, final_gold) if final_gold else None,
    }
    (EVALS / "check_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
