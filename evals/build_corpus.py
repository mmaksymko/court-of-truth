"""Serialize the 56 curated tribunal cases into evals/cases.jsonl + a manifest.

Single source of truth is the two Markdown tables in docs/completion-plan/
(13-curated-corpus.md for case metadata, 14-gold-labels.md for the gold verdict).
The article title, body and URL are pulled from the source CSV row by integer
position (0-based over data rows, header excluded), matching how the corpus was
curated. Nothing here is invented: the gold verdicts are the single-curator
labels recorded in 14-gold-labels.md and are tagged as such.

Run: uv run --no-sync python evals/build_corpus.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "docs" / "completion-plan"
DATA = ROOT / "data"
OUT_DIR = ROOT / "evals"

DATASET_CSV = {
    "jeansa": DATA / "jeansa.csv",
    "clickbait": DATA / "clickbait.csv",
    "ai_generated": DATA / "ai_generated.csv",
    "mt_translation": DATA / "mt_translation.csv",
}
# Datasets without a split column were drawn from a held-out tail and carry a
# detector-leakage caveat (see 13-curated-corpus.md).
TAIL_DATASETS = {"ai_generated", "mt_translation"}


def _table_rows(path: Path, header_key: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or cells[0] in {header_key, "---"} or set(cells[0]) <= {"-"}:
            continue
        if re.match(r"^TC\d+$", cells[0]):
            rows.append(cells)
    return rows


def _dataset_row(token: str) -> tuple[str, int]:
    name, _, row = token.partition("#")
    return name, int(row)


def _dash(value: str) -> str | None:
    value = value.strip()
    return None if value in {"", "-"} else value


def _csv_row(dataset: str, index: int) -> dict[str, str]:
    with DATASET_CSV[dataset].open(encoding="utf-8") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            if position == index:
                return row
    raise SystemExit(f"row {index} not found in {dataset}")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build() -> None:
    corpus = {cells[0]: cells for cells in _table_rows(PLAN / "13-curated-corpus.md", "item id")}
    gold = {cells[0]: cells for cells in _table_rows(PLAN / "14-gold-labels.md", "item id")}
    missing = corpus.keys() ^ gold.keys()
    if missing:
        raise SystemExit(f"corpus/gold id mismatch: {sorted(missing)}")

    cases: list[dict[str, object]] = []
    for item_id in sorted(corpus, key=lambda name: int(name[2:])):
        c = corpus[item_id]
        g = gold[item_id]
        if c[1] != g[1]:
            raise SystemExit(f"{item_id}: dataset#row differs between tables ({c[1]} vs {g[1]})")
        dataset, row = _dataset_row(c[1])
        record = _csv_row(dataset, row)
        text = (record.get("text") or "").strip()
        title = (record.get("title") or "").strip()
        cases.append(
            {
                "id": item_id,
                "dataset": dataset,
                "row": row,
                "signal": c[4],
                "control_type": c[7],
                "search_necessity": c[5],
                "checkable_claim": _dash(c[6]),
                "url_present": c[3].lower() == "yes",
                "title": title,
                "text": text,
                "source_url": _dash(record.get("source_url", "") or ""),
                "dataset_label": record.get("label", ""),
                "text_sha256": _sha256(_normalize(text)),
                "provenance": {
                    "split": "tail" if dataset in TAIL_DATASETS else "test",
                    "leakage_caveat": dataset in TAIL_DATASETS,
                },
                "gold": {
                    "verdict": g[4],
                    "confidence": float(g[5]),
                    "key_external_fact": _dash(g[6]),
                    "evidence_url": _dash(g[7]),
                    "rationale": g[8],
                    "annotator": "curator-1",
                    "method": "single-curator web search (author); not yet double-annotated",
                },
            }
        )

    OUT_DIR.mkdir(exist_ok=True)
    cases_path = OUT_DIR / "cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    verdicts = _counter(case["gold"]["verdict"] for case in cases)  # type: ignore[index]
    signals = _counter(str(case["signal"]) for case in cases)
    manifest = {
        "n_cases": len(cases),
        "verdict_counts": verdicts,
        "signal_counts": signals,
        "search_required": sum(c["search_necessity"] == "search-required" for c in cases),
        "leakage_caveat_items": sum(c["provenance"]["leakage_caveat"] for c in cases),  # type: ignore[index]
        "cases_sha256": _file_sha256(cases_path),
        "dataset_csv_sha256": {name: _file_sha256(path) for name, path in DATASET_CSV.items()},
        "annotation_status": "single-curator; second independent annotation + arbitration pending",
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {cases_path} ({len(cases)} cases)")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _counter(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:  # type: ignore[attr-defined]
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    build()
