"""Freeze deterministic forensic reports for a later judge-only refresh.

The historical run did not persist the full report object. This command rebuilds
it locally once from the unchanged article text and detector artifacts, then the
paid rejudge reads the cache and invokes only the judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from court.config import Settings
from court.forensics.registry import load_registry
from court.forensics.report import analyze
from court.forensics.schemas import AnalyzeRequest

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "evals" / "cases.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-ids", nargs="+", default=[])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.out.exists():
        raise SystemExit(f"output already exists and will not be overwritten: {args.out}")
    excluded = set(args.exclude_ids)
    settings = Settings()
    registry = load_registry(settings.artifacts_dir)
    cases = [
        json.loads(line)
        for line in CASES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for case in cases:
            if case["id"] in excluded:
                continue
            report = analyze(
                AnalyzeRequest(title=case["title"], text=case["text"]),
                registry,
                settings.low_confidence_margin,
            )
            handle.write(
                json.dumps(
                    {
                        "item_id": case["id"],
                        "case_text_sha256": case["text_sha256"],
                        "report": report.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"wrote {args.out} ({len(cases) - len(excluded)} reports)")


if __name__ == "__main__":
    main()
