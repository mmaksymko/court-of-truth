"""Emit a blind annotation sheet + an empty label template for annotator 2.

The blind sheet exposes only what a second annotator needs (id, title, text,
source_url, search_necessity) and hides the detector signal, control type and
curator verdict, so the second annotation is genuinely independent.

Run: uv run --no-sync python evals/make_annotation_sheet.py
"""

from __future__ import annotations

import json
from pathlib import Path

EVALS = Path(__file__).resolve().parent
OUT = EVALS / "annotation"

BLIND_FIELDS = ("id", "title", "text", "source_url", "search_necessity")


def main() -> None:
    cases = [
        json.loads(line)
        for line in (EVALS / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    OUT.mkdir(exist_ok=True)

    blind = OUT / "blind_sheet.jsonl"
    with blind.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {field: case[field] for field in BLIND_FIELDS},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    template = OUT / "annotator2.template.jsonl"
    with template.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(
                    {
                        "id": case["id"],
                        "verdict": "",
                        "confidence": None,
                        "search_needed": None,
                        "key_claims": "",
                        "acceptable_sources": [],
                        "rationale": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"wrote {blind} ({len(cases)} blind items)")
    print(f"wrote {template} (empty template — do not fill programmatically)")


if __name__ == "__main__":
    main()
