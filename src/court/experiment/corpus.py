"""Load the evaluation corpus and gold labels into the experiment runner.

`cases.jsonl` provides the article inputs; `gold_final.jsonl` provides the
arbitrated verdicts. Both are produced under evals/ and are decoupled from the
frozen code, so this loader is the single seam that binds them to `run_corpus`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, get_args

from court.experiment.metrics import Label
from court.experiment.runner import ItemInput

if TYPE_CHECKING:
    from pathlib import Path

_LABELS = frozenset(get_args(Label))


def load_items(cases_path: Path) -> list[ItemInput]:
    items: list[ItemInput] = []
    for line in cases_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        items.append(
            ItemInput(
                id=case["id"],
                title=case.get("title", ""),
                text=case["text"],
                source_url=case.get("source_url"),
                published=case.get("published"),
            )
        )
    return items


def load_gold(gold_path: Path, *, key: str = "final_verdict") -> dict[str, Label]:
    gold: dict[str, Label] = {}
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        verdict = record[key]
        if verdict not in _LABELS:
            raise ValueError(f"{record.get('id')}: unknown verdict {verdict!r}")
        gold[record["id"]] = verdict
    return gold
