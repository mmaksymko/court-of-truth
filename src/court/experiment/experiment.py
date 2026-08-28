"""End-to-end experiment glue: run the corpus through every mode and summarise.

Binds the loaded corpus and gold labels to `run_repeats`/`evaluate`, writes the
per-run JSONL and a machine-readable metrics summary. Fully exercisable offline
with a fake LLM and a fake report factory; the live run only swaps in the real
forensic registry and LLM client.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from court.experiment.runner import (
    ItemInput,
    LLMFor,
    ReportFor,
    RunRecord,
    evaluate,
    run_repeats,
    write_records_jsonl,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from court.experiment.metrics import Label
    from court.experiment.modes import ModeConfig


async def run_experiment(  # noqa: PLR0913
    items: Sequence[ItemInput],
    modes: Sequence[ModeConfig],
    *,
    report_for: ReportFor,
    llm_for: LLMFor,
    repeats: int = 1,
    include_swapped: bool = False,
) -> list[RunRecord]:
    """Run every item through every mode (with repeats/order-swap), one report per item."""
    clients = {mode.name: llm_for(mode) for mode in modes}
    try:
        records: list[RunRecord] = []
        for item in items:
            report = await report_for(item)
            for mode in modes:
                records.extend(
                    await run_repeats(
                        item,
                        mode,
                        report,
                        clients[mode.name],
                        repeats=repeats,
                        include_swapped=include_swapped,
                    )
                )
        return records
    finally:
        for client in clients.values():
            await client.aclose()


def summarize(records: Sequence[RunRecord], gold: dict[str, Label]) -> dict[str, object]:
    return {mode: asdict(summary) for mode, summary in evaluate(records, gold).items()}


def write_outputs(
    out_dir: Path, records: Sequence[RunRecord], gold: dict[str, Label]
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "summary.json"
    write_records_jsonl(runs_path, records)
    summary_path.write_text(
        json.dumps(summarize(records, gold), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return runs_path, summary_path
