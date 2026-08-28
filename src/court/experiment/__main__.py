"""CLI to run the tribunal experiment over the evaluation corpus.

Wires the real forensic registry (report per item) and the real LLM client per
mode, runs every configuration and writes runs.jsonl + summary.json. The live
run needs a configured OPENAI key; without it the LLM client is unavailable and
the command exits with a clear message (the forensic half still loads).

Example:
    uv run --no-sync python -m court.experiment \
        --cases evals/cases.jsonl --gold evals/annotation/gold_final.jsonl \
        --out evals/runs --repeats 3 --swapped
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from court.config import Settings
from court.experiment.corpus import load_gold, load_items
from court.experiment.experiment import run_experiment, write_outputs
from court.experiment.modes import MODES
from court.forensics.registry import load_registry
from court.forensics.report import analyze
from court.forensics.schemas import AnalyzeRequest
from court.tribunal.errors import TribunalError
from court.tribunal.llm import LLMClient, build_llm

if TYPE_CHECKING:
    from court.experiment.modes import ModeConfig
    from court.experiment.runner import ItemInput
    from court.forensics.schemas import ForensicReport

logger = logging.getLogger("court.experiment")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="court.experiment")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("evals/annotation/gold_final.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("evals/runs"))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--swapped", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    settings = Settings()
    registry = load_registry(settings.artifacts_dir)

    async def report_for(item: ItemInput) -> ForensicReport:
        request = AnalyzeRequest(title=item.title, text=item.text)
        return analyze(request, registry, settings.low_confidence_margin)

    def llm_for(mode: ModeConfig) -> LLMClient:
        client = build_llm(
            settings,
            search_context=mode.search_context,
            include_forensics=mode.include_forensics,
            adversarial=mode.adversarial,
        )
        if client is None:
            raise TribunalError(503, "tribunal_unavailable", "OPENAI key not configured")
        return client

    items = load_items(args.cases)
    gold = load_gold(args.gold)
    modes = [MODES[name] for name in args.modes]
    logger.info("running %d items x %d modes (repeats=%d)", len(items), len(modes), args.repeats)

    records = asyncio.run(
        run_experiment(
            items,
            modes,
            report_for=report_for,
            llm_for=llm_for,
            repeats=args.repeats,
            include_swapped=args.swapped,
        )
    )
    runs_path, summary_path = write_outputs(args.out, records, gold)
    logger.info("wrote %s (%d records) and %s", runs_path, len(records), summary_path)


if __name__ == "__main__":
    main()
