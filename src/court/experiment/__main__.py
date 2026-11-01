"""CLI to run the tribunal experiment over the evaluation corpus.

Wires the real forensic registry (report per item) and the real LLM client per
mode, runs every configuration and writes runs.jsonl + summary.json. The live
run needs a configured OPENAI key; without it the LLM client is unavailable and
the command exits with a clear message (the forensic half still loads).

Example:
    uv run --no-sync python -m court.experiment \
        --cases evals/cases.jsonl --gold evals/annotation/human_gold.jsonl \
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
from court.experiment.runner import append_record_jsonl
from court.forensics.registry import load_registry
from court.forensics.report import analyze
from court.forensics.schemas import AnalyzeRequest
from court.ingest.snapshot import PageArchive
from court.tribunal.errors import TribunalError
from court.tribunal.llm import LLMClient, build_llm

if TYPE_CHECKING:
    from court.experiment.modes import ModeConfig
    from court.experiment.runner import ItemInput, RunRecord
    from court.forensics.schemas import ForensicReport

logger = logging.getLogger("court.experiment")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="court.experiment")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases.jsonl"))
    parser.add_argument("--gold", type=Path, default=Path("evals/annotation/human_gold.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("evals/runs"))
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Run only these item IDs (for example: TC25 TC28 TC31).",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--swapped", action="store_true")
    parser.add_argument(
        "--sequential-parties",
        action="store_true",
        help="Run prosecutor and advocate sequentially to reduce peak token rate.",
    )
    parser.add_argument("--party-delay-seconds", type=float, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the selected matrix without loading models or making API calls.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    items = load_items(args.cases)
    if args.ids:
        requested = set(args.ids)
        known = {item.id for item in items}
        unknown = sorted(requested - known)
        if unknown:
            raise SystemExit(f"unknown item IDs: {', '.join(unknown)}")
        items = [item for item in items if item.id in requested]
    modes = [MODES[name] for name in args.modes]
    if args.dry_run:
        variants = args.repeats + int(args.swapped)
        logger.info(
            "DRY RUN: %d items x %d modes x %d variants = %d full deliberations",
            len(items),
            len(modes),
            variants,
            len(items) * len(modes) * variants,
        )
        logger.info("items: %s", " ".join(item.id for item in items))
        logger.info("modes: %s", " ".join(mode.name for mode in modes))
        return

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

    gold = load_gold(args.gold)
    logger.info("running %d items x %d modes (repeats=%d)", len(items), len(modes), args.repeats)

    # Every paid run gets a new output directory. Refusing to reuse a populated
    # path protects the raw records and their exact prompt-era provenance.
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is not empty; choose a new --out: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    runs_path = args.out / "runs.jsonl"
    runs_path.write_text("", encoding="utf-8")

    async def run() -> list[RunRecord]:
        archive = None
        if settings.evidence_cache_dir is not None:
            archive = PageArchive(settings.evidence_cache_dir, settings)
        try:
            return await run_experiment(
                items,
                modes,
                report_for=report_for,
                llm_for=llm_for,
                repeats=args.repeats,
                include_swapped=args.swapped,
                on_record=lambda record: append_record_jsonl(runs_path, record),
                sequential_parties=args.sequential_parties,
                party_delay_s=args.party_delay_seconds,
                archive=archive,
            )
        finally:
            if archive is not None:
                await archive.aclose()

    records = asyncio.run(run())
    _, summary_path = write_outputs(args.out, records, gold)
    logger.info("wrote %s (%d records) and %s", runs_path, len(records), summary_path)


if __name__ == "__main__":
    main()
