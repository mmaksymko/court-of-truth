"""Experiment runner for task 6.

Executes corpus items through the tribunal in a given mode and aggregates
per-mode metrics. Fully exercisable offline with a fake LLM; at G5 the same code
runs against the live model by supplying real report and client factories.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from court.experiment import metrics
from court.experiment.metrics import Calibration, Label, OutcomeCounts
from court.tribunal.deliberation import deliberate_mode
from court.tribunal.errors import TribunalError
from court.tribunal.telemetry import aggregate_usage, searches_as_dicts, usage_as_dict

if TYPE_CHECKING:
    from pathlib import Path

    from court.experiment.modes import ModeConfig
    from court.forensics.schemas import ForensicReport
    from court.tribunal.llm import LLMClient


@dataclass(frozen=True)
class ItemInput:
    id: str
    title: str
    text: str
    source_url: str | None = None
    gold: Label | None = None


@dataclass(frozen=True)
class RunRecord:
    item_id: str
    mode: str
    status: Literal["ok", "failed"]
    label: Label | None
    probabilities: dict[Label, float] | None
    error_code: str | None
    elapsed_s: float | None = None
    usage: dict[str, int] | None = None
    arguments: list[dict[str, object]] | None = None
    evidence: list[dict[str, object]] | None = None
    searches: list[dict[str, object]] | None = None
    variant: str | None = None


ReportFor = Callable[[ItemInput], Awaitable["ForensicReport"]]
LLMFor = Callable[["ModeConfig"], "LLMClient"]


async def run_item(  # noqa: PLR0913
    item: ItemInput,
    mode: ModeConfig,
    report: ForensicReport,
    llm: LLMClient,
    *,
    swap_order: bool = False,
    variant: str | None = None,
) -> RunRecord:
    # Telemetry (tokens, latency, the search trace) cannot be recovered after a paid
    # run, so it is captured here rather than logged and discarded.
    llm.enable_telemetry()
    start = time.perf_counter()
    try:
        arguments, evidence, verdict = await deliberate_mode(
            item.title,
            item.text,
            report,
            llm,
            adversarial=mode.adversarial,
            include_forensics=mode.include_forensics,
            source_url=item.source_url,
            swap_order=swap_order,
        )
    except TribunalError as exc:
        elapsed = time.perf_counter() - start
        calls = llm.drain_telemetry()
        return RunRecord(
            item.id,
            mode.name,
            "failed",
            None,
            None,
            exc.code,
            elapsed_s=elapsed,
            usage=usage_as_dict(aggregate_usage(calls)),
            searches=searches_as_dicts(calls),
            variant=variant,
        )
    elapsed = time.perf_counter() - start
    calls = llm.drain_telemetry()
    return RunRecord(
        item.id,
        mode.name,
        "ok",
        verdict.label,
        verdict.probabilities.as_map(),
        None,
        elapsed_s=elapsed,
        usage=usage_as_dict(aggregate_usage(calls)),
        arguments=[argument.model_dump(mode="json") for argument in arguments],
        evidence=[record.model_dump(mode="json") for record in evidence],
        searches=searches_as_dicts(calls),
        variant=variant,
    )


async def run_repeats(  # noqa: PLR0913
    item: ItemInput,
    mode: ModeConfig,
    report: ForensicReport,
    llm: LLMClient,
    *,
    repeats: int = 1,
    include_swapped: bool = False,
) -> list[RunRecord]:
    """Repeat runs and an optional order-swapped run for stability analysis.

    Each pass re-runs the full deliberation, so this captures end-to-end variance
    and order sensitivity together; the variant tag records which pass produced
    each record.
    """
    records: list[RunRecord] = []
    for index in range(1, repeats + 1):
        records.append(await run_item(item, mode, report, llm, variant=f"AB#{index}"))
    if include_swapped:
        records.append(await run_item(item, mode, report, llm, swap_order=True, variant="BA#1"))
    return records


def write_records_jsonl(path: Path, records: Sequence[RunRecord]) -> None:
    """Persist run records as one JSON object per line for downstream analysis."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


async def run_corpus(
    items: Sequence[ItemInput],
    modes: Sequence[ModeConfig],
    *,
    report_for: ReportFor,
    llm_for: LLMFor,
) -> list[RunRecord]:
    clients = {mode.name: llm_for(mode) for mode in modes}
    try:
        records: list[RunRecord] = []
        for item in items:
            report = await report_for(item)
            for mode in modes:
                records.append(await run_item(item, mode, report, clients[mode.name]))
        return records
    finally:
        for client in clients.values():
            await client.aclose()


@dataclass(frozen=True)
class ModeSummary:
    mode: str
    outcomes: OutcomeCounts
    macro_f1: float
    brier: float
    calibration: Calibration
    itt_accuracy: float


def evaluate(
    records: Sequence[RunRecord],
    gold: dict[str, Label],
) -> dict[str, ModeSummary]:
    summaries: dict[str, ModeSummary] = {}
    modes = sorted({record.mode for record in records})
    for mode in modes:
        labelled = [r for r in records if r.mode == mode and r.item_id in gold]
        completed = [r for r in labelled if r.status == "ok" and r.label and r.probabilities]
        gold_labels: list[Label] = [gold[r.item_id] for r in completed]
        predicted: list[Label] = [r.label for r in completed if r.label]
        distributions = [r.probabilities for r in completed if r.probabilities]
        correct = sum(gold[r.item_id] == r.label for r in completed)
        summaries[mode] = ModeSummary(
            mode=mode,
            outcomes=OutcomeCounts(
                total=len(labelled),
                completed=len(completed),
                failures=sum(r.status == "failed" for r in labelled),
            ),
            macro_f1=metrics.macro_f1(gold_labels, predicted),
            brier=metrics.multiclass_brier(gold_labels, distributions),
            calibration=metrics.calibration(gold_labels, predicted, distributions),
            itt_accuracy=correct / len(labelled) if labelled else 0.0,
        )
    return summaries
