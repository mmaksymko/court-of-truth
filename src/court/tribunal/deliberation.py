import asyncio

from court.forensics.schemas import ForensicReport
from court.tribunal import prompts
from court.tribunal.errors import TribunalError
from court.tribunal.evidence import build_evidence
from court.tribunal.llm import LLMClient
from court.tribunal.schemas import Argument, Deliberation, EvidenceRecord, Verdict


async def deliberate_mode(  # noqa: PLR0913
    title: str,
    text: str,
    report: ForensicReport,
    llm: LLMClient,
    *,
    adversarial: bool = True,
    include_forensics: bool = True,
    source_url: str | None = None,
    swap_order: bool = False,
) -> tuple[list[Argument], list[EvidenceRecord], Verdict]:
    """Mode-aware core: one neutral researcher or two adversarial sides, then Judge.

    Search-context reduction (B3) is applied at the client level via `build_llm`.
    ``swap_order`` reverses the order in which the parties are presented to the
    Judge, so the protocol can probe order sensitivity. Returns the raw arguments,
    verified evidence and the verdict.
    """
    research = prompts.research(
        title, text, report, source_url, include_forensics=include_forensics
    )
    if adversarial:
        try:
            async with asyncio.TaskGroup() as tasks:
                prosecution_task = tasks.create_task(llm.argue("prosecutor", research))
                defence_task = tasks.create_task(llm.argue("advocate", research))
        except* Exception as caught:
            raise caught.exceptions[0] from None
        arguments = [
            _validated_argument(prosecution_task.result(), report),
            _validated_argument(defence_task.result(), report),
        ]
        cases = [("prosecution", arguments[0]), ("defence", arguments[1])]
    else:
        arguments = [_validated_argument(await llm.argue("neutral", research), report)]
        cases = [("research", arguments[0])]
    evidence = build_evidence(arguments)
    presented = list(reversed(cases)) if swap_order else cases
    verdict = await llm.judge(
        prompts.judge(
            title,
            text,
            report,
            presented,
            evidence,
            source_url=source_url,
            include_forensics=include_forensics,
        )
    )
    claim_ids = {claim.id for argument in arguments for claim in argument.claims}
    _validate_verdict(verdict, evidence, report, claim_ids)
    return arguments, evidence, verdict


async def deliberate(
    title: str,
    text: str,
    report: ForensicReport,
    llm: LLMClient,
    *,
    source_url: str | None = None,
) -> Deliberation:
    arguments, evidence, verdict = await deliberate_mode(
        title, text, report, llm, adversarial=True, include_forensics=True, source_url=source_url
    )
    prosecution, defence = arguments
    return Deliberation(
        prosecutor=prosecution, advocate=defence, evidence=evidence, verdict=verdict
    )


def _validate_verdict(
    verdict: Verdict,
    evidence: list[EvidenceRecord],
    report: ForensicReport,
    claim_ids: set[str],
) -> None:
    evidence_ids = {record.id for record in evidence}
    if set(verdict.used_evidence_ids) - evidence_ids:
        raise TribunalError(
            502,
            "tribunal_invalid_output",
            "verdict cited evidence absent from the deliberation",
        )
    if {objection.claim_id for objection in verdict.objections} - claim_ids:
        raise TribunalError(
            502,
            "tribunal_invalid_output",
            "verdict objected to a claim absent from the arguments",
        )
    detector_ids = {result.id for result in report.results}
    if set(verdict.cited_detectors) - detector_ids:
        raise TribunalError(
            502,
            "tribunal_invalid_output",
            "verdict cited detectors absent from the forensic report",
        )


def _validated_argument(argument: Argument, report: ForensicReport) -> Argument:
    detector_ids = {result.id for result in report.results}
    unknown = set(argument.cited_detectors) - detector_ids
    if unknown:
        raise TribunalError(
            502,
            "tribunal_invalid_output",
            "tribunal cited detectors absent from the forensic report",
        )
    claim_ids = {claim.id for claim in argument.claims}
    if any(source.claim_id not in claim_ids for source in argument.sources):
        raise TribunalError(
            502,
            "tribunal_invalid_output",
            "tribunal attached a source to a claim absent from the argument",
        )
    return argument
