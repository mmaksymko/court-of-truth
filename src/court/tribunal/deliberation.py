import asyncio
import logging

from court.forensics.schemas import ForensicReport
from court.tribunal import prompts
from court.tribunal.evidence import build_evidence
from court.tribunal.llm import LLMClient
from court.tribunal.schemas import Argument, Deliberation, EvidenceRecord, Verdict

logger = logging.getLogger("court.tribunal")


async def deliberate_mode(  # noqa: PLR0913
    title: str,
    text: str,
    report: ForensicReport,
    llm: LLMClient,
    *,
    adversarial: bool = True,
    include_forensics: bool = True,
    source_url: str | None = None,
    published: str | None = None,
    swap_order: bool = False,
    sequential_parties: bool = False,
    party_delay_s: float = 0,
) -> tuple[list[Argument], list[EvidenceRecord], Verdict]:
    """Mode-aware core: one neutral researcher or two adversarial sides, then Judge.

    Search-context reduction (B3) is applied at the client level via `build_llm`.
    ``swap_order`` reverses the order in which the parties are presented to the
    Judge, so the protocol can probe order sensitivity. Returns the raw arguments,
    verified evidence and the verdict.
    """
    research = prompts.research(
        title, text, report, source_url, include_forensics=include_forensics, published=published
    )
    if adversarial:
        if sequential_parties:
            prosecution = await llm.argue("prosecutor", research)
            if party_delay_s > 0:
                await asyncio.sleep(party_delay_s)
            defence = await llm.argue("advocate", research)
            arguments = [
                _validated_argument(prosecution, report),
                _validated_argument(defence, report),
            ]
        else:
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
            published=published,
        )
    )
    claim_ids = {claim.id for argument in arguments for claim in argument.claims}
    _sanitize_verdict(verdict, evidence, report, claim_ids)
    return arguments, evidence, verdict


async def deliberate(  # noqa: PLR0913
    title: str,
    text: str,
    report: ForensicReport,
    llm: LLMClient,
    *,
    source_url: str | None = None,
    published: str | None = None,
) -> Deliberation:
    arguments, evidence, verdict = await deliberate_mode(
        title,
        text,
        report,
        llm,
        adversarial=True,
        include_forensics=True,
        source_url=source_url,
        published=published,
    )
    prosecution, defence = arguments
    return Deliberation(
        prosecutor=prosecution, advocate=defence, evidence=evidence, verdict=verdict
    )


def _sanitize_verdict(
    verdict: Verdict,
    evidence: list[EvidenceRecord],
    report: ForensicReport,
    claim_ids: set[str],
) -> None:
    """Drop dangling provenance references instead of failing the whole review.

    ``used_evidence_ids``, ``objections`` and ``cited_detectors`` are provenance
    metadata; they do not affect the scored label or probabilities. A single
    hallucinated id used to raise a 502 that, under the experiment's
    intention-to-treat rule, poisoned an otherwise correct verdict - and such slips
    grow more likely on the richest, hardest cases. We now discard the unknown
    references and keep the verdict, logging what was trimmed.
    """
    evidence_ids = {record.id for record in evidence}
    detector_ids = {result.id for result in report.results}
    unknown_evidence = set(verdict.used_evidence_ids) - evidence_ids
    unknown_objections = {objection.claim_id for objection in verdict.objections} - claim_ids
    unknown_detectors = set(verdict.cited_detectors) - detector_ids
    if unknown_evidence or unknown_objections or unknown_detectors:
        logger.warning(
            "verdict provenance trimmed: evidence=%s objections=%s detectors=%s",
            sorted(unknown_evidence),
            sorted(unknown_objections),
            sorted(unknown_detectors),
        )
    verdict.used_evidence_ids = [
        record_id for record_id in verdict.used_evidence_ids if record_id in evidence_ids
    ]
    verdict.objections = [
        objection for objection in verdict.objections if objection.claim_id in claim_ids
    ]
    verdict.cited_detectors = [
        detector_id for detector_id in verdict.cited_detectors if detector_id in detector_ids
    ]


def _validated_argument(argument: Argument, report: ForensicReport) -> Argument:
    """Trim a party's dangling detector citations and orphaned sources in place.

    Same rationale as the verdict sanitizer: an argument's factual thesis and
    claims stand on their own, so a citation slip should shrink the provenance, not
    kill the deliberation with a 502.
    """
    detector_ids = {result.id for result in report.results}
    unknown_detectors = set(argument.cited_detectors) - detector_ids
    claim_ids = {claim.id for claim in argument.claims}
    orphaned = [source for source in argument.sources if source.claim_id not in claim_ids]
    if unknown_detectors or orphaned:
        logger.warning(
            "argument provenance trimmed: detectors=%s orphaned_sources=%s",
            sorted(unknown_detectors),
            sorted(str(source.url) for source in orphaned),
        )
    argument.cited_detectors = [
        detector_id for detector_id in argument.cited_detectors if detector_id in detector_ids
    ]
    argument.sources = [source for source in argument.sources if source.claim_id in claim_ids]
    return argument
