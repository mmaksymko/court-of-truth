from collections.abc import Sequence

from court.tribunal.schemas import Argument, EvidenceRecord


def build_evidence(arguments: Sequence[Argument]) -> list[EvidenceRecord]:
    """Assemble stable, program-verified evidence records from the parties' sources.

    Accepts one argument (neutral mode) or two (adversarial). The Judge never
    authors evidence; it only references these ids. Each surviving source already
    passed the search-provenance guard in ``provenance.py``.
    """
    records: list[EvidenceRecord] = []
    for argument in arguments:
        prefix = argument.role[0]
        for index, source in enumerate(argument.sources, start=1):
            records.append(
                EvidenceRecord(
                    id=f"{prefix}{index}",
                    side=argument.role,
                    claim_id=source.claim_id,
                    url=source.url,
                    title=source.title,
                    excerpt=source.excerpt,
                    supports=source.supports,
                )
            )
    return records
