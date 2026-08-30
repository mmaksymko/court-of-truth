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
        # Prefix evidence ids with "e" (ep1, ea1, en1) so they live in a namespace
        # distinct from claim ids (p1, a1, n1). Sharing one namespace let the Judge
        # silently swap an evidence id for a claim id (both "p1"), a confusion the
        # cross-reference checks could not catch.
        prefix = f"e{argument.role[0]}"
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
