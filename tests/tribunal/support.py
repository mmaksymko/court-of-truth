from court.forensics.schemas import ForensicReport, RiskSummary, SkippedResult


def report() -> ForensicReport:
    return ForensicReport(
        title_present=True,
        text_chars=100,
        results=[
            SkippedResult(id="ai_generated", scope="body", reason="too_short"),
            SkippedResult(id="jeansa", scope="body", reason="too_short"),
        ],
        risk=RiskSummary(flagged_count=0, flagged_ids=[], low_confidence_ids=[]),
    )
