import re
from collections.abc import Sequence

from court.forensics.schemas import ForensicReport, OkResult
from court.tribunal.schemas import Argument, EvidenceRecord

_MAX_ARTICLE_CHARS = 24_000


def _detectors(report: ForensicReport, include_forensics: bool) -> str:
    if not include_forensics:
        return "ДЕТЕКТОРИ:\n(вимкнено в цьому режимі)"
    return f"ДЕТЕКТОРИ:\n{_evidence(report)}"


def research(  # noqa: PLR0913
    title: str,
    text: str,
    report: ForensicReport,
    source_url: str | None = None,
    *,
    include_forensics: bool = True,
    published: str | None = None,
) -> str:
    return (
        f"{_article(title, text, source_url, published)}\n\n"
        f"{_detectors(report, include_forensics)}"
    )


def judge(  # noqa: PLR0913
    title: str,
    text: str,
    report: ForensicReport,
    cases: Sequence[tuple[str, Argument]],
    evidence: list[EvidenceRecord],
    *,
    source_url: str | None = None,
    include_forensics: bool = True,
    published: str | None = None,
) -> str:
    rendered_cases = "\n\n".join(_case(name, argument) for name, argument in cases)
    return (
        f"{_article(title, text, source_url, published)}\n\n"
        f"{_detectors(report, include_forensics)}\n\n"
        f"ДОКАЗИ:\n{_records(evidence)}\n\n{rendered_cases}"
    )


def _records(evidence: list[EvidenceRecord]) -> str:
    lines = [
        f"- [{record.id}] {record.url} (сторона {record.side}): {_sanitize(record.supports)}"
        for record in evidence
    ]
    return "\n".join(lines) or "(немає)"


def _sanitize(text: str) -> str:
    return re.sub(r"</\s*(article|case)", r"< \1", text, flags=re.IGNORECASE)


def _article(
    title: str, text: str, source_url: str | None, published: str | None = None
) -> str:
    safe_title = title.replace("<", " ").replace(">", " ").replace('"', "'")
    bounded = _bound(text)
    bounded = re.sub(r"</\s*article", "< article", bounded, flags=re.IGNORECASE)
    origin = f"\nSOURCE URL: {source_url}" if source_url else ""
    # Publication period (quarter bucket) anchors search and the judge's
    # "as of publication date" test to the article's real time window, so agents
    # stop inventing precise dates they cannot verify.
    when = f"\nПЕРІОД ПУБЛІКАЦІЇ: {_sanitize_meta(published)}" if published else ""
    return f'<article title="{safe_title}">{origin}{when}\n{bounded}\n</article>'


def _sanitize_meta(value: str) -> str:
    return value.replace("<", " ").replace(">", " ").replace('"', "'")


def _bound(text: str) -> str:
    if len(text) <= _MAX_ARTICLE_CHARS:
        return text
    # Keep more of the tail than the head: ad-disclosure markers ("На правах
    # реклами", "Матеріал оплачено", "за підтримки") cluster at the end. The full
    # text is still analysed by the detectors, so their signals survive truncation.
    budget = _MAX_ARTICLE_CHARS - 60
    head = int(budget * 0.55)
    tail = budget - head
    return (
        f"{text[:head]}\n"
        "[...середину скорочено; повний текст проаналізовано детекторами...]\n"
        f"{text[-tail:]}"
    )


def _evidence(report: ForensicReport) -> str:
    # Deliberately no raw probability: a value like "p=0.87" reads as a calibrated,
    # precise degree of guilt and anchors the verdict, contrary to the independence
    # principle. We surface only the binary "look closer" signal plus the detector's
    # own low-confidence flag and caveats, which the raw rendering used to discard.
    lines = []
    for result in report.results:
        if isinstance(result, OkResult):
            verdict = "варто придивитися" if result.flagged else "поріг не перевищено"
            caveat = " [надійність сигналу низька]" if result.low_confidence else ""
            note = f" ({'; '.join(result.caveats)})" if result.caveats else ""
            lines.append(f"- {result.id}: {verdict}{caveat}{note}")
        else:
            lines.append(f"- {result.id}: пропущено ({result.reason})")
    return "\n".join(lines) or "(немає)"


def _case(name: str, argument: Argument) -> str:
    data = argument.model_dump_json()
    data = re.sub(r"</\s*case", "< case", data, flags=re.IGNORECASE)
    return f'<case role="{name}">\n{data}\n</case>'
