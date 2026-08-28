import re
from collections.abc import Sequence

from court.forensics.schemas import ForensicReport, OkResult
from court.tribunal.schemas import Argument, EvidenceRecord

_MAX_ARTICLE_CHARS = 24_000


def _detectors(report: ForensicReport, include_forensics: bool) -> str:
    if not include_forensics:
        return "ДЕТЕКТОРИ:\n(вимкнено в цьому режимі)"
    return f"ДЕТЕКТОРИ:\n{_evidence(report)}"


def research(
    title: str,
    text: str,
    report: ForensicReport,
    source_url: str | None = None,
    *,
    include_forensics: bool = True,
) -> str:
    return f"{_article(title, text, source_url)}\n\n{_detectors(report, include_forensics)}"


def judge(  # noqa: PLR0913
    title: str,
    text: str,
    report: ForensicReport,
    cases: Sequence[tuple[str, Argument]],
    evidence: list[EvidenceRecord],
    *,
    source_url: str | None = None,
    include_forensics: bool = True,
) -> str:
    rendered_cases = "\n\n".join(_case(name, argument) for name, argument in cases)
    return (
        f"{_article(title, text, source_url)}\n\n{_detectors(report, include_forensics)}\n\n"
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


def _article(title: str, text: str, source_url: str | None) -> str:
    safe_title = title.replace("<", " ").replace(">", " ").replace('"', "'")
    bounded = _bound(text)
    bounded = re.sub(r"</\s*article", "< article", bounded, flags=re.IGNORECASE)
    origin = f"\nSOURCE URL: {source_url}" if source_url else ""
    return f'<article title="{safe_title}">{origin}\n{bounded}\n</article>'


def _bound(text: str) -> str:
    if len(text) <= _MAX_ARTICLE_CHARS:
        return text
    half = (_MAX_ARTICLE_CHARS - 40) // 2
    return f"{text[:half]}\n[...матеріал скорочено...]\n{text[-half:]}"


def _evidence(report: ForensicReport) -> str:
    lines = []
    for result in report.results:
        if isinstance(result, OkResult):
            verdict = "спрацював" if result.flagged else "поріг не перевищено"
            lines.append(f"- {result.id}: {verdict}, p={result.probability:.2f}")
        else:
            lines.append(f"- {result.id}: пропущено ({result.reason})")
    return "\n".join(lines) or "(немає)"


def _case(name: str, argument: Argument) -> str:
    data = argument.model_dump_json()
    data = re.sub(r"</\s*case", "< case", data, flags=re.IGNORECASE)
    return f'<case role="{name}">\n{data}\n</case>'
