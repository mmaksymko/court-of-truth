"""Build a blind, evidence-rich workbench for human gold annotation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

LABELS = ("reliable", "questionable", "unreliable")
MODES = ("F", "B2", "B3")
RECOMMENDATION_MODES = ("F", "B2")

_CAUTION_MARKERS = (
    "без зовніш",
    "не вдалося",
    "не можна підтверд",
    "не підтвердж",
    "не перевір",
    "не довод",
    "не встановлено",
    "обмежен",
    "слабк",
    "супереч",
    "оманлив",
    "однобіч",
    "змішан",
    "потребує перевір",
    "залишається заяв",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_static_workspace(root: Path) -> dict[str, Any]:
    """Load immutable corpus, run and prior-annotation artifacts."""
    cases = {row["id"]: row for row in load_jsonl(root / "evals/cases.jsonl")}
    runs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(root / "evals/runs/runs.jsonl"):
        runs[row["item_id"]][row["mode"]] = row

    second = {
        row["id"]: row
        for row in load_jsonl(root / "evals/annotation/annotator2.fable.jsonl")
    }
    final = {
        row["id"]: row for row in load_jsonl(root / "evals/annotation/gold_final.jsonl")
    }
    arbitration = {
        row["id"]: row
        for row in load_jsonl(root / "evals/annotation/arbitration.jsonl")
    }
    adjudication = {
        row["id"]: row
        for row in load_jsonl(root / "evals/annotation/adjudication_high.jsonl")
    }
    recovered_path = root / "evals/annotation/recovered_source_urls.json"
    recovered = (
        json.loads(recovered_path.read_text(encoding="utf-8"))
        if recovered_path.exists()
        else {}
    )
    replacement_path = root / "evals/annotation/replacement_reviews.json"
    replacement_reviews = (
        {
            row["id"]: row
            for row in json.loads(replacement_path.read_text(encoding="utf-8"))
        }
        if replacement_path.exists()
        else {}
    )
    return {
        "cases": cases,
        "runs": dict(runs),
        "second": second,
        "final": final,
        "arbitration": arbitration,
        "adjudication": adjudication,
        "recovered": recovered,
        "replacement_reviews": replacement_reviews,
    }


def display_title(case: dict[str, Any]) -> str:
    title = str(case.get("title") or "").strip()
    if title:
        return title
    text = str(case.get("text") or "").strip()
    first_line = text.splitlines()[0] if text else "Без заголовка"
    return first_line[:180]


def source_info(workspace: dict[str, Any], item_id: str) -> dict[str, str | bool]:
    case = workspace["cases"][item_id]
    original = str(case.get("source_url") or "").strip()
    if _is_http_url(original):
        return {"url": original, "provenance": "dataset", "is_original": True}
    recovered = workspace["recovered"].get(item_id, {})
    url = str(recovered.get("url") or "").strip()
    return {
        "url": url,
        "provenance": str(recovered.get("provenance") or "missing"),
        "is_original": bool(recovered.get("is_original", False)),
    }


def compile_case_analysis(workspace: dict[str, Any], item_id: str) -> dict[str, Any]:
    """Compile role arguments without exposing any system or prior-gold label."""
    case = workspace["cases"][item_id]
    replacement_review = workspace.get("replacement_reviews", {}).get(item_id)
    if replacement_review:
        return _compile_replacement_analysis(workspace, item_id, replacement_review)
    role_material = {
        "advocate": _collect_role_material(workspace, item_id, "advocate"),
        "prosecutor": _collect_role_material(workspace, item_id, "prosecutor"),
    }
    shared_sources = _shared_verification_sources(workspace, item_id, role_material)
    cautious = _caution_arguments(role_material)
    if case.get("search_necessity") == "search-required":
        cautious.insert(
            0,
            "Ключове твердження потребує зовнішньої перевірки; сам текст статті "
            "не є незалежним підтвердженням власних заяв.",
        )
    if not source_info(workspace, item_id)["is_original"]:
        cautious.append(
            "Оригінальний URL не був збережений у корпусі; відновлене посилання "
            "потрібно звірити з текстом перед остаточним рішенням."
        )

    reliable = {
        "criterion": (
            "Обирайте reliable, якщо ключові факти підтверджуються якісними "
            "незалежними джерелами, а суттєвих спростувань або оманливого "
            "обрамлення немає."
        ),
        "arguments": role_material["advocate"]["theses"],
        "claims": role_material["advocate"]["claims"],
        "sources": role_material["advocate"]["sources"],
    }
    questionable = {
        "criterion": (
            "Обирайте questionable, якщо ключові твердження перевірені лише "
            "частково, джерела залежні або слабкі, чи правдиві факти подано в "
            "істотно оманливому контексті."
        ),
        "arguments": _dedupe(cautious),
        "claims": _questionable_claims(role_material),
        "sources": shared_sources,
    }
    unreliable = {
        "criterion": (
            "Обирайте unreliable лише за наявності надійного спростування, "
            "фабрикації ключових фактів або систематично оманливого матеріалу."
        ),
        "arguments": role_material["prosecutor"]["theses"],
        "claims": role_material["prosecutor"]["claims"],
        "sources": role_material["prosecutor"]["sources"],
    }
    for position in (reliable, questionable, unreliable):
        if not position["arguments"]:
            position["arguments"] = [
                "Для цієї позиції окремого сильного аргументу в зібраних "
                "матеріалах немає; не обирайте її лише через відсутність доказів."
            ]
    return {
        "id": item_id,
        "case": case,
        "title": display_title(case),
        "source": source_info(workspace, item_id),
        "positions": {
            "reliable": reliable,
            "questionable": questionable,
            "unreliable": unreliable,
        },
        "verification_sources": shared_sources,
    }


def recommend_verdict(workspace: dict[str, Any], item_id: str) -> dict[str, Any]:
    """Recommend a label without consulting the previous gold or arbitration."""
    replacement_review = workspace.get("replacement_reviews", {}).get(item_id)
    if replacement_review:
        verdict = str(replacement_review["recommendation"])
        return {
            "verdict": verdict,
            "reason": str(replacement_review["reason"]),
            "votes": [{"source": "ручна вебперевірка", "verdict": verdict}],
            "counts": {
                label: int(label == verdict)
                for label in LABELS
            },
        }

    votes: list[dict[str, str]] = []
    for mode in RECOMMENDATION_MODES:
        label = str(workspace["runs"].get(item_id, {}).get(mode, {}).get("label") or "")
        if label in LABELS:
            votes.append({"source": mode, "verdict": label})
    second_label = str(workspace["second"].get(item_id, {}).get("verdict") or "")
    if second_label in LABELS:
        votes.append({"source": "сліпа друга анотація", "verdict": second_label})

    counts = {label: sum(vote["verdict"] == label for vote in votes) for label in LABELS}
    highest = max(counts.values(), default=0)
    leaders = [label for label in LABELS if counts[label] == highest]
    verdict = leaders[0] if len(leaders) == 1 else "questionable"
    if not votes:
        reason = (
            "Попередніх незалежних оцінок немає, тому рекомендовано обережний "
            "клас questionable до ручної перевірки джерел."
        )
    elif len(leaders) > 1:
        reason = (
            "Незалежні оцінки розділилися порівну; рекомендовано questionable, "
            "щоб не перетворювати невизначеність на категоричний висновок."
        )
    else:
        reason = (
            f"{highest} з {len(votes)} пошукових оцінок підтримують {verdict}. "
            "Попередній gold, арбітраж і режим B3 без вебпошуку в рекомендації "
            "не використовувалися."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "votes": votes,
        "counts": counts,
    }


def load_human_annotations(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(path)}


def load_human_instructions(path: Path) -> dict[str, dict[str, Any]]:
    """Load the latest working instruction per item from the audit history."""
    history_path = path.with_name(f"{path.stem}_history.jsonl")
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(history_path):
        item_id = str(row.get("id") or "")
        if item_id:
            latest[item_id] = row
    return latest


def save_human_annotation(  # noqa: PLR0913 - explicit audit fields are intentional
    output_path: Path,
    *,
    item_id: str,
    verdict: str,
    instruction: str,
    source: dict[str, str | bool],
    annotator: str = "human-author",
) -> dict[str, Any]:
    if verdict not in LABELS:
        raise ValueError(f"unsupported verdict: {verdict}")
    instruction = instruction.strip()
    now = datetime.now(UTC).isoformat()
    gold_record = {
        "id": item_id,
        "final_verdict": verdict,
    }
    audit_record = {
        **gold_record,
        "instruction": instruction or None,
        "annotator": annotator,
        "decided_at": now,
        "article_url": source.get("url") or None,
        "article_url_provenance": source.get("provenance") or "missing",
    }

    latest = load_human_annotations(output_path)
    latest[item_id] = gold_record
    minimal_gold = [
        {"id": key, "final_verdict": latest[key]["final_verdict"]}
        for key in sorted(latest, key=_id_number)
    ]
    _atomic_write_jsonl(output_path, minimal_gold)

    history_path = output_path.with_name(f"{output_path.stem}_history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return gold_record


def annotations_as_jsonl(annotations: dict[str, dict[str, Any]]) -> str:
    return "".join(
        json.dumps(
            {"id": key, "final_verdict": annotations[key]["final_verdict"]},
            ensure_ascii=False,
        )
        + "\n"
        for key in sorted(annotations, key=_id_number)
    )


def _collect_role_material(
    workspace: dict[str, Any], item_id: str, role: str
) -> dict[str, list[Any]]:
    theses: list[str] = []
    claims: list[str] = []
    sources: list[dict[str, str]] = []
    for mode in MODES:
        record = workspace["runs"].get(item_id, {}).get(mode, {})
        for argument in record.get("arguments") or []:
            if argument.get("role") != role:
                continue
            thesis = str(argument.get("thesis") or "").strip()
            if thesis:
                theses.append(thesis)
            claims.extend(
                str(claim.get("text") or "").strip()
                for claim in argument.get("claims") or []
                if str(claim.get("text") or "").strip()
            )
            for source in argument.get("sources") or []:
                url = str(source.get("url") or "").strip()
                if not _is_http_url(url):
                    continue
                sources.append(
                    {
                        "url": url,
                        "title": str(source.get("title") or "").strip(),
                        "supports": str(source.get("supports") or "").strip(),
                        "role": role,
                        "mode": mode,
                    }
                )
    return {
        "theses": _dedupe(theses),
        "claims": _dedupe(claims),
        "sources": _dedupe_sources(sources),
    }


def _compile_replacement_analysis(
    workspace: dict[str, Any], item_id: str, review: dict[str, Any]
) -> dict[str, Any]:
    """Render a replacement case from its fresh manual review only.

    Historical F/B2/B3 records refer to the article that previously occupied
    this stable TC identifier and therefore must never leak into the new case.
    """
    case = workspace["cases"][item_id]
    sources = _dedupe_sources(list(review.get("verification_sources") or []))
    criteria = {
        "reliable": (
            "Обирайте reliable, якщо ключові факти підтверджуються якісними "
            "незалежними або первинними джерелами, без матеріальних суперечностей."
        ),
        "questionable": (
            "Обирайте questionable, якщо джерело підтверджує лише частину "
            "ключового твердження, а контекст або формулювання істотно вводять в оману."
        ),
        "unreliable": (
            "Обирайте unreliable лише якщо ключове твердження спростоване, "
            "сфабриковане або матеріал систематично викривляє встановлені факти."
        ),
    }
    positions: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        positions[label] = {
            "criterion": criteria[label],
            "arguments": list(review.get(f"{label}_arguments") or []),
            "claims": list(review.get("checked_claims") or []),
            "sources": sources,
        }
    return {
        "id": item_id,
        "case": case,
        "title": display_title(case),
        "source": source_info(workspace, item_id),
        "positions": positions,
        "verification_sources": sources,
        "is_replacement": True,
    }


def _shared_verification_sources(
    workspace: dict[str, Any],
    item_id: str,
    role_material: dict[str, dict[str, list[Any]]],
) -> list[dict[str, str]]:
    sources = [
        *role_material["advocate"]["sources"],
        *role_material["prosecutor"]["sources"],
    ]
    second = workspace["second"].get(item_id, {})
    for url in second.get("sources") or []:
        if _is_http_url(str(url)):
            sources.append(
                {
                    "url": str(url),
                    "title": "Джерело сліпої другої анотації",
                    "supports": "Перевірте безпосередньо перед рішенням.",
                    "role": "independent-check",
                    "mode": "annotation",
                }
            )
    case = workspace["cases"][item_id]
    curator_url = str(case.get("gold", {}).get("evidence_url") or "").strip()
    if _is_http_url(curator_url):
        sources.append(
            {
                "url": curator_url,
                "title": "Джерело первинної перевірки",
                "supports": "Перевірте безпосередньо перед рішенням.",
                "role": "independent-check",
                "mode": "annotation",
            }
        )
    for bucket in ("arbitration", "adjudication"):
        url = str(workspace[bucket].get(item_id, {}).get("decisive_source") or "").strip()
        match = re.search(r"https?://[^\s)]+", url)
        if match:
            sources.append(
                {
                    "url": match.group(0).rstrip(".,;"),
                    "title": "Джерело попереднього арбітражу",
                    "supports": "Перевірте безпосередньо; висновок арбітра приховано.",
                    "role": "independent-check",
                    "mode": "annotation",
                }
            )
    return _dedupe_sources(sources)


def _caution_arguments(role_material: dict[str, dict[str, list[Any]]]) -> list[str]:
    cautious: list[str] = []
    for role in ("advocate", "prosecutor"):
        for thesis in role_material[role]["theses"]:
            sentences = re.split(r"(?<=[.!?])\s+", thesis)
            cautious.extend(
                sentence.strip()
                for sentence in sentences
                if any(marker in sentence.lower() for marker in _CAUTION_MARKERS)
            )
    return cautious


def _questionable_claims(role_material: dict[str, dict[str, list[Any]]]) -> list[str]:
    claims = [
        claim
        for claim in role_material["prosecutor"]["claims"]
        if any(marker in claim.lower() for marker in _CAUTION_MARKERS)
    ]
    if claims:
        return _dedupe(claims)
    return role_material["prosecutor"]["claims"][:8]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip().casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(re.sub(r"\s+", " ", value).strip())
    return result


def _dedupe_sources(values: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for value in values:
        url = value["url"]
        if url not in merged:
            merged[url] = dict(value)
            continue
        if not merged[url].get("title") and value.get("title"):
            merged[url]["title"] = value["title"]
        if not merged[url].get("supports") and value.get("supports"):
            merged[url]["supports"] = value["supports"]
    return list(merged.values())


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp_path.replace(path)


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _id_number(value: str) -> int:
    match = re.search(r"\d+", value)
    return int(match.group()) if match else 0
