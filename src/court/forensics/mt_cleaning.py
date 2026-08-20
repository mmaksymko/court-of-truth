import re

from court.forensics.cleaning import (
    WORD_CAP,
    cap_words,
    collapse_ws,
    normalize_glyphs,
    strip_urls,
)

_SCAFFOLD = re.compile(
    r"\b(?:Джерело|Деталі|Пряма мова|Дослівно|Оновлено|Пряма цитата|Довідка)\s*:\s*",
    re.IGNORECASE,
)
_DOMAINS = ("pravda.com.ua", "news-pravda", "politnavigator")
_CTA = (
    "monobank",
    "монобанк",
    "підтримай",
    "донат",
    "підписуйтесь",
    "підписатися",
    "телеграм-канал",
    "telegram",
)
_SENTENCE = re.compile(r"([.!?]+)")


def clean_mt(text: str) -> str:
    text = normalize_glyphs(strip_urls(_SCAFFOLD.sub(" ", text)))
    text = _drop_cta_sentences(_drop_domain_tokens(text))
    return cap_words(collapse_ws(text), WORD_CAP)


def _drop_domain_tokens(text: str) -> str:
    return " ".join(word for word in text.split() if not _has_domain(word))


def _has_domain(word: str) -> bool:
    lowered = word.lower()
    return any(domain in lowered for domain in _DOMAINS)


def _drop_cta_sentences(text: str) -> str:
    parts = _SENTENCE.split(text)
    kept = []
    for index in range(0, len(parts), 2):
        segment = parts[index]
        delimiter = parts[index + 1] if index + 1 < len(parts) else ""
        if not any(trigger in segment.lower() for trigger in _CTA):
            kept.append(segment + delimiter)
    return "".join(kept)
