"""Human-readable promotional signals for jeansa (undisclosed-advertising) verdicts.

The trained jeansa classifier keys on character n-grams, which make poor courtroom
exhibits. These signals are the interpretable corroboration a reader can check by
eye: concrete promotional lexicon, calls to action, price/discount mentions, and
so on. The lexicons mirror those used to build the jeansa dataset so the evidence
speaks the same language as the training analysis.
"""

import re

_PROMO_LEXICON = (
    "унікальн", "інноваційн", "провідн", "лідер", "найкращ", "якісн",
    "вигідн", "спеціальн пропозиц", "знижк", "акці", "бонус", "подарунок",
    "безкоштовн", "ексклюзивн", "новинк", "сучасн", "надійн", "зручн",
    "professional", "преміум", "гарантія", "довіра", "успішн", "ефективн",
)
_CTA_PATTERNS = (
    "дізнатися більше", "детальніше на", "замовити", "придбати", "реєструйт",
    "долучайт", "переходьте", "за посиланням", "на сайті", "звертайтеся",
    "телефонуйте", "запрошуємо",
)
_WE_WORDS = frozenset(
    ("ми", "наш", "наша", "наші", "нашого", "нашу", "компанія", "компанії")
)
_CONTACT_RE = re.compile(r"https?://|www\.|\+?38\s?0\d{2}|\b\d{3}[-\s]?\d{2}[-\s]?\d{2}\b")
_PERCENT_RE = re.compile(r"\d+\s?%|\d+\s?(?:грн|₴|дол|usd|eur|€)", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ'ʼ-]+")
_EXAMPLES_PER_SIGNAL = 3
_MIN_SUPERLATIVE_LEN = 5


def _label(name: str, examples: list[str]) -> str:
    unique: list[str] = []
    for example in examples:
        trimmed = example.strip()
        if trimmed and trimmed not in unique:
            unique.append(trimmed)
        if len(unique) == _EXAMPLES_PER_SIGNAL:
            break
    return f"{name}: {', '.join(unique)}"


def promo_evidence(text: str, k: int = 5) -> list[str]:
    low = text.lower()
    toks = _TOKEN_RE.findall(low)
    scored: list[tuple[int, str]] = []

    promo = [stem for stem in _PROMO_LEXICON if stem in low]
    if promo:
        scored.append((sum(low.count(stem) for stem in promo), _label("промо-лексика", promo)))

    cta = [phrase for phrase in _CTA_PATTERNS if phrase in low]
    if cta:
        scored.append((sum(low.count(phrase) for phrase in cta), _label("заклики до дії", cta)))

    prices = _PERCENT_RE.findall(text)
    if prices:
        scored.append((len(prices), _label("ціни та знижки", prices)))

    superlatives = [
        tok for tok in toks if tok.startswith("най") and len(tok) > _MIN_SUPERLATIVE_LEN
    ]
    if superlatives:
        scored.append((len(superlatives), _label("найвищий ступінь", superlatives)))

    we_mentions = [tok for tok in toks if tok in _WE_WORDS]
    if we_mentions:
        scored.append((len(we_mentions), _label("самопрезентація компанії", we_mentions)))

    contacts = _CONTACT_RE.findall(text)
    if contacts:
        scored.append((len(contacts), _label("контакти та посилання", contacts)))

    exclamations = text.count("!")
    if exclamations:
        scored.append((exclamations, f"оклики: {exclamations}×"))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [label for _, label in scored[:k]]
