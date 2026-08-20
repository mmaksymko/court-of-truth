import re

WORD_CAP = 300
JEANSA_WORD_CAP = 250

_WS = re.compile(r"\s+")
_SCAFFOLD_AI = re.compile(
    r"\b(?:Джерело|Деталі|Пряма мова|Дослівно|Оновлено|Нагадаємо|Читайте також)\s*:\s*",
    re.IGNORECASE,
)
_DISCLOSURE_MARKERS = [
    "на правах реклами",
    "новини компаній",
    "новини компанії",
    "партнерський матеріал",
    "партнерський проєкт",
    "партнерський проект",
    "матеріал підготовлено",
    "прес-реліз",
    "прес реліз",
    "пресреліз",
    "promoted",
    "реклама",
    "рекламний матеріал",
]
_DISCLOSURE = re.compile("|".join(re.escape(m) for m in _DISCLOSURE_MARKERS), re.IGNORECASE)

_URL = re.compile(r"(?:https?://|t\.me/|www\.)\S+|#\S+")
_QUOTES = re.compile(r"[«»“”„‟\"']")
_GUILLEMETS = re.compile(r"[«»“”„‟]")
_DASH = re.compile(r"[–—―]")
_APOS = re.compile(r"[ʼ’‘`]")
_EMOJI = re.compile("[\U0001f000-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff\ufe0f]+")
_SITE_SUFFIX = re.compile(
    r"\s*[-–—|•]\s*(?:Znaj\.?ua|Politeka|Політека|Укр\.?Медіа|ukr\.media|"
    r"Українська правда|Громадське|Еспресо|ZN\.UA)\s*$",
    re.IGNORECASE,
)


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def strip_disclosure(text: str) -> str:
    return _DISCLOSURE.sub(" ", text)


def strip_urls(text: str) -> str:
    return _URL.sub(" ", text)


def strip_emoji(text: str) -> str:
    return _EMOJI.sub(" ", text)


def strip_site_suffix(text: str) -> str:
    return _SITE_SUFFIX.sub("", text)


def normalize_glyphs(text: str) -> str:
    return _APOS.sub("'", _DASH.sub("-", _QUOTES.sub('"', text)))


def normalize_guillemets(text: str) -> str:
    return _DASH.sub("-", _GUILLEMETS.sub('"', text))


def cap_words(text: str, limit: int) -> str:
    return " ".join(text.split()[:limit])


def clean_manipulation(text: str) -> str:
    return collapse_ws(text)


def clean_clickbait(text: str) -> str:
    return collapse_ws(strip_site_suffix(strip_emoji(text)))


def clean_ai(text: str) -> str:
    return cap_words(collapse_ws(normalize_guillemets(_SCAFFOLD_AI.sub(" ", text))), WORD_CAP)


def clean_jeansa(text: str) -> str:
    return cap_words(collapse_ws(strip_disclosure(text)), JEANSA_WORD_CAP)
