DETECTOR_NAMES = {
    "ai_generated": "Ознаки генерації ШІ",
    "mt_translation": "Ознаки машинного перекладу",
    "jeansa": "Ознаки джинси",
    "clickbait": "Ознаки клікбейту",
}

LABELS = {
    "reliable": "надійний",
    "questionable": "сумнівний",
    "unreliable": "ненадійний",
    "ai_generated": "згенеровано ШІ",
    "human_news": "авторський новинний текст",
    "ai_translated_ua": "машинний переклад",
    "human_ua": "авторський український текст",
    "sponsored": "матеріал з ознаками джинси",
    "editorial": "редакційний матеріал",
    "clickbait": "клікбейт",
    "neutral": "нейтральний заголовок",
}

ROLE_NAMES = {
    "prosecutor": "прокурор",
    "advocate": "адвокат",
    "neutral": "дослідник",
}

VERIFICATION_NAMES = {"search_url_only": "підтверджено лише наявність URL у результатах пошуку"}


def detector_name(detector_id: str) -> str:
    return DETECTOR_NAMES.get(detector_id, detector_id)


def label_name(label: str) -> str:
    return LABELS.get(label, label)


def role_name(role: str) -> str:
    return ROLE_NAMES.get(role, role)


def verification_name(verification: str) -> str:
    return VERIFICATION_NAMES.get(verification, verification)


def percent(value: float | int | str | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"
