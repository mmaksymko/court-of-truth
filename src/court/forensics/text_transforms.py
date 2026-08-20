from court.forensics import cleaning, mt_cleaning
from court.forensics.schemas import MAX_TEXT_CHARS

_BY_DETECTOR = {
    "ai_generated": cleaning.clean_ai,
    "jeansa": cleaning.clean_jeansa,
    "mt_translation": mt_cleaning.clean_mt,
    "clickbait": cleaning.clean_clickbait,
}


def transform_for(detector_id: str, text: str) -> str:
    clean = _BY_DETECTOR.get(detector_id, cleaning.clean_manipulation)
    return clean((text or "")[:MAX_TEXT_CHARS])
