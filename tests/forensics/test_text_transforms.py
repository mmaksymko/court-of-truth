from court.forensics.cleaning import (
    clean_ai,
    clean_clickbait,
    clean_jeansa,
    collapse_ws,
)
from court.forensics.mt_cleaning import clean_mt
from court.forensics.text_transforms import transform_for


def test_collapse_ws():
    assert collapse_ws("  а\n\tб   в ") == "а б в"


def test_collapse_ws_handles_none():
    assert collapse_ws(None) == ""


def test_jeansa_strips_disclosure():
    out = clean_jeansa("Купуйте зараз на правах реклами найкраще")
    assert "на правах реклами" not in out
    assert "Купуйте зараз" in out


def test_jeansa_caps_at_250_words():
    assert len(clean_jeansa(" ".join(["слово"] * 300)).split()) == 250


def test_mt_folds_glyphs():
    assert clean_mt("«текст» — з тире") == '"текст" - з тире'


def test_mt_drops_urls():
    assert "http" not in clean_mt("дивись https://ex.ua/a тут")


def test_mt_url_regex_is_linear_on_long_token():
    assert clean_mt("https://" + "a" * 50_000) == ""


def test_clickbait_strips_emoji():
    assert "😀" not in clean_clickbait("заголовок 😀")


def test_clickbait_strips_site_suffix():
    assert clean_clickbait("Шокуюча правда — Українська правда") == "Шокуюча правда"


def test_ai_strips_scaffold_labels():
    assert "Джерело" not in clean_ai("Джерело: щось справді важливе сталося сьогодні")


def test_ai_caps_at_300_words():
    assert len(clean_ai(" ".join(["слово"] * 400)).split()) == 300


def test_transform_for_dispatches_by_detector():
    assert "😀" not in transform_for("clickbait", "заголовок 😀")
    assert "Джерело" not in transform_for("ai_generated", "Джерело: текст статті")


def test_transform_for_handles_none():
    assert transform_for("jeansa", None) == ""
