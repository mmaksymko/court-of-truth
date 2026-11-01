from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from court.ui.api import CourtApi, Readiness

APP = Path(__file__).parents[1] / "streamlit_app.py"


def report() -> dict:
    return {
        "title_present": True,
        "text_chars": 40,
        "source_title": "Перевірка матеріалу",
        "source_text": "Текст без небезпечного HTML.",
        "risk": {"kind": "heuristic", "flagged_count": 1},
        "results": [
            {
                "status": "ok",
                "id": "jeansa",
                "scope": "body",
                "label": "sponsored",
                "probability": 0.81,
                "flagged": True,
                "low_confidence": False,
                "caveats": [],
                "evidence": ["оцінна лексика"],
            }
        ],
    }


def review() -> dict:
    return {
        "report": report(),
        "deliberation": {
            "prosecutor": {
                "role": "prosecutor",
                "thesis": "Матеріал містить неперевірені твердження.",
                "claims": [{"id": "p1", "text": "Бракує первинного джерела."}],
                "cited_detectors": ["jeansa"],
                "sources": [{"claim_id": "p1"}],
            },
            "advocate": {
                "role": "advocate",
                "thesis": "Основну подію підтверджено.",
                "claims": [{"id": "a1", "text": "Подію описано в офіційному повідомленні."}],
                "cited_detectors": [],
                "sources": [{"claim_id": "a1"}],
            },
            "evidence": [
                {
                    "id": "ep1",
                    "side": "prosecutor",
                    "claim_id": "p1",
                    "url": "https://example.org/source",
                    "title": "Зовнішній матеріал",
                    "excerpt": "Короткий опис матеріалу.",
                    "supports": "Уточнює походження твердження.",
                    "verification": "search_url_only",
                }
            ],
            "verdict": {
                "label": "questionable",
                "probabilities": {"reliable": 0.2, "questionable": 0.7, "unreliable": 0.1},
                "confidence": 0.7,
                "rationale": "Частину істотних тверджень підтверджено недостатньо.",
                "key_signals": ["Неповне посилання на першоджерело."],
                "source_assessment": "Джерельна база неповна.",
                "used_evidence_ids": ["ep1"],
                "objections": [{"claim_id": "p1", "reason": "Потрібне уточнення."}],
                "cited_detectors": ["jeansa"],
                "schema_version": 1,
            },
        },
    }


def _ready(monkeypatch) -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(CourtApi, "health", lambda _self: Readiness(True, True))


def test_streamlit_app_renders_full_review_form(monkeypatch):
    _ready(monkeypatch)
    app = AppTest.from_file(APP).run()

    assert not app.exception
    assert app.title[0].value == "Судово-змагальна верифікація новин"
    assert app.segmented_control[0].label == "Режим аналізу"
    assert app.segmented_control[1].label == "Спосіб введення матеріалу"
    assert app.button[0].label == "Розпочати повний розгляд"
    assert not app.button[0].disabled


def test_tribunal_is_blocked_without_key_but_local_mode_remains_available(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(CourtApi, "health", lambda _self: Readiness(True, False))
    app = AppTest.from_file(APP).run()

    assert app.button[0].disabled
    app.segmented_control[0].set_value("Локальні сигнали").run()
    assert app.button[0].label == "Проаналізувати локально"
    assert not app.button[0].disabled


def test_full_review_response_renders_verdict_roles_and_evidence(monkeypatch):
    _ready(monkeypatch)
    monkeypatch.setattr(CourtApi, "review", lambda _self, _payload: review())
    app = AppTest.from_file(APP).run()
    app.text_input[0].set_value("https://example.org/news")
    app.button[0].click().run(timeout=10)

    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "Вердикт: Сумнівний" in "\n".join(item.value for item in app.subheader)
    assert "Матеріал містить неперевірені твердження" in rendered
    assert "враховано у вердикті" in rendered
    assert [tab.label for tab in app.tabs] == [
        "Прокурор",
        "Адвокат",
        "Зовнішні матеріали",
        "Локальні сигнали",
    ]


def test_local_result_uses_precise_non_evidentiary_wording(monkeypatch):
    _ready(monkeypatch)
    local = report()
    local["risk"]["flagged_count"] = 0
    monkeypatch.setattr(CourtApi, "analyze", lambda _self, _payload: local)
    app = AppTest.from_file(APP).run()
    app.segmented_control[0].set_value("Локальні сигнали").run()
    app.text_input[0].set_value("https://example.org/news")
    app.button[0].click().run(timeout=10)

    messages = "\n".join(item.value for item in app.success)
    captions = "\n".join(item.value for item in app.caption)
    assert "Локальні детектори не спрацювали" in messages
    assert "не доводять достовірності" in captions
    assert "Пояснювальні ознаки" in captions
