import hashlib
import json
import os
from typing import Any

import streamlit as st

from court.ui.api import ApiError, CourtApi, Readiness
from court.ui.presentation import (
    detector_name,
    label_name,
    percent,
    role_name,
    verification_name,
)

API_URL = os.environ.get("COURT_API_URL", "http://127.0.0.1:8000")
LOCAL_MODE = "Локальні сигнали"
TRIBUNAL_MODE = "Судово-змагальний розгляд"


@st.cache_resource
def api_client() -> CourtApi:
    return CourtApi(API_URL)


@st.cache_data(ttl=10, show_spinner=False)
def readiness() -> Readiness:
    try:
        return api_client().health()
    except ApiError as exc:
        return Readiness(False, False, str(exc))


def render_readiness(state: Readiness) -> None:
    first, second = st.columns(2)
    with first:
        if state.forensics:
            st.success("Локальний модуль готовий", icon=":material/check_circle:")
        else:
            st.error("Локальний модуль недоступний", icon=":material/error:")
    with second:
        if state.tribunal:
            st.success("Трибунал готовий", icon=":material/check_circle:")
        else:
            st.warning("Трибунал недоступний", icon=":material/warning:")


def render_report(report: dict[str, Any]) -> None:
    risk = report.get("risk", {})
    flagged = int(risk.get("flagged_count", 0))
    if flagged == 0:
        st.success("Локальні детектори не спрацювали.", icon=":material/check_circle:")
    else:
        st.warning(f"Кількість детекторів, що спрацювали: {flagged}.", icon=":material/flag:")
    st.caption(
        "Локальні сигнали є орієнтирами для подальшої перевірки й самі по собі "
        "не доводять достовірності або недостовірності матеріалу."
    )
    if report.get("source_title"):
        st.markdown("**Заголовок матеріалу**")
        st.write(report["source_title"])
    if report.get("source_text"):
        with st.expander(f"Проаналізований текст · {report.get('text_chars', 0)} символів"):
            st.write(report["source_text"])

    for result in report.get("results", []):
        with st.container(border=True):
            st.markdown(f"**{detector_name(str(result.get('id', '')))}**")
            if result.get("status") == "skipped":
                st.caption("Пропущено")
                st.write(result.get("reason", "Причину не вказано."))
                continue
            status = "Сигнал виявлено" if result.get("flagged") else "Сигнал не виявлено"
            st.write(status)
            probability = float(result.get("probability", 0.0))
            st.progress(
                probability,
                text=(
                    f"Оцінка позитивного класу: {percent(probability)} · "
                    f"результат: {label_name(str(result.get('label', '')))}"
                ),
            )
            evidence = result.get("evidence") or []
            if evidence:
                st.caption("Пояснювальні ознаки")
                for item in evidence:
                    st.write(f"• {item}")
            if result.get("low_confidence"):
                st.warning("Оцінка розташована поблизу порога рішення.", icon=":material/info:")
            for caveat in result.get("caveats") or []:
                st.caption(f"Застереження: {caveat}")
            with st.expander("Технічні відомості"):
                st.json(result)


def render_argument(argument: dict[str, Any], evidence: list[dict[str, Any]]) -> None:
    st.markdown("**Теза сторони**")
    st.write(argument.get("thesis", "—"))
    st.markdown("**Твердження**")
    claims = argument.get("claims") or []
    if not claims:
        st.caption("Твердження не наведено.")
    for claim in claims:
        st.write(f"{claim.get('id', '—')}. {claim.get('text', '—')}")
    cited = argument.get("cited_detectors") or []
    if cited:
        st.caption("Згадані локальні детектори")
        st.write(" · ".join(detector_name(str(item)) for item in cited))
    source_refs = argument.get("sources") or []
    if source_refs:
        st.caption("Посилання на твердження та зовнішні матеріали")
        references = []
        for source in source_refs:
            claim_id = str(source.get("claim_id", "—"))
            evidence_id = next(
                (
                    str(record.get("id"))
                    for record in evidence
                    if record.get("claim_id") == source.get("claim_id")
                    and record.get("url") == source.get("url")
                ),
                "",
            )
            references.append(f"{claim_id} → {evidence_id}" if evidence_id else claim_id)
        st.write(" · ".join(references))


def render_evidence(evidence: list[dict[str, Any]], used_ids: set[str]) -> None:
    st.warning(
        "Для зовнішніх матеріалів програмно підтверджено лише URL із результатів пошуку. "
        "Назви, уривки та пояснення сформовано мовною моделлю; вони не є незалежною "
        "фактологічною перевіркою.",
        icon=":material/warning:",
    )
    if not evidence:
        st.info("Сторони не подали зовнішніх матеріалів.")
        return
    ordered = sorted(evidence, key=lambda item: str(item.get("id")) not in used_ids)
    for record in ordered:
        record_id = str(record.get("id", "—"))
        with st.container(border=True):
            heading = f"{record_id} · {role_name(str(record.get('side', '')))}"
            if record_id in used_ids:
                heading += " · враховано у вердикті"
            st.markdown(f"**{heading}**")
            st.write(record.get("title") or "Матеріал без назви")
            url = str(record.get("url", ""))
            if url:
                st.link_button("Відкрити джерело", url, icon=":material/open_in_new:")
            st.caption(f"Пов’язане твердження: {record.get('claim_id', '—')}")
            st.write(record.get("excerpt", "—"))
            st.caption(f"Значення для позиції: {record.get('supports', '—')}")
            st.caption("Перевірка: " + verification_name(str(record.get("verification", ""))))


def render_verdict(review: dict[str, Any]) -> None:
    deliberation = review.get("deliberation", {})
    verdict = deliberation.get("verdict", {})
    probabilities = verdict.get("probabilities", {})
    with st.container(border=True):
        st.subheader(f"Вердикт: {label_name(str(verdict.get('label', ''))).capitalize()}")
        st.caption("Упевненість Судді")
        st.markdown(f"## {percent(verdict.get('confidence'))}")
        for key in ("reliable", "questionable", "unreliable"):
            value = float(probabilities.get(key, 0.0))
            st.progress(value, text=f"{label_name(key).capitalize()}: {percent(value)}")

    st.markdown("**Обґрунтування**")
    st.write(verdict.get("rationale", "—"))
    st.markdown("**Ключові сигнали**")
    signals = verdict.get("key_signals") or []
    if signals:
        for signal in signals:
            st.write(f"• {signal}")
    else:
        st.caption("Окремі ключові сигнали не наведено.")
    st.markdown("**Оцінка джерел**")
    st.write(verdict.get("source_assessment", "—"))

    evidence = deliberation.get("evidence") or []
    tabs = st.tabs(["Прокурор", "Адвокат", "Зовнішні матеріали", "Локальні сигнали"])
    with tabs[0]:
        render_argument(deliberation.get("prosecutor", {}), evidence)
    with tabs[1]:
        render_argument(deliberation.get("advocate", {}), evidence)
    with tabs[2]:
        render_evidence(evidence, set(verdict.get("used_evidence_ids") or []))
    with tabs[3]:
        render_report(review.get("report", {}))

    objections = verdict.get("objections") or []
    if objections:
        st.markdown("**Зауваження Судді до тверджень сторін**")
        for objection in objections:
            st.write(f"{objection.get('claim_id', '—')}: {objection.get('reason', '—')}")
    cited = verdict.get("cited_detectors") or []
    if cited:
        st.caption(
            "Локальні сигнали, враховані Суддею: "
            + " · ".join(detector_name(str(item)) for item in cited)
        )


def result_key(mode: str, source: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps([mode, source, payload], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def render_api_error(exc: ApiError) -> None:
    messages = {
        422: "Перевірте, чи введено коректне посилання або непорожній текст.",
        429: "Сервер тимчасово обмежив частоту запитів. Повторіть спробу пізніше.",
        502: "Зовнішній сервіс не зміг сформувати коректну відповідь.",
        503: "Обраний компонент системи зараз не готовий до роботи.",
        504: "Зовнішній сервіс не завершив відповідь у відведений час.",
    }
    st.error(messages.get(exc.status_code, exc.message), icon=":material/error:")
    with st.expander("Технічні відомості про помилку"):
        st.write(f"Код: {exc.code}")
        if exc.request_id:
            st.write(f"Ідентифікатор запиту: {exc.request_id}")
        st.write(f"API: {API_URL}")


st.set_page_config(
    page_title="Судово-змагальна верифікація новин",
    page_icon=":material/gavel:",
    layout="wide",
)
st.title("Судово-змагальна верифікація новин")
st.caption(
    "Локальні детектори формують сигнали, а Прокурор і Адвокат досліджують зовнішні "
    "матеріали. Суддя зіставляє позиції сторін і виносить підсумковий вердикт."
)

system_state = readiness()
render_readiness(system_state)

analysis_mode = st.segmented_control(
    "Режим аналізу",
    [LOCAL_MODE, TRIBUNAL_MODE],
    default=TRIBUNAL_MODE,
    required=True,
)
source_mode = st.segmented_control(
    "Спосіб введення матеріалу",
    ["Посилання", "Заголовок і текст"],
    default="Посилання",
    required=True,
)

if analysis_mode == TRIBUNAL_MODE:
    st.info(
        "Повний розгляд використовує зовнішню мовну модель і вебпошук та може тривати "
        "кілька хвилин.",
        icon=":material/schedule:",
    )
    if not system_state.tribunal:
        st.warning(
            "Повний розгляд недоступний: сервер не має налаштованого компонента Трибуналу. "
            "Локальний аналіз можна виконати без нього.",
            icon=":material/key_off:",
        )

with st.form("analysis_form"):
    url = title = article_text = ""
    if source_mode == "Заголовок і текст":
        title = st.text_input("Заголовок", placeholder="Заголовок новинного матеріалу")
        article_text = st.text_area(
            "Текст матеріалу", height=240, placeholder="Вставте текст новинного матеріалу"
        )
    else:
        url = st.text_input("Посилання на матеріал", placeholder="https://example.org/news")
    button_label = (
        "Розпочати повний розгляд" if analysis_mode == TRIBUNAL_MODE else "Проаналізувати локально"
    )
    submitted = st.form_submit_button(
        button_label,
        icon=":material/gavel:" if analysis_mode == TRIBUNAL_MODE else ":material/search:",
        type="primary",
        disabled=(
            not system_state.forensics
            or (analysis_mode == TRIBUNAL_MODE and not system_state.tribunal)
        ),
    )

payload = (
    {"url": url.strip()}
    if source_mode == "Посилання"
    else {"title": title.strip(), "text": article_text.strip()}
)
current_key = result_key(str(analysis_mode), str(source_mode), payload)

if submitted:
    if source_mode == "Посилання" and not url.strip():
        st.warning("Вкажіть посилання на новинний матеріал.")
    elif source_mode == "Заголовок і текст" and not title.strip() and not article_text.strip():
        st.warning("Вкажіть заголовок або текст новинного матеріалу.")
    else:
        spinner = (
            "Трибунал досліджує матеріал…"
            if analysis_mode == TRIBUNAL_MODE
            else "Локальні детектори аналізують матеріал…"
        )
        try:
            with st.spinner(spinner, show_time=True):
                response = (
                    api_client().review(payload)
                    if analysis_mode == TRIBUNAL_MODE
                    else api_client().analyze(payload)
                )
        except ApiError as exc:
            render_api_error(exc)
        else:
            st.session_state["analysis_result"] = response
            st.session_state["analysis_result_mode"] = analysis_mode
            st.session_state["analysis_result_key"] = current_key

if st.session_state.get("analysis_result_key") == current_key:
    stored = st.session_state.get("analysis_result", {})
    stored_mode = st.session_state.get("analysis_result_mode")
    st.divider()
    if stored_mode == TRIBUNAL_MODE:
        render_verdict(stored)
    else:
        st.subheader("Результати локального аналізу")
        render_report(stored)
    with st.expander("Повна технічна відповідь"):
        st.download_button(
            "Завантажити JSON",
            data=json.dumps(stored, ensure_ascii=False, indent=2),
            file_name="court-of-truth-result.json",
            mime="application/json",
            icon=":material/download:",
        )
        st.json(stored)
