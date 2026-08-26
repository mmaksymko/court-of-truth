import os

import httpx
import streamlit as st

API_URL = os.environ.get("COURT_API_URL", "http://127.0.0.1:8000")

DETECTOR_NAMES = {
    "ai_generated": "Згенерований ШІ",
    "mt_translation": "Машинний переклад",
    "jeansa": "Джинса",
    "clickbait": "Клікбейт",
}


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=30.0)


def render_report(report: dict) -> None:
    risk = report["risk"]
    flagged = risk["flagged_count"]
    if flagged == 0:
        st.success("Чисто: жоден детектор не спрацював.", icon=":material/check_circle:")
    else:
        st.warning(f"Спрацювало детекторів: {flagged}.", icon=":material/flag:")
    if report.get("source_title"):
        st.markdown(f"**Заголовок:** {report['source_title']}")
    if report.get("source_text"):
        with st.expander(f"Проаналізований текст · {report['text_chars']} символів"):
            st.write(report["source_text"])

    for result in report["results"]:
        name = DETECTOR_NAMES.get(result["id"], result["id"])
        with st.container(border=True):
            if result["status"] == "skipped":
                st.markdown(f"**{name}** — :gray[пропущено] ({result['reason']})")
                continue
            badge = ":red[спрацював]" if result["flagged"] else ":green[чисто]"
            st.markdown(f"**{name}** — {badge}")
            st.progress(
                result["probability"],
                text=f"Ймовірність: {result['probability']:.2f} · мітка: {result['label']}",
            )
            if result["evidence"]:
                st.caption("Докази: " + " · ".join(result["evidence"]))
            if result["low_confidence"]:
                st.caption(":orange[Низька впевненість.]")
            for caveat in result["caveats"]:
                st.caption(f":material/warning: {caveat}")


st.set_page_config(page_title="Аналіз новинного тексту", page_icon=":material/gavel:")
st.title("Аналіз новинного тексту")
st.caption(
    "Криміналіст: локальний аналіз чотирма детекторами (згенерований ШІ, "
    "машинний переклад, джинса, клікбейт). Встав посилання або текст."
)

mode = st.segmented_control(
    "Джерело",
    ["Посилання", "Текст"],
    default="Посилання",
    required=True,
    label_visibility="collapsed",
)

with st.form("analyze"):
    url = title = text = ""
    if mode == "Текст":
        title = st.text_input("Заголовок", placeholder="Заголовок статті")
        text = st.text_area("Текст статті", height=220, placeholder="Вставте текст статті")
    else:
        url = st.text_input("Посилання на статтю", placeholder="https://...")
    submitted = st.form_submit_button("Аналізувати", icon=":material/search:", type="primary")

results = st.container()

if submitted:
    if mode == "Посилання":
        if not url.strip():
            st.warning("Вкажіть посилання на статтю.")
            st.stop()
        payload = {"url": url.strip()}
    else:
        if not title.strip() and not text.strip():
            st.warning("Вкажіть заголовок або текст.")
            st.stop()
        payload = {"title": title, "text": text}

    with results:
        with st.spinner("Аналізую…"):
            try:
                response = client().post("/v1/analyze", json=payload)
            except httpx.HTTPError as exc:
                st.error(
                    f"Немає зв'язку з API ({API_URL}). Запустіть сервер: "
                    f"`python -m court`. Деталі: {exc}"
                )
                st.stop()
        if response.status_code != httpx.codes.OK:
            try:
                body = response.json()
            except ValueError:
                body = {}
            message = body.get("message", response.text)
            code = body.get("code", response.status_code)
            st.error(f"{message} (код {code})")
            st.stop()
        render_report(response.json())
