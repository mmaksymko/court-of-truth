from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from court.annotation.workbench import (
    LABELS,
    annotations_as_jsonl,
    compile_case_analysis,
    load_human_annotations,
    load_human_instructions,
    load_static_workspace,
    recommend_verdict,
    save_human_annotation,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "evals/annotation/human_gold.jsonl"
OUTPUT_PATH = Path(os.environ.get("COURT_HUMAN_GOLD_PATH", DEFAULT_OUTPUT))

LABEL_UI = {
    "reliable": "Надійна",
    "questionable": "Сумнівна",
    "unreliable": "Ненадійна",
}
st.set_page_config(
    page_title="Людське маркування gold",
    page_icon=":material/fact_check:",
    layout="wide",
)


@st.cache_data
def static_workspace() -> dict:
    return load_static_workspace(ROOT)


def go_to(item_id: str) -> None:
    st.session_state["case_picker"] = item_id


def next_unlabelled(ids: list[str], annotations: dict, current: str) -> str:
    start = ids.index(current)
    ordered = ids[start + 1 :] + ids[: start + 1]
    return next((item_id for item_id in ordered if item_id not in annotations), current)


def record_decision(item_id: str, ids: list[str]) -> None:
    try:
        annotations = load_human_annotations(OUTPUT_PATH)
        analysis = compile_case_analysis(static_workspace(), item_id)
        verdict = str(st.session_state[f"verdict_{item_id}"])
        record = save_human_annotation(
            OUTPUT_PATH,
            item_id=item_id,
            verdict=verdict,
            instruction=str(st.session_state[f"instruction_{item_id}"]),
            source=analysis["source"],
        )
        annotations[item_id] = record
        st.session_state["last_saved"] = item_id
        st.session_state["save_error"] = ""
        st.session_state["case_picker"] = next_unlabelled(ids, annotations, item_id)
    except (OSError, ValueError) as exc:
        st.session_state["save_error"] = str(exc)


def render_position(position: dict, label: str) -> None:
    st.caption(position["criterion"])
    st.markdown("**Основні аргументи**")
    for argument in position["arguments"]:
        st.markdown(f"- {argument}")
    with st.expander(f"Усі релевантні твердження · {len(position['claims'])}"):
        for claim in position["claims"]:
            st.markdown(f"- {claim}")
    with st.expander(f"Джерела цієї позиції · {len(position['sources'])}"):
        render_sources(position["sources"], prefix=f"{label}-source")


def render_sources(sources: list[dict], *, prefix: str) -> None:
    if not sources:
        st.info("Для цієї позиції окремих зовнішніх джерел не зібрано.")
        return
    for index, source in enumerate(sources, start=1):
        url = source["url"]
        host = urlparse(url).netloc or "джерело"
        title = source.get("title") or host
        with st.container(border=True, key=f"{prefix}-{index}"):
            st.markdown(f"**[{title}]({url})**")
            if source.get("supports"):
                st.caption(source["supports"])


workspace = static_workspace()
ids = sorted(workspace["cases"], key=lambda value: int(value[2:]))
annotations = load_human_annotations(OUTPUT_PATH)
instructions = load_human_instructions(OUTPUT_PATH)
st.session_state.setdefault("case_picker", ids[0])
st.session_state.setdefault("last_saved", "")
st.session_state.setdefault("save_error", "")

with st.sidebar:
    st.subheader("Прогрес")
    st.progress(len(annotations) / len(ids), text=f"{len(annotations)} з {len(ids)} промарковано")
    only_unlabelled = st.toggle("Показувати лише непромарковані", value=False)
    available = [item_id for item_id in ids if not only_unlabelled or item_id not in annotations]
    if not available:
        available = ids
    if st.session_state["case_picker"] not in available:
        st.session_state["case_picker"] = available[0]
    st.selectbox(
        "Кейс",
        available,
        key="case_picker",
        format_func=lambda item_id: (
            f"{item_id} · {'готово' if item_id in annotations else 'не промарковано'}"
        ),
    )
    show_prior = st.toggle(
        "Показати попередній gold після рішення",
        value=True,
        help="До вашого першого рішення попередній gold завжди прихований.",
    )
    st.download_button(
        "Завантажити людський gold",
        data=annotations_as_jsonl(annotations),
        file_name="human_gold.jsonl",
        mime="application/x-ndjson",
        icon=":material/download:",
        disabled=not annotations,
        on_click="ignore",
        width="stretch",
    )
    st.caption(f"Автозбереження: `{OUTPUT_PATH.name}`")

item_id = st.session_state["case_picker"]
analysis = compile_case_analysis(workspace, item_id)
recommendation = recommend_verdict(workspace, item_id)
case = analysis["case"]
existing = annotations.get(item_id)

st.title("Людське маркування gold")
st.caption(
    "Спочатку перевірте статтю й джерела, потім порівняйте найсильніші аргументи "
    "за кожен клас. Попередній gold і результати F/B2/B3 до рішення приховані."
)

index = ids.index(item_id)
with st.container(horizontal=True, horizontal_alignment="distribute"):
    st.button(
        "Попередній",
        icon=":material/arrow_back:",
        disabled=index == 0,
        on_click=go_to,
        args=(ids[max(0, index - 1)],),
    )
    st.markdown(f"**{item_id} · {index + 1}/{len(ids)}**")
    st.button(
        "Наступний",
        icon=":material/arrow_forward:",
        icon_position="right",
        disabled=index == len(ids) - 1,
        on_click=go_to,
        args=(ids[min(len(ids) - 1, index + 1)],),
    )

with st.container(border=True):
    st.subheader(analysis["title"])
    if analysis.get("is_replacement"):
        st.info(
            "Цей ID отримав новий повнотекстовий матеріал замість дефектного "
            "headline-only кейсу. Старі результати F/B2/B3 та старий gold до "
            "цієї статті не застосовуються і тут не використовуються.",
            icon=":material/new_releases:",
        )
    source = analysis["source"]
    if source["url"]:
        st.link_button(
            "Відкрити матеріал",
            source["url"],
            icon=":material/open_in_new:",
        )
        if source["provenance"] == "dataset":
            st.caption("URL збережений безпосередньо в датасеті.")
        elif source["is_original"]:
            st.caption("URL відновлений пошуком за точним заголовком і доменом.")
        else:
            st.warning(
                "Оригінальний URL не відновлено; показано дзеркало з тим самим заголовком."
            )
    else:
        st.error("URL матеріалу не знайдено.")
    st.caption(
        f"Потреба в пошуку: {case['search_necessity']} · "
        f"обсяг тексту: {len(case['text']):,} символів"
    )
    with st.expander("Повний текст матеріалу"):
        st.write(case["text"])

st.subheader("Аргументи за кожну позицію")
reliable_tab, questionable_tab, unreliable_tab = st.tabs(
    ["За reliable", "За questionable", "За unreliable"]
)
with reliable_tab:
    render_position(analysis["positions"]["reliable"], "reliable")
with questionable_tab:
    render_position(analysis["positions"]["questionable"], "questionable")
with unreliable_tab:
    render_position(analysis["positions"]["unreliable"], "unreliable")

source_count = len(analysis["verification_sources"])
with st.expander(f"Усі зібрані джерела для ручної перевірки · {source_count}"):
    render_sources(analysis["verification_sources"], prefix=f"{item_id}-all")

st.subheader("Ваш вердикт та інструкції")
instruction_key = f"instruction_{item_id}"
verdict_key = f"verdict_{item_id}"
saved_instruction = instructions.get(item_id, {}).get("instruction") or ""
st.session_state.setdefault(instruction_key, saved_instruction)
st.session_state.setdefault(
    verdict_key,
    existing["final_verdict"] if existing else recommendation["verdict"],
)

recommended_position = analysis["positions"][recommendation["verdict"]]
with st.container(border=True):
    st.markdown(
        f"**Моя рекомендація:** `{recommendation['verdict']}` — "
        f"{LABEL_UI[recommendation['verdict']]}"
    )
    st.write(recommendation["reason"])
    for argument in recommended_position["arguments"][:3]:
        st.markdown(f"- {argument}")
    st.caption(
        "Це стартова рекомендація, а не gold. Для замінених кейсів вона базується "
        "на новій ручній вебперевірці; для решти B3 не враховується, бо він не має "
        "вебпошуку. Наявність чисел не є негативним сигналом: важить лише результат "
        "їх перевірки. Рекомендацію можна змінити нижче перед збереженням."
    )

st.segmented_control(
    "Ваш вердикт",
    options=list(LABELS),
    format_func=lambda verdict: LABEL_UI[verdict],
    selection_mode="single",
    key=verdict_key,
    width="stretch",
)

st.text_area(
    "Опційний коментар / інструкція для асистента",
    key=instruction_key,
    height=120,
    placeholder=(
        "Наприклад: перевірити конкретне твердження або врахувати контекст. "
        "Можна залишити порожнім."
    ),
    help="Цей текст зберігається лише в журналі інструкцій і не входить у gold.",
)
st.info(
    "У `human_gold.jsonl` потрапляють лише ID кейсу та ваш вердикт. "
    "Текст вище — окрема робоча інструкція, а не обґрунтування датасету.",
    icon=":material/info:",
)

st.button(
    "Зберегти вердикт і перейти далі",
    key=f"save_{item_id}",
    type="primary",
    icon=":material/save:",
    width="stretch",
    shortcut="enter",
    on_click=record_decision,
    args=(item_id, ids),
)

if st.session_state.get("save_error"):
    st.error(f"Не вдалося зберегти рішення: {st.session_state['save_error']}")
elif st.session_state.get("last_saved") == item_id:
    st.success("Рішення збережено; попередня версія лишилася в історії змін.")

if not analysis.get("is_replacement"):
    system_rows = []
    for mode in ("F", "B2", "B3"):
        run = workspace["runs"].get(item_id, {}).get(mode, {})
        label = run.get("label")
        if label:
            probabilities = run.get("probabilities") or {}
            system_rows.append(
                {
                    "Режим": mode,
                    "Мітка": label,
                    "Ймовірність класу": probabilities.get(label),
                }
            )
    if system_rows:
        with st.container(border=True):
            st.markdown("**Вердикти системних режимів (F/B2/B3) з прогону**")
            st.caption(
                "Мітки системи з експериментального прогону — довідково, не gold. "
                "Для замінених кейсів не показуються, бо стосуються старої статті."
            )
            st.table(system_rows)

if existing:
    with st.container(border=True):
        st.markdown(f"**Ваш поточний gold:** `{existing['final_verdict']}`")
        if saved_instruction:
            st.caption("Окремо збережена робоча інструкція:")
            st.write(saved_instruction)

if existing and show_prior and not analysis.get("is_replacement"):
    with st.expander("Порівняння з попередніми мітками"):
        prior = workspace["final"].get(item_id)
        if prior:
            st.markdown(f"Попередня мітка: `{prior['final_verdict']}`")
            st.caption(f"Метод: {prior.get('method', 'невідомо')}")
            if prior.get("reason"):
                st.write(prior["reason"])
