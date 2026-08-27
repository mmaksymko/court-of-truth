from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_analysis_form():
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py").run()

    assert not app.exception
    assert app.title[0].value == "Аналіз новинного тексту"
    assert app.segmented_control[0].label == "Джерело"
    assert app.button[0].label == "Аналізувати"
