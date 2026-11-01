from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_gold_annotation_app_renders_blind_decision_workflow(tmp_path, monkeypatch):
    output_path = tmp_path / "human_gold.jsonl"
    monkeypatch.setenv("COURT_HUMAN_GOLD_PATH", str(output_path))
    app = AppTest.from_file(Path(__file__).parents[1] / "gold_annotation_app.py").run()

    assert not app.exception
    assert app.title[0].value == "Людське маркування gold"
    labels = [button.label for button in app.button]
    assert labels.count("Зберегти вердикт і перейти далі") == 1
    assert app.segmented_control[0].value in {"reliable", "questionable", "unreliable"}
    assert "Попередня мітка" not in " ".join(markdown.value for markdown in app.markdown)

    instruction = "Перевірити часовий контекст перед фінальним використанням."
    app.text_area[0].input(instruction).run()
    app.segmented_control[0].set_value("reliable").run()
    next(
        button for button in app.button if button.label == "Зберегти вердикт і перейти далі"
    ).click().run()

    assert not app.exception
    assert output_path.exists()
    gold = output_path.read_text(encoding="utf-8")
    history = output_path.with_name("human_gold_history.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"final_verdict": "reliable"' in gold
    assert instruction not in gold
    assert instruction in history
