import json
from pathlib import Path

from court.annotation.workbench import (
    compile_case_analysis,
    load_human_annotations,
    load_static_workspace,
    recommend_verdict,
    save_human_annotation,
    source_info,
)

ROOT = Path(__file__).parents[2]


def test_every_case_has_an_article_url_after_recovery():
    workspace = load_static_workspace(ROOT)

    assert len(workspace["cases"]) == 60
    assert all(source_info(workspace, item_id)["url"] for item_id in workspace["cases"])
    assert source_info(workspace, "TC53")["is_original"] is True


def test_replacement_cases_are_full_text_and_use_fresh_manual_reviews():
    workspace = load_static_workspace(ROOT)
    reviews = workspace["replacement_reviews"]

    # 24 prior replacements plus 2 newly replaced IDs; three other IDs were replaced again.
    assert len(reviews) == 26
    assert {
        "TC08",
        "TC22",
        "TC25",
        "TC28",
        "TC30",
        "TC31",
        "TC45",
        "TC51",
        "TC57",
        "TC60",
    } <= set(reviews)

    for item_id, review in reviews.items():
        case = workspace["cases"][item_id]
        analysis = compile_case_analysis(workspace, item_id)
        recommendation = recommend_verdict(workspace, item_id)

        assert len(case["text"]) >= 900
        assert case["source_url"].startswith("http")
        assert analysis["is_replacement"] is True
        assert recommendation["verdict"] == review["recommendation"]
        assert recommendation["votes"] == [
            {"source": "ручна вебперевірка", "verdict": review["recommendation"]}
        ]
        rendered = json.dumps(analysis, ensure_ascii=False)
        assert "recovered-exact-title-search" not in rendered


def test_case_analysis_has_all_three_positions_without_exposing_labels():
    workspace = load_static_workspace(ROOT)
    analysis = compile_case_analysis(workspace, "TC03")

    assert set(analysis["positions"]) == {"reliable", "questionable", "unreliable"}
    assert all(analysis["positions"][label]["arguments"] for label in analysis["positions"])
    rendered = json.dumps(analysis["positions"], ensure_ascii=False)
    assert '"final_verdict"' not in rendered


def test_recommendation_does_not_consult_previous_gold():
    workspace = load_static_workspace(ROOT)
    expected = recommend_verdict(workspace, "TC03")
    workspace["final"]["TC03"]["final_verdict"] = "unreliable"
    workspace["runs"]["TC03"]["B3"]["label"] = "unreliable"

    assert recommend_verdict(workspace, "TC03") == expected


def test_human_annotation_overwrites_latest_and_keeps_history(tmp_path):
    output = tmp_path / "human_gold.jsonl"
    source = {"url": "https://example.com/article", "provenance": "test"}
    save_human_annotation(
        output,
        item_id="TC01",
        verdict="questionable",
        instruction="Перевірити ключове твердження ще в одному джерелі.",
        source=source,
    )
    save_human_annotation(
        output,
        item_id="TC01",
        verdict="reliable",
        instruction="Врахувати два незалежні підтвердження.",
        source=source,
    )

    latest = load_human_annotations(output)
    history = output.with_name("human_gold_history.jsonl").read_text(encoding="utf-8")
    assert latest["TC01"] == {"id": "TC01", "final_verdict": "reliable"}
    assert "instruction" not in output.read_text(encoding="utf-8")
    assert "Врахувати два незалежні підтвердження." in history
    assert len(history.splitlines()) == 2
