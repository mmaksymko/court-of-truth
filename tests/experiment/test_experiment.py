import asyncio
import json
from pathlib import Path

import pytest

from court.experiment.corpus import load_gold, load_items
from court.experiment.experiment import run_experiment, summarize, write_outputs
from court.experiment.modes import B1, FULL
from court.experiment.runner import ItemInput
from tests.fakes import FakeLLM
from tests.tribunal.support import report


def _write_cases(path: Path) -> None:
    rows = [
        {
            "id": "TC01",
            "title": "Заголовок 1",
            "text": "Текст один",
            "source_url": "https://a.example",
        },
        {"id": "TC02", "title": "", "text": "Текст два", "source_url": None},
    ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _write_gold(path: Path) -> None:
    rows = [
        {"id": "TC01", "final_verdict": "questionable"},
        {"id": "TC02", "final_verdict": "reliable"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_load_items_and_gold(tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    gold_path = tmp_path / "gold.jsonl"
    _write_cases(cases)
    _write_gold(gold_path)
    items = load_items(cases)
    gold = load_gold(gold_path)
    assert [i.id for i in items] == ["TC01", "TC02"]
    assert items[0].source_url == "https://a.example"
    assert items[1].title == ""
    assert gold == {"TC01": "questionable", "TC02": "reliable"}


def test_load_gold_rejects_unknown_verdict(tmp_path: Path):
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps({"id": "TC01", "final_verdict": "maybe"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown verdict"):
        load_gold(path)


def test_run_experiment_end_to_end(tmp_path: Path):
    items = [ItemInput(id="TC01", title="t", text="x", gold="questionable")]

    async def report_for(_item: ItemInput):
        return report()

    records = asyncio.run(
        run_experiment(
            items,
            [FULL, B1],
            report_for=report_for,
            llm_for=lambda _mode: FakeLLM(),
            repeats=2,
            include_swapped=True,
        )
    )
    # 1 item x 2 modes x (2 repeats + 1 swap) = 6 records.
    assert len(records) == 6
    assert {r.variant for r in records} == {"AB#1", "AB#2", "BA#1"}

    gold = {"TC01": "questionable"}
    summary = summarize(records, gold)
    assert set(summary) == {"F", "B1"}
    runs_path, summary_path = write_outputs(tmp_path / "out", records, gold)
    assert len(runs_path.read_text(encoding="utf-8").splitlines()) == 6
    assert json.loads(summary_path.read_text(encoding="utf-8"))["F"]["mode"] == "F"
