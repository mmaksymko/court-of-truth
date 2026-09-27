import asyncio
import json
from pathlib import Path

from court.experiment.modes import B1, B2, FULL, MODES
from court.experiment.runner import (
    ItemInput,
    evaluate,
    run_corpus,
    run_item,
    run_repeats,
    write_records_jsonl,
)
from court.tribunal.errors import TribunalError
from court.tribunal.schemas import Verdict
from tests.fakes import FakeLLM
from tests.tribunal.support import report


def _item(item_id: str, gold: str = "questionable") -> ItemInput:
    return ItemInput(id=item_id, title="Заголовок", text="Текст новини", gold=gold)


def test_all_four_modes_are_registered():
    assert set(MODES) == {"F", "B1", "B2", "B3"}


def test_b3_mode_has_no_search():
    assert MODES["B3"].search_context is None
    assert MODES["F"].search_context == "medium"


def test_run_item_full_mode_completes():
    record = asyncio.run(run_item(_item("a"), FULL, report(), FakeLLM()))
    assert record.status == "ok"
    assert record.label == "questionable"
    assert record.probabilities is not None
    assert abs(sum(record.probabilities.values()) - 1.0) < 0.01


def test_run_item_neutral_mode_uses_single_researcher():
    record = asyncio.run(run_item(_item("a"), B1, report(), FakeLLM()))
    assert record.status == "ok"
    assert record.error_code is None


def test_run_item_without_forensics_completes():
    record = asyncio.run(run_item(_item("a"), B2, report(), FakeLLM()))
    assert record.status == "ok"


def test_run_item_maps_tribunal_error_to_failed_record():
    class FailingJudge(FakeLLM):
        async def judge(self, _user: str) -> Verdict:
            raise TribunalError(502, "tribunal_failed", "boom")

    record = asyncio.run(run_item(_item("a"), FULL, report(), FailingJudge()))
    assert record.status == "failed"
    assert record.label is None
    assert record.error_code == "tribunal_failed"


def test_run_corpus_and_evaluate():
    items = [_item("a", "questionable"), _item("b", "reliable")]

    async def report_for(_item: ItemInput):
        return report()

    records = asyncio.run(
        run_corpus(items, [FULL, B1], report_for=report_for, llm_for=lambda _mode: FakeLLM())
    )
    assert len(records) == 4

    gold = {"a": "questionable", "b": "reliable"}
    summary = evaluate(records, gold)
    assert set(summary) == {"F", "B1"}
    full = summary["F"]
    assert full.outcomes.total == 2
    assert full.outcomes.completed == 2
    # FakeLLM always predicts "questionable": correct on "a", wrong on "b".
    assert full.itt_accuracy == 0.5
    assert 0.0 <= full.macro_f1 <= 1.0
    assert 0.0 <= full.brier <= 2.0


def test_run_item_captures_telemetry_positions_and_evidence():
    record = asyncio.run(run_item(_item("a"), FULL, report(), FakeLLM()))
    assert record.elapsed_s is not None and record.elapsed_s >= 0
    # Two parties plus the judge each report usage in the fake client.
    assert record.usage is not None
    assert record.usage["total_tokens"] == 45
    assert record.forensic_report is not None
    assert record.forensic_report["risk"]["flagged_count"] == 0
    assert record.verdict is not None
    assert record.verdict["label"] == record.label
    assert record.arguments is not None and len(record.arguments) == 2
    assert record.evidence is not None and len(record.evidence) >= 1
    assert record.searches is not None and len(record.searches) >= 1
    assert record.searches[0]["sources"]


def test_failed_run_still_records_telemetry():
    class FailingJudge(FakeLLM):
        async def judge(self, _user: str) -> Verdict:
            raise TribunalError(502, "tribunal_failed", "boom")

    record = asyncio.run(run_item(_item("a"), FULL, report(), FailingJudge()))
    assert record.status == "failed"
    assert record.usage is not None and record.usage["total_tokens"] > 0
    assert record.forensic_report is not None
    assert record.arguments is None


def test_write_records_jsonl_roundtrip(tmp_path: Path):
    records = asyncio.run(
        run_corpus(
            [_item("a"), _item("b")],
            [FULL],
            report_for=lambda _item: _ready_report(),
            llm_for=lambda _mode: FakeLLM(),
        )
    )
    path = tmp_path / "runs.jsonl"
    write_records_jsonl(path, records)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = json.loads(lines[0])
    assert parsed["item_id"] == "a"
    assert parsed["mode"] == "F"
    assert parsed["usage"]["total_tokens"] == 45
    assert parsed["forensic_report"]["results"]
    assert parsed["verdict"]["rationale"]
    assert parsed["arguments"] and parsed["searches"]


async def _ready_report():
    return report()


def test_run_repeats_tags_variants_and_includes_swap():
    records = asyncio.run(
        run_repeats(_item("a"), FULL, report(), FakeLLM(), repeats=2, include_swapped=True)
    )
    assert [record.variant for record in records] == ["AB#1", "AB#2", "BA#1"]
    assert all(record.status == "ok" for record in records)


def test_evaluate_counts_failures_in_outcomes():
    class FailingJudge(FakeLLM):
        async def judge(self, _user: str) -> Verdict:
            raise TribunalError(502, "tribunal_failed", "boom")

    record = asyncio.run(run_item(_item("a"), FULL, report(), FailingJudge()))
    summary = evaluate([record], {"a": "questionable"})
    assert summary["F"].outcomes.failures == 1
    assert summary["F"].itt_accuracy == 0.0


def test_append_record_jsonl_streams_and_flushes(tmp_path):
    from court.experiment.runner import RunRecord, append_record_jsonl

    path = tmp_path / "runs.jsonl"
    path.write_text("", encoding="utf-8")
    for i in range(3):
        append_record_jsonl(path, RunRecord(f"TC{i}", "F", "ok", "reliable", None, None))
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3  # each record persisted immediately, in order
    import json

    assert [json.loads(line)["item_id"] for line in lines] == ["TC0", "TC1", "TC2"]


def test_run_repeats_emits_each_record_via_callback():
    import asyncio

    from tests.fakes import FakeLLM
    from tests.tribunal.support import report as _report

    seen = []
    asyncio.run(
        run_repeats(_item("a"), FULL, _report(), FakeLLM(), repeats=2, on_record=seen.append)
    )
    assert [r.variant for r in seen] == ["AB#1", "AB#2"]  # streamed live, not just returned


def test_run_item_journals_evidence_snapshots_when_archive_given():
    class RecordingArchive:
        def __init__(self) -> None:
            self.urls: list[str] = []

        async def snapshot_all(self, urls):
            self.urls = list(urls)
            return [{"url": url, "status": "saved"} for url in self.urls]

    archive = RecordingArchive()
    record = asyncio.run(run_item(_item("a"), FULL, report(), FakeLLM(), archive=archive))
    assert archive.urls
    assert record.evidence is not None
    assert archive.urls == [entry["url"] for entry in record.evidence]
    assert record.evidence_snapshots == [{"url": url, "status": "saved"} for url in archive.urls]


def test_run_item_without_archive_keeps_snapshots_empty():
    record = asyncio.run(run_item(_item("a"), FULL, report(), FakeLLM()))
    assert record.evidence_snapshots is None
