from pathlib import Path

from court.experiment.protocol import ExperimentProtocol


def test_protocol_roundtrips(tmp_path: Path):
    protocol = ExperimentProtocol(
        model="gpt-x",
        seed=42,
        repeats=3,
        include_swapped=True,
        search_context={"F": "medium", "B3": "low"},
        corpus_path="evals/cases.jsonl",
        corpus_sha256="abc123",
        gold_labels_path="evals/annotation/human_gold.jsonl",
        gold_labels_key="final_verdict",
        gold_labels_sha256="gold-new",
        previous_gold_labels_sha256="gold-old",
        judge_prompt_version="v4-central-claim",
        judge_prompt_sha256={"F": "prompt-f", "B1": "prompt-b1"},
        status="pending-rerun-after-gold-revision-2026-08-30",
        search_transcript_sha256="def456",
        code_commit="deadbeef",
        thresholds={"clickbait": 0.7149},
    )
    path = tmp_path / "protocol.json"
    protocol.save(path)
    loaded = ExperimentProtocol.load(path)
    assert loaded == protocol
    assert loaded.modes == ["F", "B1", "B2", "B3"]
    assert loaded.search_context["F"] == "medium"
    assert loaded.gold_labels_sha256 == "gold-new"
    assert loaded.previous_gold_labels_sha256 == "gold-old"
    assert loaded.judge_prompt_version == "v4-central-claim"
    assert loaded.judge_prompt_sha256["F"] == "prompt-f"
