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
