import pytest
from pydantic import TypeAdapter, ValidationError

from court.forensics.manifest import Manifest
from court.forensics.schemas import AnalyzeRequest, DetectorResult, OkResult, SkippedResult

_adapter = TypeAdapter(DetectorResult)


def test_discriminator_parses_ok():
    parsed = _adapter.validate_python(
        {
            "status": "ok",
            "id": "jeansa",
            "scope": "body",
            "label": "sponsored",
            "probability": 0.9,
            "flagged": True,
            "low_confidence": False,
        }
    )
    assert isinstance(parsed, OkResult)


def test_discriminator_parses_skipped():
    parsed = _adapter.validate_python(
        {"status": "skipped", "id": "clickbait", "scope": "title", "reason": "no_text"}
    )
    assert isinstance(parsed, SkippedResult)


def test_skipped_does_not_carry_label():
    dumped = SkippedResult(id="x", scope="body", reason="no_text").model_dump()
    assert "label" not in dumped
    assert "probability" not in dumped


def test_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AnalyzeRequest(text="норм текст", unexpected="x")


def test_probability_bounds_enforced():
    with pytest.raises(ValidationError):
        OkResult(
            id="x", scope="body", label="y", probability=1.5, flagged=True, low_confidence=False
        )


def test_detector_result_list_roundtrip():
    items = [
        OkResult(
            id="a", scope="body", label="x", probability=0.9, flagged=True, low_confidence=False
        ),
        SkippedResult(id="b", scope="title", reason="no_text"),
    ]
    adapter = TypeAdapter(list[DetectorResult])
    reparsed = adapter.validate_python(adapter.dump_python(items))
    assert isinstance(reparsed[0], OkResult)
    assert isinstance(reparsed[1], SkippedResult)


def test_manifest_requires_sha256_model_hash():
    with pytest.raises(ValidationError, match="model_sha256"):
        Manifest.model_validate(
            {
                "detector_id": "x",
                "version": "x",
                "trained_at": "x",
                "backend": "sklearn",
                "sklearn_version": "x",
                "numpy_version": "x",
                "scipy_version": "x",
                "python_version": "x",
                "seed": 42,
                "hyperparams": {},
                "source_csv_sha256": "x",
                "model_sha256": "",
                "n_train": 1,
                "n_val": 1,
                "n_test": 1,
                "labels": ["no", "yes"],
                "label_map": {"no": 0, "yes": 1},
                "positive_label": "yes",
                "scope": "body",
                "decision_threshold": 0.5,
                "metrics": {},
                "per_class_recall": {},
            }
        )
