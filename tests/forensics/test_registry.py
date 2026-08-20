from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sklearn

from court.forensics.backends import transformers_backend
from court.forensics.hashing import path_sha256
from court.forensics.registry import load_registry
from tests.forensics.registry_support import manifest, write_artifact


def test_load_registry_and_predict(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__))
    detector = load_registry(tmp_path)["tiny"]
    assert detector.predict("buy now spam offer deal") > detector.predict("clean neutral news")


def test_unknown_detector_raises(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__))
    registry = load_registry(tmp_path)
    with pytest.raises(KeyError):
        registry["missing"]


def test_version_guard_raises_on_mismatch(tmp_path: Path):
    write_artifact(tmp_path, manifest("0.0.0"))
    with pytest.raises(RuntimeError, match="version mismatch"):
        load_registry(tmp_path)


def test_python_minor_guard(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__, python_version="3.11.0"))
    with pytest.raises(RuntimeError, match="python minor mismatch"):
        load_registry(tmp_path)


def test_model_hash_mismatch_raises(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__), valid_hash=False)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_registry(tmp_path)


def test_model_hash_match_loads(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__))
    assert load_registry(tmp_path)["tiny"]


def test_enabled_detectors_filter(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__))
    with pytest.raises(RuntimeError, match="no detectors"):
        load_registry(tmp_path, ["other"])


def test_empty_artifacts_dir_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="no detectors"):
        load_registry(tmp_path)


def test_registry_is_immutable(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__))
    registry = load_registry(tmp_path)
    with pytest.raises(TypeError):
        registry["x"] = None  # type: ignore[index]


def test_evidence_with_positive_index_zero(tmp_path: Path):
    # positive_label "neg" is classes_[0], exercising the -weights direction
    artifact_manifest = manifest(sklearn.__version__).model_copy(update={"positive_label": "neg"})
    write_artifact(tmp_path, artifact_manifest)
    tokens = load_registry(tmp_path)["tiny"].evidence("clean neutral news report")
    assert tokens


def test_evidence_reachable_after_calibration(tmp_path: Path):
    write_artifact(tmp_path, manifest(sklearn.__version__))
    tokens = load_registry(tmp_path)["tiny"].evidence("buy now spam offer deal")
    assert tokens
    assert all(isinstance(t, str) for t in tokens)


def test_transformer_registry_loads_local_model(tmp_path: Path, monkeypatch):
    artifact_dir = tmp_path / "tiny"
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"weights")
    artifact_manifest = manifest(sklearn.__version__).model_copy(
        update={
            "backend": "transformers",
            "hyperparams": {"max_length": 32, "positive_index": 1},
        }
    )
    artifact_manifest = artifact_manifest.model_copy(
        update={"model_sha256": path_sha256(model_dir)}
    )
    (artifact_dir / "manifest.json").write_text(artifact_manifest.model_dump_json())
    bundle = object()
    monkeypatch.setattr(transformers_backend, "load", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(transformers_backend, "predict_proba", lambda model, text: 0.75)

    detector = load_registry(tmp_path)["tiny"]

    assert detector.predict("headline") == 0.75
    assert detector.evidence("headline") == []


def test_transformer_evidence_by_occlusion(tmp_path: Path, monkeypatch):
    artifact_dir = tmp_path / "tiny"
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "weights.bin").write_bytes(b"weights")
    artifact_manifest = manifest(sklearn.__version__).model_copy(
        update={"backend": "transformers", "hyperparams": {"max_length": 32, "positive_index": 1}}
    )
    artifact_manifest = artifact_manifest.model_copy(
        update={"model_sha256": path_sha256(model_dir)}
    )
    (artifact_dir / "manifest.json").write_text(artifact_manifest.model_dump_json())
    monkeypatch.setattr(transformers_backend, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        transformers_backend,
        "predict_proba",
        lambda model, text: 0.9 if "shocking" in text else 0.2,
    )

    detector = load_registry(tmp_path)["tiny"]

    # only removing "shocking" drops the probability, so it is the sole evidence word;
    # edge punctuation is stripped ("shocking!" -> "shocking")
    assert detector.evidence("this shocking! headline") == ["shocking"]


def test_jeansa_evidence_uses_promo_signals(tmp_path: Path):
    artifact_manifest = manifest(sklearn.__version__).model_copy(update={"detector_id": "jeansa"})
    write_artifact(tmp_path, artifact_manifest)
    signals = load_registry(tmp_path)["jeansa"].evidence(
        "Замовити ексклюзивний товар зі знижкою 50%! Найкращий вибір, телефонуйте."
    )
    assert signals
    assert all(isinstance(signal, str) for signal in signals)
    assert any("знижк" in signal.lower() or "ціни" in signal.lower() for signal in signals)


def test_transformer_backend_probability():
    class FakeTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": text}

    class FakeModel:
        def __call__(self, **encoded):
            return SimpleNamespace(logits=encoded["input_ids"])

    class FakeTorch:
        @staticmethod
        def no_grad():
            return nullcontext()

        @staticmethod
        def softmax(logits, dim):
            assert dim == -1
            return np.asarray([[0.2, 0.8]])

    bundle = transformers_backend.TransformerBundle(
        FakeTokenizer(),
        FakeModel(),
        FakeTorch(),
        64,
        1,
    )
    assert transformers_backend.predict_proba(bundle, "headline") == pytest.approx(0.8)
