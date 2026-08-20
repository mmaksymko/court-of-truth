from pathlib import Path

import pytest

from court.forensics import models
from tests.fakes import make_detector


def _registry():
    return {
        "clickbait": make_detector("clickbait", "title", ("clickbait", "neutral"), "clickbait", 0.2)
    }


def test_local_registry_is_validated(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(models, "load_registry", lambda path, enabled: _registry())
    registry = models.ensure_registry(tmp_path, ["clickbait"])
    assert set(registry) == {"clickbait"}


def test_artifact_set_mismatch_fails(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(models, "load_registry", lambda path, enabled: {})
    with pytest.raises(RuntimeError, match="artifact set mismatch"):
        models.ensure_registry(tmp_path, ["clickbait"])


def test_extra_artifact_fails_even_when_detector_subset_is_enabled(monkeypatch, tmp_path: Path):
    registry = {
        **_registry(),
        "ai_generated": make_detector("ai_generated", "body", ("human", "ai"), "ai", 0.4),
    }
    monkeypatch.setattr(models, "load_registry", lambda path, enabled: registry)
    with pytest.raises(RuntimeError, match=r"extra=\['ai_generated'\]"):
        models.ensure_registry(
            tmp_path,
            ["clickbait"],
            enabled_detectors=["clickbait"],
        )


def test_enabled_detector_subset_is_returned_after_full_validation(monkeypatch, tmp_path: Path):
    registry = {
        **_registry(),
        "ai_generated": make_detector("ai_generated", "body", ("human", "ai"), "ai", 0.4),
    }
    monkeypatch.setattr(models, "load_registry", lambda path, enabled: registry)
    selected = models.ensure_registry(
        tmp_path,
        ["clickbait", "ai_generated"],
        enabled_detectors=["clickbait"],
    )
    assert set(selected) == {"clickbait"}


def test_unknown_enabled_detector_fails(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(models, "load_registry", lambda path, enabled: _registry())
    with pytest.raises(RuntimeError, match="unknown enabled detectors"):
        models.ensure_registry(
            tmp_path,
            ["clickbait"],
            enabled_detectors=["missing"],
        )
