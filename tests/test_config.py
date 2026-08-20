import pytest
from pydantic import ValidationError

from court.config import Settings


def test_csv_settings_parse_environment(monkeypatch):
    monkeypatch.setenv("COURT_ENABLED_DETECTORS", "jeansa, clickbait")
    assert Settings(_env_file=None).enabled_detectors == ["jeansa", "clickbait"]


def test_invalid_log_level_rejected():
    with pytest.raises(ValidationError, match="log_level"):
        Settings(_env_file=None, log_level="verbose")
