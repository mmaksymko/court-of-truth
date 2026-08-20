from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import (
    AliasChoices,
    BeforeValidator,
    Field,
    field_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}


def _split_csv(value: object) -> object:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


CsvList = Annotated[list[str], NoDecode, BeforeValidator(_split_csv)]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="COURT_",
        env_file=(".env", ".env.local"),
        extra="ignore",
        populate_by_name=True,
    )

    artifacts_dir: Path = Path("artifacts")
    enabled_detectors: CsvList = []
    detector_workers: int = Field(default=5, gt=0, le=16)
    low_confidence_margin: float = Field(default=0.1, ge=0, le=1)

    max_body_bytes: int = Field(default=1_000_000, gt=0)
    fetch_max_bytes: int = Field(default=2_000_000, gt=0)
    fetch_timeout_s: float = Field(default=15.0, gt=0)
    fetch_max_redirects: int = Field(default=3, ge=0, le=10)

    operation_rate_per_minute: int = Field(default=10, gt=0)
    operation_concurrency: int = Field(default=4, gt=0)
    uvicorn_limit_concurrency: int = Field(default=32, gt=0)
    port: int = Field(default=8000, gt=0, le=65535, validation_alias=AliasChoices("PORT"))
    cors_origins: CsvList = []
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        level = value.upper()
        if level not in _LEVELS:
            raise ValueError(f"unknown log_level {value!r}")
        return level


@lru_cache
def get_settings() -> Settings:
    return Settings()
