"""Frozen experiment protocol: the single descriptor that pins a keyed run.

Serialized as JSON (no undeclared YAML dependency) and validated with pydantic.
It records the model, seed, repeat/order plan, the modes, the corpus and search
transcript hashes and the frozen code commit, so every reported number maps back
to one reproducible configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path


class ExperimentProtocol(BaseModel):
    model: str
    seed: int = 42
    repeats: int = Field(default=1, ge=1)
    include_swapped: bool = False
    modes: list[str] = Field(default_factory=lambda: ["F", "B1", "B2", "B3"])
    search_context: dict[str, str] = Field(default_factory=dict)
    corpus_path: str = ""
    corpus_sha256: str = ""
    search_transcript_sha256: str | None = None
    code_commit: str = ""
    thresholds: dict[str, float] = Field(default_factory=dict)
    notes: str = ""

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ExperimentProtocol:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
