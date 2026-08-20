from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from court.forensics.schemas import Backend, Scope


class DetectorConfig(BaseModel):
    id: str
    source_csv: Path
    text_field: str
    label_field: str
    scope: Scope
    positive_label: str
    backend: Backend = "sklearn"
    analyzer: str = "char_wb"
    ngram: tuple[int, int] = (3, 5)
    min_df: int = Field(default=3, ge=1)
    max_features: int = Field(default=200_000, gt=0)
    c: float = Field(default=1.0, gt=0)
    class_weight: str | None = None
    split_field: str = ""
    group_field: str = ""
    threshold_objective: str = "macro_f1"
    target_recall: float = Field(default=0.95, ge=0, le=1)
    threshold_candidates: str = "probability_grid"
    model_name: str = ""
    epochs: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    batch_size: int = Field(default=16, ge=1)
    max_length: int = Field(default=64, ge=8)
    weight_decay: float = Field(default=0.01, ge=0)
    threshold_plateau_tolerance: float = Field(default=0.005, ge=0)

    @model_validator(mode="after")
    def _check_ngram(self) -> "DetectorConfig":
        if self.ngram[0] > self.ngram[1] or self.ngram[0] < 1:
            raise ValueError(f"bad ngram range {self.ngram}")
        if self.threshold_objective not in {"macro_f1", "target_recall"}:
            raise ValueError(f"bad threshold objective {self.threshold_objective!r}")
        if self.threshold_candidates not in {"probability_grid", "exact"}:
            raise ValueError(f"bad threshold candidates {self.threshold_candidates!r}")
        if self.backend == "transformers" and not self.model_name:
            raise ValueError("transformers backend requires model_name")
        return self
