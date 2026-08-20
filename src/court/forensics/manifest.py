from pydantic import BaseModel, Field, model_validator

from court.forensics.schemas import Backend, Scope

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Manifest(BaseModel):
    detector_id: str
    version: str
    trained_at: str
    backend: Backend
    sklearn_version: str
    numpy_version: str
    scipy_version: str
    python_version: str
    seed: int
    hyperparams: dict[str, object]
    source_csv_sha256: str
    model_sha256: str = Field(pattern=_SHA256_PATTERN)
    split_field: str = ""
    n_train: int
    n_val: int
    n_test: int
    labels: tuple[str, str]
    label_map: dict[str, int]
    positive_label: str
    scope: Scope
    decision_threshold: float = Field(ge=0, le=1)
    metrics: dict[str, float]
    per_class_recall: dict[str, float]
    language_note: str = ""
    git_commit: str = ""

    @model_validator(mode="after")
    def _check_positive(self) -> "Manifest":
        if self.positive_label not in self.labels:
            raise ValueError(f"positive_label {self.positive_label!r} not in {self.labels}")
        return self
