import hashlib
import json
from pathlib import Path

from court.forensics.hashing import path_sha256

ROOT = Path(__file__).resolve().parents[2]


def test_clickbait_manifest_matches_frozen_model_and_dataset() -> None:
    artifact_dir = ROOT / "artifacts" / "clickbait"
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    model_config = json.loads((artifact_dir / "model" / "config.json").read_text())
    positive_label = manifest["positive_label"]
    positive_index = manifest["hyperparams"]["positive_index"]

    assert manifest["label_map"][positive_label] == positive_index
    assert model_config["id2label"][str(positive_index)] == positive_label
    assert model_config["label2id"][positive_label] == positive_index
    assert manifest["label_map"] == model_config["label2id"]
    assert manifest["hyperparams"]["threshold_selection"] == (
        "threshold closest to zero logit-difference among those within 0.005 "
        "of the validation macro-F1 maximum"
    )
    assert manifest["model_sha256"] == path_sha256(artifact_dir / "model")

    source = ROOT / "data" / "clickbait.csv"
    assert manifest["source_csv_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
