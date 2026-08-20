import argparse
import tomllib
from pathlib import Path

import joblib
import pandas as pd

from court.forensics.manifest import Manifest
from court.forensics.text_transforms import transform_for
from training.config import DetectorConfig
from training.evaluation import calibrate, evaluate, gold_recall, split_data, tune_threshold
from training.provenance import build_manifest
from training.transfer.dedup import decontaminate_against_external, deduplicate_cross_split
from training.transformer import train_transformer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
SEED = 42


def train_one(cfg: DetectorConfig, out_dir: Path) -> Manifest:
    source = DATA_ROOT / cfg.source_csv
    frame = pd.read_csv(source)
    frame = frame[[cfg.text_field, cfg.label_field, *_split_cols(cfg, frame)]].dropna(
        subset=[cfg.text_field, cfg.label_field]
    )
    frame["_x"] = frame[cfg.text_field].map(lambda t: transform_for(cfg.id, t))
    frame = frame[frame["_x"].str.len() > 0]

    labels = sorted(frame[cfg.label_field].unique())
    if len(labels) != 2:
        raise ValueError(f"{cfg.id}: expected exactly 2 labels, got {labels}")

    if cfg.positive_label not in labels:
        raise ValueError(f"{cfg.id}: positive label {cfg.positive_label!r} is absent")

    train, val, test = split_data(cfg, frame, SEED)
    parts, dedup_audit = deduplicate_cross_split(
        {"train": train, "val": val, "test": test},
        text_field=cfg.text_field,
    )
    if cfg.id == "jeansa":
        gold_text = pd.read_csv(
            DATA_ROOT / "jeansa_gold.csv",
            usecols=[cfg.text_field],
        ).dropna()
        parts, gold_dedup_audit = decontaminate_against_external(
            parts,
            gold_text.rename(columns={cfg.text_field: "text"})[["text"]],
            text_field=cfg.text_field,
        )
        print(
            f"{cfg.id} dedup={dedup_audit['removed_rows']} "
            f"gold_protected={gold_dedup_audit['removed_rows']}",
            flush=True,
        )
    else:
        print(f"{cfg.id} dedup={dedup_audit['removed_rows']}", flush=True)
    train, val, test = (parts[name] for name in ("train", "val", "test"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.backend == "transformers":
        result = train_transformer(cfg, train, val, test, out_dir, SEED)
        threshold = result.threshold
        metrics = result.metrics
        per_class = result.per_class_recall
        model_path = out_dir / "model"
    else:
        model = calibrate(cfg, train, SEED)
        positive_index = list(model.classes_).index(cfg.positive_label)
        threshold = tune_threshold(model, val, cfg, positive_index)
        metrics, per_class = evaluate(model, test, cfg, positive_index, threshold)
        if cfg.id == "jeansa":
            metrics["gold_recall"] = gold_recall(
                model,
                cfg,
                positive_index,
                threshold,
                DATA_ROOT,
            )
        model_path = out_dir / "model.joblib"
        joblib.dump(model, model_path)
    manifest = build_manifest(
        cfg,
        labels,
        (len(train), len(val), len(test)),
        threshold,
        metrics,
        per_class,
        source,
        model_path,
        REPO_ROOT,
        SEED,
        (
            {
                "selected_epoch": result.selected_epoch,
                "validation_macro_f1": result.val_f1,
            }
            if cfg.backend == "transformers"
            else None
        ),
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()

    for detector_id, cfg in _load_configs().items():
        if args.only and args.only != detector_id:
            continue
        manifest = train_one(cfg, args.out / detector_id)
        print(detector_id, manifest.metrics, flush=True)


def _load_configs() -> dict[str, DetectorConfig]:
    raw = tomllib.loads((Path(__file__).parent / "detectors.toml").read_text())
    return {name: DetectorConfig(id=name, **body) for name, body in raw.items()}


def _split_cols(cfg: DetectorConfig, frame: pd.DataFrame) -> list[str]:
    candidates = (cfg.split_field, cfg.group_field)
    return list(dict.fromkeys(field for field in candidates if field and field in frame.columns))


if __name__ == "__main__":
    main()
