from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from training.config import DetectorConfig


@dataclass(frozen=True)
class TransformerResult:
    threshold: float
    metrics: dict[str, float]
    per_class_recall: dict[str, float]
    selected_epoch: int
    val_f1: float


class EncodedTexts(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        texts: list[str],
        labels: np.ndarray,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encoded["input_ids"][index],
            "attention_mask": self.encoded["attention_mask"][index],
            "labels": self.labels[index],
        }


def train_transformer(
    cfg: DetectorConfig,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    seed: int,
) -> TransformerResult:
    _seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    parts = {"train": train, "val": val, "test": test}
    truth = {
        name: part[cfg.label_field].eq(cfg.positive_label).astype(int).to_numpy()
        for name, part in parts.items()
    }
    datasets = {
        name: EncodedTexts(
            part["_x"].astype(str).tolist(),
            truth[name],
            tokenizer,
            cfg.max_length,
        )
        for name, part in parts.items()
    }
    device = _device()
    negative_label = next(
        label for label in train[cfg.label_field].unique() if label != cfg.positive_label
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label={0: negative_label, 1: cfg.positive_label},
        label2id={negative_label: 0, cfg.positive_label: 1},
    ).to(device)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * steps),
        num_training_steps=steps,
    )

    best: dict[str, Any] | None = None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val_scores = _scores(model, datasets["val"], device, cfg.batch_size)
        threshold, val_f1 = _plateau_threshold(
            truth["val"],
            val_scores,
            cfg.threshold_plateau_tolerance,
        )
        print(
            f"{cfg.id} epoch={epoch} val_f1={val_f1:.6f} logit_threshold={threshold:.6f}",
            flush=True,
        )
        if best is None or val_f1 > best["val_f1"]:
            best = {
                "epoch": epoch,
                "val_f1": val_f1,
                "threshold": threshold,
                "state": copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                ),
            }

    assert best is not None
    model.load_state_dict(best["state"])
    model.to(device)
    test_scores = _scores(model, datasets["test"], device, cfg.batch_size)
    predicted = test_scores >= best["threshold"]
    probabilities = 1.0 / (1.0 + np.exp(-test_scores))
    probability_threshold = 1.0 / (1.0 + math.exp(-best["threshold"]))
    metrics = {
        "f1_macro": round(f1_score(truth["test"], predicted, average="macro"), 4),
        "brier_score": round(brier_score_loss(truth["test"], probabilities), 4),
    }
    per_class = {
        cfg.positive_label: round(recall_score(truth["test"], predicted), 4),
        negative_label: round(
            recall_score(truth["test"], predicted, pos_label=False),
            4,
        ),
    }
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return TransformerResult(
        threshold=probability_threshold,
        metrics=metrics,
        per_class_recall=per_class,
        selected_epoch=int(best["epoch"]),
        val_f1=float(best["val_f1"]),
    )


def _scores(
    model: torch.nn.Module,
    dataset: EncodedTexts,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            scores.extend((logits[:, 1] - logits[:, 0]).detach().cpu().tolist())
    return np.asarray(scores)


def _plateau_threshold(
    truth: np.ndarray,
    scores: np.ndarray,
    tolerance: float,
) -> tuple[float, float]:
    ordered = np.sort(scores)
    candidates = np.r_[
        np.nextafter(ordered[0], -np.inf),
        (ordered[:-1] + ordered[1:]) / 2,
        np.nextafter(ordered[-1], np.inf),
    ]
    metrics = np.asarray(
        [f1_score(truth, scores >= threshold, average="macro") for threshold in candidates]
    )
    eligible = np.flatnonzero(metrics >= metrics.max() - tolerance)
    selected = int(eligible[-1])
    return float(candidates[selected]), float(metrics[selected])


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
