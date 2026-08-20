from __future__ import annotations

from typing import Any, NamedTuple


class TransformerBundle(NamedTuple):
    tokenizer: Any
    model: Any
    torch: Any
    max_length: int
    positive_index: int


def load(model_dir: str, *, max_length: int, positive_index: int) -> TransformerBundle:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model.eval()
    return TransformerBundle(tokenizer, model, torch, max_length, positive_index)


def predict_proba(bundle: TransformerBundle, text: str) -> float:
    encoded = bundle.tokenizer(
        text,
        truncation=True,
        max_length=bundle.max_length,
        return_tensors="pt",
    )
    with bundle.torch.no_grad():
        logits = bundle.model(**encoded).logits
        probability = bundle.torch.softmax(logits, dim=-1)[0, bundle.positive_index]
    return float(probability)
