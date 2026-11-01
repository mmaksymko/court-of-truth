from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from court.forensics.text_transforms import transform_for
from training.transfer.config import SEED, SPLIT_FRACTIONS, SPLIT_NAMES


def prepare_frame(frame: pd.DataFrame, *, detector_id: str) -> pd.DataFrame:
    required = {"text", "label"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"{detector_id}: missing columns {sorted(missing)}")
    prepared = frame.dropna(subset=["text", "label"]).copy()
    prepared["text"] = prepared["text"].astype(str)
    prepared["_x"] = prepared["text"].map(lambda value: transform_for(detector_id, value))
    return prepared.loc[prepared["_x"].str.len() > 0].copy()


def seeded_outlet_split(
    frame: pd.DataFrame,
    *,
    group_field: str = "outlet",
    seed: int = SEED,
    fractions: Sequence[float] = SPLIT_FRACTIONS,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if group_field not in frame:
        raise ValueError(f"missing clickbait group field {group_field!r}")
    if len(fractions) != len(SPLIT_NAMES) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must contain three values summing to one")

    counts = frame[group_field].astype(str).value_counts().sort_index()
    outlets = counts.index.to_numpy(copy=True)
    np.random.default_rng(seed).shuffle(outlets)

    targets = np.asarray(fractions, dtype=float) * len(frame)
    assigned_rows = np.zeros(len(SPLIT_NAMES), dtype=int)
    assignment: dict[str, str] = {}
    destination = 0
    for outlet in outlets:
        group_rows = int(counts.loc[outlet])
        if destination < len(SPLIT_NAMES) - 1 and assigned_rows[destination] > 0:
            current_error = abs(targets[destination] - assigned_rows[destination])
            added_error = abs(targets[destination] - assigned_rows[destination] - group_rows)
            if added_error > current_error:
                destination += 1
        assignment[str(outlet)] = SPLIT_NAMES[destination]
        assigned_rows[destination] += group_rows

    split_tag = frame[group_field].astype(str).map(assignment)
    parts = {name: frame.loc[split_tag == name].copy() for name in SPLIT_NAMES}
    validate_binary_parts(parts, "label")
    assert_group_isolation(parts, group_field)
    audit = {
        "algorithm": "default_rng(seed) shuffle + sequential nearest-row-budget allocation",
        "seed": seed,
        "target_fractions": dict(zip(SPLIT_NAMES, fractions, strict=True)),
        "actual_rows": {name: len(part) for name, part in parts.items()},
        "actual_fractions": {
            name: round(len(part) / len(frame), 8) for name, part in parts.items()
        },
        "outlets": {
            name: sorted(part[group_field].astype(str).unique().tolist())
            for name, part in parts.items()
        },
        "shuffled_outlets": outlets.tolist(),
    }
    return parts, audit


def predefined_split(
    frame: pd.DataFrame,
    *,
    split_field: str = "split",
    label_field: str = "label",
) -> dict[str, pd.DataFrame]:
    if split_field not in frame:
        raise ValueError(f"missing predefined split field {split_field!r}")
    actual = set(frame[split_field].astype(str).unique())
    if actual != set(SPLIT_NAMES):
        raise ValueError(f"expected train/val/test, got {sorted(actual)}")
    parts = {name: frame.loc[frame[split_field].astype(str) == name].copy() for name in SPLIT_NAMES}
    validate_binary_parts(parts, label_field)
    return parts


def validate_binary_parts(parts: Mapping[str, pd.DataFrame], label_field: str) -> None:
    for name in SPLIT_NAMES:
        labels = set(parts[name][label_field].astype(str).unique())
        if len(labels) != 2:
            raise ValueError(f"{name} split must contain both labels, got {sorted(labels)}")


def assert_group_isolation(parts: Mapping[str, pd.DataFrame], group_field: str) -> None:
    groups = {name: set(part[group_field].astype(str).unique()) for name, part in parts.items()}
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap = groups[left] & groups[right]
            if overlap:
                raise AssertionError(
                    f"{group_field} leakage between {left}/{right}: {sorted(overlap)}"
                )
