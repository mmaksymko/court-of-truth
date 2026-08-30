"""Offline metrics for the tribunal experiment (task 6).

Pure functions over gold labels, predicted labels and predicted class
probabilities. Technical failures are accounted separately (E-018): callers map
a failed run to a label under the intention-to-treat rule before scoring, and use
`OutcomeCounts` to report the raw failure rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Label = Literal["reliable", "questionable", "unreliable"]
LABELS: tuple[Label, ...] = ("reliable", "questionable", "unreliable")


def _check_length(*sequences: Sequence[object]) -> int:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("metric inputs must have equal length")
    return lengths.pop()


def _f1_for(gold: Sequence[Label], predicted: Sequence[Label], label: Label) -> float:
    true_positive = sum(g == label and p == label for g, p in zip(gold, predicted, strict=True))
    predicted_positive = sum(p == label for p in predicted)
    actual_positive = sum(g == label for g in gold)
    if predicted_positive == 0 or actual_positive == 0:
        return 0.0
    precision = true_positive / predicted_positive
    recall = true_positive / actual_positive
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def macro_f1(gold: Sequence[Label], predicted: Sequence[Label]) -> float:
    _check_length(gold, predicted)
    if not gold:
        return 0.0
    return sum(_f1_for(gold, predicted, label) for label in LABELS) / len(LABELS)


def multiclass_brier(gold: Sequence[Label], probabilities: Sequence[dict[Label, float]]) -> float:
    count = _check_length(gold, probabilities)
    if count == 0:
        return 0.0
    total = 0.0
    for true_label, distribution in zip(gold, probabilities, strict=True):
        for label in LABELS:
            target = 1.0 if label == true_label else 0.0
            total += (distribution.get(label, 0.0) - target) ** 2
    return total / count


def _ece(pairs: Sequence[tuple[float, bool]], bins: int) -> float:
    if not pairs:
        return 0.0
    edges = [index / bins for index in range(bins + 1)]
    total = len(pairs)
    error = 0.0
    for lower, upper, is_last in ((edges[i], edges[i + 1], i == bins - 1) for i in range(bins)):
        bucket = [
            (confidence, correct)
            for confidence, correct in pairs
            if lower <= confidence < upper or (is_last and confidence == upper)
        ]
        if not bucket:
            continue
        avg_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        accuracy = sum(correct for _, correct in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(avg_confidence - accuracy)
    return error


@dataclass(frozen=True)
class Calibration:
    top_label_ece: float
    per_class_ece: dict[Label, float]
    macro_ece: float


def calibration(
    gold: Sequence[Label],
    predicted: Sequence[Label],
    probabilities: Sequence[dict[Label, float]],
    *,
    bins: int = 10,
) -> Calibration:
    _check_length(gold, predicted, probabilities)
    top_pairs = [
        (probabilities[i].get(predicted[i], 0.0), predicted[i] == gold[i]) for i in range(len(gold))
    ]
    per_class: dict[Label, float] = {}
    for label in LABELS:
        class_pairs = [
            (probabilities[i].get(label, 0.0), gold[i] == label) for i in range(len(gold))
        ]
        per_class[label] = _ece(class_pairs, bins)
    macro = sum(per_class.values()) / len(LABELS) if per_class else 0.0
    return Calibration(
        top_label_ece=_ece(top_pairs, bins), per_class_ece=per_class, macro_ece=macro
    )


def quantile_ece(
    gold: Sequence[Label],
    predicted: Sequence[Label],
    probabilities: Sequence[dict[Label, float]],
    *,
    bins: int = 5,
) -> float:
    """Top-label ECE with equal-count (quantile) bins instead of equal width.

    At n=56 fixed 10-width bins are uninterpretable: most are empty and a couple
    hold 20-40 points, so the value is dominated by where a handful of points fall
    against the bin edges. Equal-count bins keep each bucket populated. Report this
    with a bootstrap CI, and prefer Brier (bin-free) as the primary calibration
    metric; treat any ECE as illustrative only.
    """
    _check_length(gold, predicted, probabilities)
    pairs = sorted(
        (
            (probabilities[i].get(predicted[i], 0.0), predicted[i] == gold[i])
            for i in range(len(gold))
        ),
        key=lambda pair: pair[0],
    )
    total = len(pairs)
    if total == 0:
        return 0.0
    error = 0.0
    for index in range(bins):
        bucket = pairs[index * total // bins : (index + 1) * total // bins]
        if not bucket:
            continue
        avg_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        accuracy = sum(correct for _, correct in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(avg_confidence - accuracy)
    return error


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float


def bootstrap_ci(
    items: Sequence[object],
    statistic: Callable[[Sequence[object]], float],
    *,
    resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Interval:
    """Percentile bootstrap CI for a paired statistic over items (deterministic).

    ``items`` is any per-case sequence (e.g. ``list(zip(gold, predicted))``) and
    ``statistic`` maps a resampled list to a scalar. Resamples case indices with
    replacement, so it respects the paired structure of the 56 cases. Always report
    corpus metrics as (point, CI) at N=56 - the CI is wide, and that is the honest
    signal.
    """
    count = len(items)
    if count == 0:
        return Interval(0.0, 0.0, 0.0)
    point = statistic(items)
    rng = Random(seed)
    estimates = sorted(
        statistic([items[rng.randrange(count)] for _ in range(count)]) for _ in range(resamples)
    )
    low = estimates[max(0, int((alpha / 2) * resamples))]
    high = estimates[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Interval(point=point, low=low, high=high)


@dataclass(frozen=True)
class OutcomeCounts:
    total: int
    completed: int
    failures: int

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0
