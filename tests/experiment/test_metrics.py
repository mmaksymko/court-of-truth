import pytest

from court.experiment import metrics
from court.experiment.metrics import OutcomeCounts

_ONEHOT = {"reliable": 1.0, "questionable": 0.0, "unreliable": 0.0}
_WRONGHOT = {"reliable": 0.0, "questionable": 0.0, "unreliable": 1.0}


def test_macro_f1_perfect_and_zero():
    labels = ["reliable", "questionable", "unreliable"]
    assert metrics.macro_f1(labels, labels) == 1.0
    assert metrics.macro_f1(["reliable", "reliable"], ["questionable", "questionable"]) == 0.0
    assert metrics.macro_f1([], []) == 0.0


def test_macro_f1_known_mixed_value():
    gold = ["reliable", "reliable", "questionable", "unreliable"]
    predicted = ["reliable", "questionable", "questionable", "unreliable"]
    # Per-class F1: reliable 2/3, questionable 2/3, unreliable 1 -> macro 7/9.
    assert metrics.macro_f1(gold, predicted) == pytest.approx(7 / 9, abs=1e-6)


def test_multiclass_brier_bounds():
    assert metrics.multiclass_brier(["reliable"], [_ONEHOT]) == 0.0
    assert metrics.multiclass_brier(["reliable"], [_WRONGHOT]) == 2.0


def test_multiclass_brier_known_value():
    distribution = {"reliable": 0.7, "questionable": 0.2, "unreliable": 0.1}
    # (0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2 = 0.14
    assert metrics.multiclass_brier(["reliable"], [distribution]) == pytest.approx(0.14, abs=1e-9)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="equal length"):
        metrics.macro_f1(["reliable"], ["reliable", "questionable"])


def test_outcome_counts_failure_rate():
    assert OutcomeCounts(total=4, completed=3, failures=1).failure_rate == 0.25
    assert OutcomeCounts(total=0, completed=0, failures=0).failure_rate == 0.0


def test_bootstrap_ci_is_deterministic_and_brackets_point():
    items = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    def mean(sample):
        return sum(sample) / len(sample)

    first = metrics.bootstrap_ci(items, mean, resamples=500, seed=42)
    second = metrics.bootstrap_ci(items, mean, resamples=500, seed=42)
    assert (first.point, first.low, first.high) == (second.point, second.low, second.high)
    assert first.point == pytest.approx(0.625)
    assert first.low <= first.point <= first.high


def test_bootstrap_ci_empty():
    result = metrics.bootstrap_ci([], lambda sample: 0.0)
    assert (result.point, result.low, result.high) == (0.0, 0.0, 0.0)
