import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from training.transfer.config import DedupConfig, ThresholdPolicy
from training.transfer.dedup import (
    decontaminate_against_external,
    deduplicate_cross_split,
)
from training.transfer.experiments import append_jsonl, evaluate_jeansa
from training.transfer.models import positive_scores
from training.transfer.splits import seeded_outlet_split
from training.transfer.thresholds import choose_threshold


def _outlet_frame():
    sizes = {
        "ukr.media": 1000,
        "pravda.com.ua": 900,
        "znaj.ua": 800,
        "politeka.net": 800,
        "hromadske.ua": 310,
        "espreso.tv": 160,
        "zn.ua": 40,
    }
    rows = []
    for outlet, size in sizes.items():
        rows.extend(
            {
                "text": f"{outlet} unique title {index}",
                "label": "clickbait" if index % 2 else "neutral",
                "outlet": outlet,
            }
            for index in range(size)
        )
    return pd.DataFrame(rows)


def test_seeded_outlet_split_matches_canonical_row_budget_and_is_isolated():
    parts, audit = seeded_outlet_split(_outlet_frame(), seed=42)

    assert audit["actual_rows"] == {"train": 2500, "val": 1000, "test": 510}
    assert audit["outlets"] == {
        "train": ["politeka.net", "pravda.com.ua", "znaj.ua"],
        "val": ["ukr.media"],
        "test": ["espreso.tv", "hromadske.ua", "zn.ua"],
    }
    group_sets = {name: set(part["outlet"]) for name, part in parts.items()}
    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])


def test_cross_split_dedup_preserves_test_and_removes_earlier_copies():
    exact = "Це той самий нормалізований текст із кількома словами"
    fuzzy_test = "великий бренд відкрив новий сучасний центр допомоги у києві сьогодні"
    fuzzy_train = "великий бренд відкрив новий сучасний центр допомоги у києві вчора"
    parts = {
        "train": pd.DataFrame(
            {
                "text": [exact.upper(), fuzzy_train, "унікальний train текст"],
                "label": ["clickbait", "neutral", "clickbait"],
            }
        ),
        "val": pd.DataFrame(
            {
                "text": [exact, "унікальний validation текст"],
                "label": ["clickbait", "neutral"],
            }
        ),
        "test": pd.DataFrame(
            {
                "text": [f"  {exact}  ", fuzzy_test, "унікальний test текст"],
                "label": ["clickbait", "neutral", "clickbait"],
            }
        ),
    }

    clean, audit = deduplicate_cross_split(
        parts,
        config=DedupConfig(fuzzy_threshold=0.80),
    )

    assert len(clean["test"]) == 3
    assert len(clean["val"]) == 1
    assert len(clean["train"]) == 1
    assert audit["removed_rows"] == {"train": 2, "val": 1, "test": 0}
    assert {match["method"] for match in audit["example_matches"]} == {
        "sha256",
        "fuzzy_shingles",
    }
    assert audit["postcheck_cross_split_matches"] == 0


def test_threshold_selection_uses_only_the_arrays_it_is_given():
    validation_scores = np.asarray([0.1, 0.2, 0.8, 0.9])
    validation_truth = np.asarray([False, False, True, True])
    policy = ThresholdPolicy(candidates="exact")

    first = choose_threshold(validation_scores, validation_truth, policy)
    # A hypothetical test set is intentionally irrelevant to selection.
    hypothetical_test_truth = np.asarray([True, False, True])
    hypothetical_test_scores = np.asarray([0.0, 1.0, 0.2])
    assert hypothetical_test_truth.shape == hypothetical_test_scores.shape
    second = choose_threshold(validation_scores, validation_truth, policy)

    assert first == second
    assert 0.2 < first[0] < 0.8
    assert first[1]["macro_f1"] == 1.0


def test_external_decontamination_accepts_only_text_and_preserves_external_rows():
    parts = {
        name: pd.DataFrame(
            {
                "text": [f"{name} унікальний текст", "спільний довгий текст для дедупу"],
                "label": ["editorial", "sponsored"],
            }
        )
        for name in ("train", "val", "test")
    }
    external = pd.DataFrame({"text": ["спільний довгий текст для дедупу"]})

    clean, audit = decontaminate_against_external(parts, external)

    assert all(len(part) == 1 for part in clean.values())
    assert len(external) == 1
    assert audit["gold_used_for_dedup_only"] is True
    assert audit["gold_columns_read_before_fit"] == ["text"]
    assert audit["removed_rows"] == {"train": 1, "val": 1, "test": 1}


def test_jeansa_evaluation_uses_merged_internal_splits_without_gold_file():
    frame = pd.DataFrame(
        [
            {
                "text": f"{split} {label} unique article {index}",
                "label": label,
                "split": split,
            }
            for split in ("train", "val", "test")
            for index, label in enumerate(("editorial", "sponsored"))
        ]
    )

    result = evaluate_jeansa(
        frame,
        estimator_factory=lambda _: DummyClassifier(strategy="prior"),
        model_config={"model": "dummy"},
        threshold_policy=ThresholdPolicy(candidates="exact"),
    )

    assert result["protocol"] == "predefined-internal"
    assert result["split_rows"] == {"train": 2, "val": 2, "test": 2}
    assert "gold_recall" not in result


def test_positive_scores_orients_binary_decision_function_to_named_positive():
    class FakeEstimator:
        classes_ = np.asarray(["clickbait", "neutral"])

        @staticmethod
        def decision_function(text):
            return np.asarray([1.0, -2.0])[: len(text)]

    scores = positive_scores(
        FakeEstimator(),
        pd.Series(["first", "second"]),
        "clickbait",
    )

    assert scores.tolist() == [-1.0, 2.0]


def test_jsonl_append_is_complete_under_concurrent_writers(tmp_path):
    path = tmp_path / "experiments" / "transfer_log.jsonl"

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: append_jsonl(path, {"run": index}), range(50)))

    lines = path.read_text().splitlines()
    assert len(lines) == 50
    assert {json.loads(line)["run"] for line in lines} == set(range(50))
