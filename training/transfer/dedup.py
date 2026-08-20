from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from training.transfer.config import DEFAULT_DEDUP_CONFIG, SPLIT_NAMES, DedupConfig

_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w’']+", flags=re.UNICODE)


def deduplicate_cross_split(
    parts: Mapping[str, pd.DataFrame],
    *,
    text_field: str = "text",
    config: DedupConfig = DEFAULT_DEDUP_CONFIG,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    missing = set(SPLIT_NAMES) - set(parts)
    if missing:
        raise ValueError(f"missing split(s): {sorted(missing)}")
    clean = {name: part.copy() for name, part in parts.items()}
    before = {name: len(part) for name, part in clean.items()}
    matches: list[dict[str, Any]] = []

    for lower_name, higher_names in (
        ("val", ("test",)),
        ("train", ("val", "test")),
    ):
        higher = pd.concat([clean[name] for name in higher_names], axis=0)
        found = find_fuzzy_matches(
            clean[lower_name],
            higher,
            text_field=text_field,
            config=config,
        )
        if found:
            remove_positions = {match["left_position"] for match in found}
            keep = np.asarray(
                [position not in remove_positions for position in range(len(clean[lower_name]))]
            )
            clean[lower_name] = clean[lower_name].iloc[keep].copy()
            for match in found:
                if len(matches) >= config.audit_examples:
                    break
                matches.append(
                    {
                        "removed_from": lower_name,
                        "protected_splits": list(higher_names),
                        **match,
                    }
                )

    after = {name: len(part) for name, part in clean.items()}
    remaining = []
    for lower_name, higher_names in (
        ("val", ("test",)),
        ("train", ("val", "test")),
    ):
        higher = pd.concat([clean[name] for name in higher_names], axis=0)
        remaining.extend(
            find_fuzzy_matches(
                clean[lower_name],
                higher,
                text_field=text_field,
                config=config,
                stop_after=1,
            )
        )
    if remaining:
        raise AssertionError("cross-split duplicate remained after de-duplication")

    audit = {
        "method": (
            "normalized SHA-256, then rare token-shingle fuzzy Jaccard/containment matching"
        ),
        "policy": "preserve test; remove matching val before test and train before val/test",
        "fuzzy_threshold": config.fuzzy_threshold,
        "shingle_size": config.shingle_size,
        "max_shingle_df": config.max_shingle_df,
        "min_shared_shingles": config.min_shared_shingles,
        "min_length_ratio": config.min_length_ratio,
        "before_rows": before,
        "after_rows": after,
        "removed_rows": {name: before[name] - after[name] for name in SPLIT_NAMES},
        "example_matches": matches,
        "example_matches_truncated": sum(before.values()) - sum(after.values()) > len(matches),
        "postcheck_cross_split_matches": 0,
    }
    return clean, audit


def find_fuzzy_matches(  # noqa: PLR0912
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    text_field: str,
    config: DedupConfig = DEFAULT_DEDUP_CONFIG,
    stop_after: int | None = None,
) -> list[dict[str, Any]]:
    if text_field not in left or text_field not in right:
        raise ValueError(f"missing de-duplication text field {text_field!r}")

    right_norm = [_normalize_for_dedup(value) for value in right[text_field]]
    right_hashes: dict[str, int] = {}
    for position, normalized in enumerate(right_norm):
        right_hashes.setdefault(_text_hash(normalized), position)

    right_units = [_shingles(value, config.shingle_size) for value in right_norm]
    document_frequency = Counter(unit for units in right_units for unit in units)
    inverted: dict[str, list[int]] = defaultdict(list)
    for position, units in enumerate(right_units):
        for unit in units:
            if document_frequency[unit] <= config.max_shingle_df:
                inverted[unit].append(position)

    matches: list[dict[str, Any]] = []
    for left_position, raw in enumerate(left[text_field]):
        normalized = _normalize_for_dedup(raw)
        digest = _text_hash(normalized)
        if digest in right_hashes:
            right_position = right_hashes[digest]
            matches.append(
                _match_audit(
                    left_position,
                    right_position,
                    normalized,
                    right_norm[right_position],
                    method="sha256",
                    score=1.0,
                )
            )
        else:
            units = _shingles(normalized, config.shingle_size)
            candidate_overlap: Counter[int] = Counter(
                position
                for unit in units
                if document_frequency[unit] <= config.max_shingle_df
                for position in inverted.get(unit, ())
            )
            best: tuple[float, int] | None = None
            for right_position, rare_intersection in candidate_overlap.items():
                if rare_intersection < config.min_shared_shingles:
                    continue
                intersection = len(units & right_units[right_position])
                score = _fuzzy_score_from_intersection(
                    intersection,
                    len(units),
                    len(right_units[right_position]),
                    len(normalized),
                    len(right_norm[right_position]),
                    config,
                )
                if score >= config.fuzzy_threshold and (best is None or score > best[0]):
                    best = (score, right_position)
            if best is not None:
                score, right_position = best
                matches.append(
                    _match_audit(
                        left_position,
                        right_position,
                        normalized,
                        right_norm[right_position],
                        method="fuzzy_shingles",
                        score=score,
                    )
                )
        if stop_after is not None and len(matches) >= stop_after:
            break
    return matches


def decontaminate_against_external(
    parts: Mapping[str, pd.DataFrame],
    external_text: pd.DataFrame,
    *,
    text_field: str = "text",
    config: DedupConfig = DEFAULT_DEDUP_CONFIG,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if list(external_text.columns) != [text_field]:
        raise ValueError("external decontamination accepts the text column only")
    clean = {name: part.copy() for name, part in parts.items()}
    all_matches: list[dict[str, Any]] = []
    before = {name: len(part) for name, part in clean.items()}
    for name in SPLIT_NAMES:
        found = find_fuzzy_matches(
            clean[name],
            external_text,
            text_field=text_field,
            config=config,
        )
        remove_positions = {match["left_position"] for match in found}
        if remove_positions:
            keep = np.asarray(
                [position not in remove_positions for position in range(len(clean[name]))]
            )
            clean[name] = clean[name].iloc[keep].copy()
        for match in found:
            all_matches.append(
                {
                    "removed_internal_split": name,
                    "internal_position": match["left_position"],
                    "gold_position": match["right_position"],
                    **{
                        key: value
                        for key, value in match.items()
                        if key not in {"left_position", "right_position"}
                    },
                }
            )
    after = {name: len(part) for name, part in clean.items()}
    return clean, {
        "gold_used_for_dedup_only": True,
        "gold_columns_read_before_fit": [text_field],
        "protected_split": "jeansa_gold (rows never removed)",
        "before_rows": before,
        "after_rows": after,
        "removed_rows": {name: before[name] - after[name] for name in SPLIT_NAMES},
        "matches": all_matches[: config.audit_examples],
        "matches_truncated": len(all_matches) > config.audit_examples,
        "match_count": len(all_matches),
    }


def _normalize_for_dedup(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = normalized.replace("’", "'").replace("`", "'")
    return _SPACE_RE.sub(" ", normalized).strip()


def _text_hash(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _shingles(normalized: str, size: int) -> frozenset[str]:
    tokens = _TOKEN_RE.findall(normalized)
    if len(tokens) >= size:
        return frozenset(
            "\x1f".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
        )
    compact = normalized.replace(" ", "")
    if len(compact) <= 5:
        return frozenset({compact}) if compact else frozenset()
    return frozenset(compact[index : index + 5] for index in range(len(compact) - 4))


def _fuzzy_score_from_intersection(
    intersection: int,
    left_units: int,
    right_units: int,
    left_length: int,
    right_length: int,
    config: DedupConfig,
) -> float:
    if not left_units or not right_units:
        return 0.0
    union = left_units + right_units - intersection
    jaccard = intersection / union
    containment = intersection / min(left_units, right_units)
    length_ratio = min(left_length, right_length) / max(left_length, right_length)
    if length_ratio < config.min_length_ratio:
        return jaccard
    return max(jaccard, containment)


def _match_audit(
    left_position: int,
    right_position: int,
    left: str,
    right: str,
    *,
    method: str,
    score: float,
) -> dict[str, Any]:
    return {
        "left_position": left_position,
        "right_position": right_position,
        "method": method,
        "score": round(score, 6),
        "left_sha256": _text_hash(left),
        "right_sha256": _text_hash(right),
        "left_preview": left[:120],
        "right_preview": right[:120],
    }
