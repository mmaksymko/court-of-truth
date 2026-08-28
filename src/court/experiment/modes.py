"""Ablation modes for task 6.

F is the full system. B1 drops the adversarial split (one neutral researcher).
B2 drops the forensic detector signals. B3 uses a reduced search context. There is
no re-search in any mode (decision S-008), so B3 differs from F only by the
reduced search context and remains a composite baseline, not a single-factor
ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from court.tribunal.agents import SearchContext


@dataclass(frozen=True)
class ModeConfig:
    name: str
    adversarial: bool
    include_forensics: bool
    search_context: SearchContext


FULL = ModeConfig("F", adversarial=True, include_forensics=True, search_context="medium")
B1 = ModeConfig("B1", adversarial=False, include_forensics=True, search_context="medium")
B2 = ModeConfig("B2", adversarial=True, include_forensics=False, search_context="medium")
B3 = ModeConfig("B3", adversarial=True, include_forensics=True, search_context="low")

MODES: dict[str, ModeConfig] = {mode.name: mode for mode in (FULL, B1, B2, B3)}
