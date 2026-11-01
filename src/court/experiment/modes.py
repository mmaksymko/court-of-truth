"""Ablation modes for task 6.

Each baseline removes exactly one pillar of the full system F:
- F  : two adversarial sides, forensic detectors, web search enabled.
- B1 : drops the adversarial split (one neutral researcher instead of two sides).
- B2 : drops the forensic detector signals.
- B3 : drops external web search entirely (`search_context=None`); the sides argue
       from the article and detector signals only and return no sources.
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
    search_context: SearchContext | None  # None = no web search


FULL = ModeConfig("F", adversarial=True, include_forensics=True, search_context="medium")
B1 = ModeConfig("B1", adversarial=False, include_forensics=True, search_context="medium")
B2 = ModeConfig("B2", adversarial=True, include_forensics=False, search_context="medium")
B3 = ModeConfig("B3", adversarial=True, include_forensics=True, search_context=None)

MODES: dict[str, ModeConfig] = {mode.name: mode for mode in (FULL, B1, B2, B3)}
