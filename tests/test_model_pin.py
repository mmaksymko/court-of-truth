import re
from pathlib import Path

from tests.fakes import make_settings

MODEL_ID = "gpt-5.6-luna"
_MODEL_LITERAL = re.compile(r"gpt-[0-9][\w.-]*")
_SOURCE_ROOT = Path(__file__).parents[1] / "src" / "court"


def test_default_model_is_luna():
    assert make_settings().model == MODEL_ID


def test_source_pins_exactly_one_openai_model_literal():
    found: dict[str, list[str]] = {}
    for path in _SOURCE_ROOT.rglob("*.py"):
        for literal in _MODEL_LITERAL.findall(path.read_text(encoding="utf-8")):
            found.setdefault(literal, []).append(str(path.relative_to(_SOURCE_ROOT)))
    assert set(found) <= {MODEL_ID}, found
