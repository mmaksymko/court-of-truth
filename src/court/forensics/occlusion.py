"""Model-agnostic evidence by occlusion.

Masks each word in turn and measures how far the positive probability drops; the
words whose removal drops it most are the ones driving the flag. Works with any
detector that exposes a ``predict(text) -> float`` callable, so sklearn and
transformer backends share one faithful evidence method.
"""

import re
from collections.abc import Callable

_EDGE_PUNCT_RE = re.compile(r"^\W+|\W+$")
_MIN_WORDS = 2


def occlusion_evidence(predict: Callable[[str], float], text: str, k: int) -> list[str]:
    words = text.split()
    if len(words) < _MIN_WORDS:
        return []
    baseline = predict(text)

    def drop_without(index: int) -> float:
        without = " ".join(words[:index] + words[index + 1 :])
        return baseline - predict(without)

    scored = sorted(
        ((words[index], drop_without(index)) for index in range(len(words))),
        key=lambda item: item[1],
        reverse=True,
    )
    important: list[str] = []
    seen: set[str] = set()
    for word, drop in scored:
        if drop <= 0:
            break
        trimmed = _EDGE_PUNCT_RE.sub("", word)
        if trimmed and trimmed.lower() not in seen:
            seen.add(trimmed.lower())
            important.append(trimmed)
        if len(important) == k:
            break
    return important
