from court.forensics.occlusion import occlusion_evidence


def _only_when(marker: str):
    return lambda text: 0.9 if marker in text else 0.2


def test_occlusion_returns_word_driving_positive():
    assert occlusion_evidence(_only_when("shocking"), "this shocking headline", k=3) == ["shocking"]


def test_occlusion_strips_edge_punctuation():
    assert occlusion_evidence(_only_when("shocking"), "this shocking! headline", k=3) == ["shocking"]


def test_occlusion_short_text_returns_empty():
    assert occlusion_evidence(lambda _text: 0.9, "word", k=3) == []


def test_occlusion_ranks_by_drop_and_caps_k():
    def predict(text: str) -> float:
        probability = 0.1
        if "aaa" in text:
            probability += 0.5
        if "bbb" in text:
            probability += 0.3
        return probability

    assert occlusion_evidence(predict, "aaa bbb ccc", k=1) == ["aaa"]
