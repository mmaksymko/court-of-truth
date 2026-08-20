from court.forensics.promo_signals import promo_evidence


def test_promo_evidence_surfaces_named_signals():
    text = (
        "Замовити ексклюзивний товар зі знижкою 50%! Найкращий вибір на ринку, "
        "телефонуйте нам — наша компанія гарантує якість."
    )
    signals = promo_evidence(text)
    assert signals
    joined = " | ".join(signals).lower()
    assert "промо-лексика" in joined
    assert "заклики до дії" in joined
    assert "ціни та знижки" in joined


def test_promo_evidence_empty_for_neutral_text():
    text = "Верховна Рада ухвалила закон у другому читанні під час пленарного засідання."
    assert promo_evidence(text) == []


def test_promo_evidence_respects_k():
    text = (
        "Ексклюзивна знижка! Замовити зараз за посиланням. Найкращий бонус 30%! "
        "Телефонуйте, наша компанія гарантує подарунок www.example.com"
    )
    assert len(promo_evidence(text, k=2)) == 2
