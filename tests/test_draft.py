from app.draft import DraftRequest, classify_intent, draft_reply


def test_classify_refund_simple():
    assert classify_intent("I want a refund, voucher unused, bought yesterday") == "refund_simple"


def test_classify_delivery():
    assert classify_intent("My delivery hasn't arrived for over a week") == "delivery_issue"


def test_classify_unknown():
    assert classify_intent("just saying hi!") == "unknown"


def test_draft_returns_grounded_reply():
    req = DraftRequest(
        inbound="Hi - I bought a voucher 3 days ago and haven't redeemed it. Can I get a refund?",
        customer_first_name="Marta",
    )
    res = draft_reply(req)
    assert res.intent == "refund_simple"
    assert res.draft, "draft should be non-empty"
    assert res.kb_chunks, "should retrieve at least one KB chunk"
    assert 0.0 <= res.confidence <= 1.0


def test_unknown_intent_drops_confidence():
    req = DraftRequest(inbound="hello there friend", customer_first_name="Carla")
    res = draft_reply(req)
    assert res.intent == "unknown"
    assert res.confidence < 0.85
    assert not res.auto_send
