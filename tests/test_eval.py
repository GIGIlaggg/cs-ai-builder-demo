from app.eval import run_eval_set


def test_eval_set_meets_quality_bars():
    s = run_eval_set()
    assert s["n_cases"] >= 20
    assert s["intent_accuracy"] >= 0.80, s
    assert s["hallucination_rate"] <= 0.05, s
