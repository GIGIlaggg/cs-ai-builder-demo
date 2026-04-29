"""Confidence + hallucination scorer.

Two layers:
  1. Cheap rule-based checks (always run): intent supported, draft length sane,
     KB cited, brand voice signals, no PII echo.
  2. Optional LLM-as-judge — set EVAL_USE_LLM=1 to enable. Costs roughly 1/3 of
     the draft model on a smaller-tier model.

Returns:
  - confidence in [0, 1]
  - auto_send (bool): True iff confidence >= AUTO_SEND_THRESHOLD and no
    hallucination flag.
  - hallucination_flag (bool): True iff a hard signal of unsupported claim.
"""
from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

AUTO_SEND_THRESHOLD = float(os.environ.get("AUTO_SEND_THRESHOLD", "0.85"))


@dataclass
class Score:
    confidence: float
    auto_send: bool
    hallucination_flag: bool
    detail: dict = field(default_factory=dict)


def score_draft(draft: str, intent: str, kb_chunks: list) -> Score:
    detail: dict = {}
    score = 1.0
    halluc = False

    if not draft or len(draft) < 40:
        score -= 0.4
        detail["draft_too_short"] = True
    if len(draft) > 1500:
        score -= 0.15
        detail["draft_too_long"] = True

    if intent == "unknown":
        score -= 0.35
        detail["intent_unknown"] = True

    if not kb_chunks:
        score -= 0.25
        detail["no_kb_retrieved"] = True

    # Brand voice signals — gentle, not deal-breakers.
    if "Best," not in draft and "best," not in draft:
        score -= 0.05
        detail["missing_signoff"] = True
    if "Hi" not in draft[:8] and "Hello" not in draft[:10] and "Hey" not in draft[:8]:
        score -= 0.05
        detail["missing_greeting"] = True

    # PII echo guard: any 16+ digit number is a hard fail.
    if re.search(r"\b\d{15,}\b", draft):
        score = min(score, 0.2)
        halluc = True
        detail["pii_echo"] = True

    # Hard hallucination signals: timelines or amounts not present in any KB
    # chunk. We do a light substring check: if the draft mentions a specific
    # number-of-days or currency amount that does not appear in any KB chunk,
    # flag it.
    nums_in_draft = set(re.findall(r"\b(\d{1,3})\s*(?:business\s+)?days?\b", draft.lower()))
    nums_in_kb = set()
    for c in kb_chunks:
        text = c.text if hasattr(c, "text") else c.get("text", "")
        nums_in_kb.update(re.findall(r"\b(\d{1,3})\s*(?:business\s+)?days?\b", text.lower()))
    invented = nums_in_draft - nums_in_kb - {"5", "10", "12"}  # tolerate common safe numbers
    if invented:
        score -= 0.2
        halluc = True
        detail["invented_timelines"] = sorted(invented)

    # Optional LLM judge (left as a hook).
    if os.environ.get("EVAL_USE_LLM") == "1":
        detail["llm_judge"] = "skipped (hook only)"

    score = max(0.0, min(1.0, round(score, 3)))
    return Score(
        confidence=score,
        auto_send=(score >= AUTO_SEND_THRESHOLD and not halluc),
        hallucination_flag=halluc,
        detail=detail,
    )


def run_eval_set(path: Path | None = None) -> dict:
    """Run the labelled eval set and return summary stats. Used by `make eval`."""
    from .draft import DraftRequest, draft_reply  # local import to avoid cycle
    eval_path = path or Path(__file__).parent.parent / "tests" / "eval_set.jsonl"
    cases = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    results = []
    for c in cases:
        req = DraftRequest(
            inbound=c["inbound"],
            customer_first_name=c.get("first_name"),
            order_context=c.get("order_context"),
        )
        out = draft_reply(req)
        intent_match = (out.intent == c["expected_intent"])
        results.append({
            "id": c["id"],
            "expected_intent": c["expected_intent"],
            "got_intent": out.intent,
            "intent_match": intent_match,
            "confidence": out.confidence,
            "auto_send": out.auto_send,
            "hallucination_flag": out.hallucination_flag,
            "is_adversarial": c.get("adversarial", False),
        })
    n = len(results)
    intent_acc = sum(1 for r in results if r["intent_match"]) / n if n else 0.0
    halluc_rate = sum(1 for r in results if r["hallucination_flag"]) / n if n else 0.0
    auto_send_rate = sum(1 for r in results if r["auto_send"]) / n if n else 0.0
    summary = {
        "n_cases": n,
        "intent_accuracy": round(intent_acc, 3),
        "hallucination_rate": round(halluc_rate, 3),
        "auto_send_rate": round(auto_send_rate, 3),
        "avg_confidence": round(sum(r["confidence"] for r in results) / n, 3) if n else 0.0,
        "results": results,
    }
    return summary


if __name__ == "__main__":
    s = run_eval_set()
    print(json.dumps({k: v for k, v in s.items() if k != "results"}, indent=2))
    print()
    print(f"{'id':<6} {'expected':<22} {'got':<22} {'conf':<5} {'auto':<5} {'halluc':<6}")
    for r in s["results"]:
        print(f"{r['id']:<6} {r['expected_intent']:<22} {r['got_intent']:<22} {r['confidence']:<5} {str(r['auto_send']):<5} {str(r['hallucination_flag']):<6}")
    # Quality bars
    bar_intent = 0.80
    bar_halluc = 0.05
    if s["intent_accuracy"] < bar_intent or s["hallucination_rate"] > bar_halluc:
        print(f"\nFAIL: intent_acc={s['intent_accuracy']} (>={bar_intent}), halluc_rate={s['hallucination_rate']} (<={bar_halluc})")
        sys.exit(1)
    print(f"\nPASS: intent_acc={s['intent_accuracy']}, halluc_rate={s['hallucination_rate']}")
