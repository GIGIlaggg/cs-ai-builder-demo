"""Draft orchestrator: classify intent → retrieve KB → draft → score."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path

from .eval import score_draft
from .kb import KB, KBChunk
from .llm import LLMClient

SUPPORTED_INTENTS = (
    "order_status",
    "refund_simple",
    "voucher_redemption",
    "deal_question_pre_sale",
    "delivery_issue",
    "cancel_modify_booking",
    "refund_complex",
)

INTENT_KEYWORDS: dict[str, tuple[tuple[str, int], ...]] = {
    "refund_complex":         (("dispute", 3), ("months ago", 3), ("six months", 3), ("no-show", 3), ("merchant cancelled", 3)),
    "refund_simple":          (("refund", 3), ("money back", 3), ("haven't redeemed", 2), ("haven't used", 2), ("voucher unused", 2)),
    "delivery_issue":         (("not arrived", 3), ("hasn't arrived", 3), ("lost in transit", 3), ("missing package", 3), ("delivery", 2), ("tracking", 2), ("delayed", 2)),
    "order_status":           (("order status", 3), ("where is my order", 3), ("has it shipped", 3), ("tracking for my", 2), ("shipped", 1)),
    "cancel_modify_booking":  (("cancel booking", 3), ("cancel my booking", 3), ("reschedule", 3), ("change date", 3), ("modify", 2)),
    "deal_question_pre_sale": (("before i buy", 3), ("does this deal", 3), ("is the deal", 3), ("what is included", 3), ("deal include", 2), ("location", 1)),
    "voucher_redemption":     (("redeem", 3), ("expired", 2), ("voucher code", 2), ("how do i use", 2), ("combine my voucher", 3)),
}

PROMPT_TEMPLATE = (Path(__file__).parent / "prompts" / "draft_v1.txt").read_text()


@dataclass
class DraftRequest:
    inbound: str
    customer_first_name: str | None = None
    order_id: str | None = None
    order_context: str | None = None


@dataclass
class DraftResult:
    intent: str
    confidence: float
    auto_send: bool
    hallucination_flag: bool
    draft: str
    kb_chunks: list[dict] = field(default_factory=list)
    trace: dict = field(default_factory=dict)


def classify_intent(inbound: str) -> str:
    """Weighted keyword classifier. Order of dict matters as a soft tiebreaker:
    refund_complex precedes refund_simple precedes voucher_redemption so a
    "voucher unused — refund please" maps to refund_simple, not voucher_redemption.
    """
    text = inbound.lower()
    best, best_score = "unknown", 0
    for intent, weighted in INTENT_KEYWORDS.items():
        score = sum(w for kw, w in weighted if kw in text)
        if score > best_score:
            best, best_score = intent, score
    if best_score < 2:
        return "unknown"
    return best


_kb: KB | None = None


def get_kb() -> KB:
    global _kb
    if _kb is None:
        _kb = KB.load()
    return _kb


def _format_kb(chunks: list[KBChunk]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[chunk:{c.id} | {c.title}]")
        lines.append(c.text)
        lines.append("")
    return "\n".join(lines).strip()


def _redact_pii(text: str) -> str:
    # Strip 12-19 digit runs (likely card numbers) leaving last 4.
    return re.sub(r"\b(\d{8,15})(\d{4})\b", r"****\2", text)


def draft_reply(req: DraftRequest, llm: LLMClient | None = None) -> DraftResult:
    llm = llm or LLMClient()
    intent = classify_intent(req.inbound)
    kb = get_kb()
    chunks = kb.search(req.inbound, intent=intent if intent in SUPPORTED_INTENTS else None, k=4)
    kb_block = _format_kb(chunks)

    prompt = PROMPT_TEMPLATE.format(
        intent=intent,
        first_name=req.customer_first_name or "Unknown",
        order_context=req.order_context or "(none provided)",
        inbound=req.inbound.strip(),
        kb_block=kb_block,
    )
    response = llm.complete(prompt, max_tokens=500)
    draft = _redact_pii(response.text.strip())

    score = score_draft(draft=draft, intent=intent, kb_chunks=chunks)
    return DraftResult(
        intent=intent,
        confidence=score.confidence,
        auto_send=score.auto_send,
        hallucination_flag=score.hallucination_flag,
        draft=draft,
        kb_chunks=[c.as_dict() for c in chunks],
        trace={
            "provider": response.provider,
            "model": response.model,
            "scoring": score.detail,
        },
    )
