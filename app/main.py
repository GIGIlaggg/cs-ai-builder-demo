"""FastAPI service exposing the draft assistant."""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .draft import DraftRequest, draft_reply
from .llm import LLMClient

app = FastAPI(
    title="cs-ai-builder-demo",
    version="0.1.0",
    description="30-day-to-production email draft assistant for marketplace CS.",
)


class DraftIn(BaseModel):
    inbound: str = Field(..., description="The customer's inbound email body.")
    customer_first_name: str | None = Field(None, description="Known first name, if any.")
    order_id: str | None = Field(None, description="Order ID, if available.")
    order_context: str | None = Field(None, description="Pre-formatted order context block.")


class DraftOut(BaseModel):
    intent: str
    confidence: float
    auto_send: bool
    hallucination_flag: bool
    draft: str
    kb_chunks: list[dict]
    trace: dict


@app.get("/healthz")
def health() -> dict:
    return {"status": "ok", "llm_provider": LLMClient().provider}


@app.post("/draft", response_model=DraftOut)
def draft(payload: DraftIn) -> DraftOut:
    req = DraftRequest(
        inbound=payload.inbound,
        customer_first_name=payload.customer_first_name,
        order_id=payload.order_id,
        order_context=payload.order_context,
    )
    res = draft_reply(req)
    return DraftOut(
        intent=res.intent,
        confidence=res.confidence,
        auto_send=res.auto_send,
        hallucination_flag=res.hallucination_flag,
        draft=res.draft,
        kb_chunks=res.kb_chunks,
        trace=res.trace,
    )
