# cs-ai-builder-demo

> A 30-day-to-production email draft assistant for marketplace customer support, demonstrating the Phase 0 build of the **10 → 120: AI-Native CommOps** business case.

This repo is a **working prototype** — not a slide deck. It runs locally without API keys (deterministic stub mode), runs against a real LLM with one env var, ships behind a feature flag, and has an evaluation harness from day one.

---

## What it does

Given an inbound customer support email + customer/order context, the service returns:

1. A **drafted reply** in the company's voice.
2. A **confidence score** (0–1) on whether to auto-send vs route to human review.
3. **Policy citations** from the knowledge base — every claim in the draft is grounded in a retrieved KB chunk, or the draft is downgraded.
4. A **hallucination flag** if the draft asserts anything not in the retrieved context.
5. The full **decision trace** (retrieved KB chunks, prompt, model output) for audit.

If confidence ≥ 0.85 *and* no hallucination flag, the draft is auto-sent (in production-mode behind a feature flag). Otherwise it goes to a human review queue rendered as a Streamlit UI.

This is the core Phase 0 capability of the CommOps business case. It is the smallest thing you can put in production in 30 days and start measuring savings against.

## Why this scope

- **Email is async** — no real-time SLA risk.
- **Human-in-the-loop by default** — the auto-send path starts at 0% traffic and is only opened up after the eval set passes.
- **Five intents to start** — order_status, refund_simple, voucher_redemption, deal_question_pre_sale, delivery_issue. ~70% of CS email volume is in these five intents.
- **Falsifiable** — the eval set is labelled, so we can prove or disprove every claim about quality.

## Architecture

```
                  ┌─────────────┐
inbound email  →  │  FastAPI    │  →  drafted reply + confidence + citations + trace
                  │  /draft     │
                  └──────┬──────┘
                         │
              ┌──────────┼──────────┬─────────────┐
              ↓          ↓          ↓             ↓
         intent      KB retrieval  draft     hallucination
       classifier   (BM25-ish)    (LLM)        scorer
```

- `app/main.py` — FastAPI service exposing `POST /draft` and `GET /healthz`.
- `app/draft.py` — orchestrator: classify → retrieve → draft → score.
- `app/kb.py` — synthetic knowledge base (10 policy docs) with simple keyword retrieval. Swap for pgvector / Pinecone in production.
- `app/eval.py` — confidence + hallucination scorer. Uses a smaller / cheaper model (or rule-based stub) to grade the larger model's draft.
- `app/llm.py` — provider-agnostic LLM client. Anthropic (default), OpenAI, or stub.
- `app/prompts/` — versioned prompt files.
- `streamlit_demo.py` — human review queue UI.
- `tests/eval_set.jsonl` — 20 labelled test cases. CI fails if quality drops.

## Quickstart

```bash
git clone https://github.com/GIGIlaggg/cs-ai-builder-demo.git
cd cs-ai-builder-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the API locally (uses deterministic stub LLM by default)
make run

# In another terminal — run the human-review Streamlit demo
make demo

# Run the eval set
make eval

# Run tests
make test
```

To use a real LLM, set one of:

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # default if both set
export OPENAI_API_KEY=sk-...
```

Without keys, the stub returns deterministic responses good enough to exercise the full pipeline (intent classification, retrieval, draft, eval). The README's quality numbers are from runs against Anthropic's Claude Sonnet.

## What ships in 30 days (Phase 0)

| Day | Milestone |
|-----|-----------|
| 1–5 | This repo. FastAPI service + 5 intents + KB + eval harness + 20-case test set. |
| 6–10 | Wire to staging email queue. Shadow mode — drafts produced for every inbound, sent only to review queue, never to customers. |
| 11–18 | Human reviewers grade 500 shadow drafts. Eval set expanded to 200 labelled cases. Quality bar: ≥85% acceptable, <2% hallucination. |
| 19–25 | 10% live traffic on the 5 intents, auto-send only at confidence ≥ 0.85. The remaining 90% stays human-only as control. |
| 26–30 | A/B readout. If AHT cut ≥ 40% on the 10% AI-handled cohort and CSAT delta ≥ 0, we have crossed the bar. Ramp toward Phase 1. |

## Eval set

`tests/eval_set.jsonl` — 20 cases with golden labels:

- 12 standard cases (one per intent × 2–3 variations)
- 5 adversarial cases (off-policy refund requests, missing order, ambiguous intent)
- 3 hallucination traps (questions where the KB has no answer — model should defer, not invent)

CI runs the eval set on every PR and fails if hallucination rate > 5% or acceptable rate < 80%.

## What this prototype is not

- A production email pipeline. There's no SMTP/IMAP integration here — the service takes JSON in and returns JSON out. Wiring it to a real email queue is a 1-week job.
- A finished agent. Phase 1 wraps this service in a copilot UI for human agents. Phase 2 adds tool-calling for refund/order modifications. Phase 3 adds multi-step agentic workflows. This repo is the bedrock layer.
- A polished UI. The Streamlit demo is intentionally barebones — it exists to show the human review queue ergonomics, not to be the final product.

## License

MIT. Synthetic data only. No customer data. Safe to fork.

## Companion docs

The full business case this prototype is the Phase 0 of: see `business-case.md` in the parent repository.
