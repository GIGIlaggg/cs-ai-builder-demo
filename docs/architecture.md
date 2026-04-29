# Architecture

## Pipeline

```
inbound email
     │
     ▼
┌──────────────┐
│ classify     │  rule-based today, swap for embedding-classifier in week 2
│ intent       │
└──────┬───────┘
       ▼
┌──────────────┐
│ retrieve KB  │  BM25 today, pgvector + hybrid retrieval in production
└──────┬───────┘
       ▼
┌──────────────┐
│ draft (LLM)  │  Anthropic / OpenAI / stub. Versioned prompt in app/prompts.
└──────┬───────┘
       ▼
┌──────────────┐
│ score draft  │  cheap rule-based + optional LLM-as-judge.
│ confidence + │  hallucination signals: invented timelines, PII echo,
│ hallucinate  │  unsupported numbers.
└──────┬───────┘
       ▼
┌──────────────┐
│ decide       │  auto-send (≥0.85 + no halluc) | review queue | block
└──────────────┘
```

## What each piece is and what it becomes in production

| Layer | This repo | Production swap |
|-------|-----------|-----------------|
| Intent | keyword rules in `app/draft.py` | small fine-tuned classifier (DistilBERT) or LLM with structured output |
| Retrieval | BM25 over 10 docs in `data/kb.jsonl` | pgvector + hybrid (BM25 + dense) over the live policy + KB corpus |
| LLM | provider-agnostic in `app/llm.py` | same interface, same prompt versioning, prod-grade observability around it |
| Eval | rule-based in `app/eval.py` | rule-based + Haiku-tier LLM judge running on every output |
| Decision | threshold in `app/eval.py` | feature-flag driven; per-intent thresholds; per-region carve-outs |
| Audit | trace in API response | append-only log to BigQuery / Snowflake; replayable via prompt version |

## Why these choices

- **No vector DB in week 1.** BM25 on 10 docs gets you 80% of the way for an MVP. You add a vector layer when retrieval starts to leak (you'll see it in the eval set).
- **LLM client is a thin shim.** Every provider has its own SDK, but the surface we care about is `complete(prompt, system) → text`. We hold ourselves to that interface so we can swap providers without touching prompt or eval code.
- **Confidence scoring runs on every output.** We start cheap and rule-based so it adds zero latency. The LLM judge is a hook (`EVAL_USE_LLM=1`) we'll turn on as soon as we're ready to pay for it.
- **The eval set IS the spec.** When PMs and Ops disagree about whether a draft is "good", we add it to the eval set. The eval set is canonical.

## Day 30 production deployment

- Deploy as a containerised FastAPI service behind an internal LB.
- Pull from inbound email queue (SQS / Pub/Sub). Push drafted replies to a `pending_review` table for the Streamlit-replacement review UI.
- Auto-send path is feature-flagged per-intent and per-region, default off.
- Every output logs: prompt template version, model, retrieved chunk IDs, draft, scoring detail, decision, downstream send/edit/reject signal.
- Daily eval run on a refreshed labelled set; PagerDuty alert if hallucination rate >2% over 24h.
