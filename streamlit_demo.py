"""Streamlit human review queue UI.

Run:  streamlit run streamlit_demo.py
"""
from __future__ import annotations
import json
from pathlib import Path

import streamlit as st

from app.draft import DraftRequest, draft_reply
from app.llm import LLMClient

EXAMPLES = [
    {
        "label": "Refund — within 14 days, unredeemed",
        "first_name": "Marta",
        "inbound": "Hi — I bought a spa voucher last weekend but my plans changed and I won't use it. Can I get a refund please? I haven't redeemed it.",
    },
    {
        "label": "Lost delivery, 7 days late",
        "first_name": "Pablo",
        "inbound": "My order #ORD-44219 was supposed to arrive on the 15th and it's now the 23rd. I checked the carrier site and there's been no update for 7 days. What can you do?",
    },
    {
        "label": "Pre-sale: is the deal in central Madrid?",
        "first_name": "Lukas",
        "inbound": "Before I buy — is the brunch deal in central Madrid? The page wasn't super clear and I want to make sure I can get there easily.",
    },
    {
        "label": "Voucher expired, asks if it's gone",
        "first_name": "Sofia",
        "inbound": "My voucher said it expired last month. Did I lose all my money? I never got around to using it because of work travel.",
    },
    {
        "label": "Off-policy — wants cash refund 6 months later",
        "first_name": "Diego",
        "inbound": "I bought a deal six months ago and never used it. The merchant just told me they don't have appointments. I want a cash refund, not credit.",
    },
]


def main() -> None:
    st.set_page_config(page_title="CS AI Builder — Review Queue", layout="wide")
    st.title("Customer Support — AI Draft Review Queue")
    st.caption(f"LLM provider: **{LLMClient().provider}**  ·  set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to enable real LLM.")

    col_in, col_out = st.columns([5, 7], gap="large")

    with col_in:
        st.subheader("Inbound")
        choice = st.selectbox("Pick an example, or write your own below:", ["(custom)"] + [e["label"] for e in EXAMPLES])
        default_first = ""
        default_inbound = ""
        if choice != "(custom)":
            ex = next(e for e in EXAMPLES if e["label"] == choice)
            default_first = ex["first_name"]
            default_inbound = ex["inbound"]
        first_name = st.text_input("Customer first name", value=default_first)
        inbound = st.text_area("Inbound email", value=default_inbound, height=240, placeholder="Paste a customer email…")
        order_context = st.text_area(
            "Order context (optional)",
            value="",
            height=80,
            placeholder="Order ID: ORD-12345\nStatus: CONFIRMED\nValue: 49.00 EUR",
        )
        go = st.button("Draft reply", type="primary", disabled=not inbound.strip())

    with col_out:
        st.subheader("Draft + decision")
        if go and inbound.strip():
            with st.spinner("Drafting…"):
                res = draft_reply(DraftRequest(
                    inbound=inbound,
                    customer_first_name=first_name or None,
                    order_context=order_context or None,
                ))
            top1, top2, top3 = st.columns(3)
            top1.metric("Intent", res.intent)
            top2.metric("Confidence", f"{res.confidence:.2f}")
            decision = "AUTO-SEND" if res.auto_send else ("HUMAN REVIEW" if not res.hallucination_flag else "BLOCK")
            top3.metric("Decision", decision)
            if res.hallucination_flag:
                st.error("Hallucination flag raised — do not send. See trace.")
            elif res.auto_send:
                st.success("Confidence above threshold. In production this would auto-send.")
            else:
                st.info("Below auto-send threshold. Routes to human review.")

            st.markdown("**Drafted reply**")
            st.code(res.draft, language="markdown")

            st.markdown("**Retrieved KB chunks**")
            for c in res.kb_chunks:
                with st.expander(f"{c['id']} — {c['title']}"):
                    st.write(c["text"])

            st.markdown("**Decision trace**")
            st.json(res.trace, expanded=False)
        else:
            st.info("Pick an example or paste an email, then click Draft reply.")


if __name__ == "__main__":
    main()
