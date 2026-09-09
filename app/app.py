"""
Smart Inquiry Triage Assistant — Streamlit chat interface.

Collects inquiries and settings, displays results, and preserves session history.
"""

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.main import triage_inquiry as run_triage

# --- Configuration (sidebar controls) ------------------------------------

st.set_page_config(page_title="Smart Inquiry Triage", page_icon="📨")
st.title("📨 Smart Inquiry Triage Assistant")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K past cases", min_value=1, max_value=10, value=3)
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )


# --- Backend-----------------------------------------

def triage_inquiry(query: str, top_k: int, confidence_threshold: float) -> dict:
   
    return run_triage(
        query=query,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )


# --- Result rendering -----------------------------------------------------

def render_result(result: dict) -> None:
    st.write("**Query:**", result["query"])
    st.write("**Category:**", result["category"])
    st.write("**Priority:**", result["priority"])
    st.write("**Routed queue:**", result["routed_queue"])
    st.write("**Confidence score:**", round(result["confidence"], 3))
    st.caption("Experimental evidence score; not a probability of correctness.")

    st.markdown("**Suggested resolution notes:**")
    st.text(result["resolution_notes"])

    if result["escalated"]:
        st.warning("Human review required. Suggested actions need verification.")
        for reason in result["review_reasons"]:
            st.write(reason)

    with st.expander("Retrieved historical cases"):
        for case in result["retrieved_past_cases"]:
            st.write(
                f"{case['case_id']} | "
                f"{case['category']} | "
                f"{case['priority']} | "
                f"similarity {case['cosine_similarity']:.3f}"
            )
            st.text(case["inquiry_text"])

    with st.expander("Decision details"):
        st.json({
            "classification_reason": result["classification_reason"],
            "priority_votes": result["priority_votes"],
            "confidence_components": result["confidence_components"],
            "review_status": result["review_status"],
        })


# --- Chat / session view --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"query": str, "result": dict|None}

# Replay history for this session.
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        if turn["result"] is not None:
            render_result(turn["result"])
        else:
            st.error(turn.get("error", "No result."))

# New inquiry input.
query = st.chat_input("Paste a customer inquiry...")
if query:
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Triaging..."):
                result = triage_inquiry(query, top_k, confidence_threshold)
            render_result(result)
            st.session_state.history.append({"query": query, "result": result})
        except Exception as error:
            msg = f"Triage could not complete: {error}"
            st.error(msg)
            st.session_state.history.append(
                {"query": query, "result": None, "error": msg}
            )
