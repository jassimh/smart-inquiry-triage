"""Compose classification, retrieval, decisions, and resolution into a single workflow."""


from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.classify import classify_inquiry
from src.decisions import assess_evidence, route_category
from src.knowledge_base import retrieve_cases
from src.resolution import draft_resolution


class TriageState(TypedDict, total=False):
    # Inputs
    query: str
    top_k: int
    confidence_threshold: float

    # Results added as the workflow progresses
    category: str
    classification_reason: str
    retrieved_past_cases: list[dict]
    decision: dict
    routed_queue: str
    review_status: str
    resolution_notes: str


def classify_node(state: TriageState) -> dict:
    """Return the model's category and explanation as updates to shared state."""
    classification = classify_inquiry(state["query"])

    return {
        "category": classification.category,
        "classification_reason": classification.reason,
    }


def retrieve_node(state: TriageState) -> dict:
    """Return historical matches using the inquiry and requested Top-K."""
    cases = retrieve_cases(
        state["query"],
        top_k=state["top_k"],
    )

    return {"retrieved_past_cases": cases}


def assess_node(state: TriageState) -> dict:
    """Return priority, confidence, and review reasons from retrieved evidence."""
    decision = assess_evidence(
        category=state["category"],
        retrieved_cases=state["retrieved_past_cases"],
        confidence_threshold=state["confidence_threshold"],
    )

    return {"decision": decision}


def route_node(state: TriageState) -> dict:
    """Return the queue configured for the predicted category."""
    queue = route_category(state["category"])
    return {"routed_queue": queue}


def choose_review_path(state: TriageState) -> str:
    """Choose the final branch from the decision's existing escalation flag."""
    if state["decision"]["escalated"]:
        return "review"

    return "finish"


def human_review_node(state: TriageState) -> dict:
    """Mark the result for review without notifying or waiting for a person."""
    # This flags the result; it does not contact a person.
    return {"review_status": "human_review_required"}


def resolution_node(state: TriageState) -> dict:
    """Add suggested notes using the inquiry, decisions, and historical context."""
    notes = draft_resolution(
        query=state["query"],
        category=state["category"],
        decision=state["decision"],
        routed_queue=state["routed_queue"],
        retrieved_cases=state["retrieved_past_cases"],
    )

    return {"resolution_notes": notes}


def finish_node(state: TriageState) -> dict:
    """Mark the result as not flagged; this does not certify answer correctness."""
    return {"review_status": "not_flagged"}


def build_graph():
    """Compile the ordered triage nodes and final review branch without running them."""
    builder = StateGraph(TriageState)

    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("assess", assess_node)
    builder.add_node("route", route_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("resolution", resolution_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_edge("assess", "route")
    builder.add_edge("route", "resolution")

    # Draft notes before selecting the final review status.
    builder.add_conditional_edges(
        "resolution",
        choose_review_path,
        {
            "review": "human_review",
            "finish": "finish",
        },
    )


    builder.add_edge("human_review", END)
    builder.add_edge("finish", END)

    return builder.compile()


def make_initial_state(
    query: str,
    top_k: int = 3,
    confidence_threshold: float = 0.5,
) -> TriageState:
    """Validate workflow inputs and return a fresh state containing only inputs.

    Requires a nonblank string, integer Top-K from 1 to 10, and a threshold
    between 0 and 1. Raises ValueError for the explicit validation failures.
    The original query text is retained; later nodes add their own results.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The inquiry must be a nonempty string.")

    if type(top_k) is not int or not 1 <= top_k <= 10:
        raise ValueError("top_k must be an integer between 1 and 10.")

    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    return {
        "query": query,
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
    }


# Compilation prepares the workflow. It does not run the model.
triage_graph = build_graph()