from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.classify import classify_inquiry
from src.decisions import assess_evidence, route_category
from src.knowledge_base import retrieve_cases


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


def classify_node(state: TriageState) -> dict:
    classification = classify_inquiry(state["query"])

    return {
        "category": classification.category,
        "classification_reason": classification.reason,
    }


def retrieve_node(state: TriageState) -> dict:
    cases = retrieve_cases(
        state["query"],
        top_k=state["top_k"],
    )

    return {"retrieved_past_cases": cases}


def assess_node(state: TriageState) -> dict:
    decision = assess_evidence(
        category=state["category"],
        retrieved_cases=state["retrieved_past_cases"],
        confidence_threshold=state["confidence_threshold"],
    )

    return {"decision": decision}


def route_node(state: TriageState) -> dict:
    queue = route_category(state["category"])
    return {"routed_queue": queue}


def choose_review_path(state: TriageState) -> str:
    if state["decision"]["escalated"]:
        return "review"

    return "finish"


def human_review_node(state: TriageState) -> dict:
    # This flags the result; it does not contact a person.
    return {"review_status": "human_review_required"}


def finish_node(state: TriageState) -> dict:
    return {"review_status": "not_flagged"}


def build_graph():
    builder = StateGraph(TriageState)

    builder.add_node("classify", classify_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("assess", assess_node)
    builder.add_node("route", route_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finish", finish_node)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "retrieve")
    builder.add_edge("retrieve", "assess")
    builder.add_edge("assess", "route")

    builder.add_conditional_edges(
        "route",
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