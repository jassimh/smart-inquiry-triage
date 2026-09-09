import argparse
import json

from src.workflow import make_initial_state, triage_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    query = input("Customer inquiry: ").strip()

    try:
        initial_state = make_initial_state(
            query=query,
            top_k=args.top_k,
            confidence_threshold=args.threshold,
        )
    except ValueError as error:
        parser.error(str(error))

    # Keep a display copy of the state as updates arrive.
    final_state = dict(initial_state)

    print("\nRunning LangGraph workflow...", flush=True)

    for event in triage_graph.stream(
        initial_state,
        stream_mode="updates",
    ):
        for node_name, updates in event.items():
            print(f"Completed node: {node_name}", flush=True)
            final_state.update(updates)

    print("\nFinal workflow state:")
    print(json.dumps(final_state, indent=2, ensure_ascii=False))


def triage_inquiry(
    query: str,
    top_k: int = 3,
    confidence_threshold: float = 0.5,
) -> dict:
    initial_state = make_initial_state(
        query=query,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )

    final_state = triage_graph.invoke(initial_state)

    return {
        "query": final_state["query"],
        "category": final_state["category"],
        "classification_reason": final_state["classification_reason"],
        "routed_queue": final_state["routed_queue"],
        "resolution_notes": final_state["resolution_notes"],
        "retrieved_past_cases": final_state["retrieved_past_cases"],
        "review_status": final_state["review_status"],
        **final_state["decision"],
    }

if __name__ == "__main__":
    main()