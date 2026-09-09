"""Derive priority, queue, and an uncalibrated review score from historical evidence."""

import argparse
import json
from math import isfinite

from src.classify import classify_inquiry
from src.knowledge_base import load_cases, retrieve_cases


PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def route_category(category: str) -> str:
    """Return the CSV queue mapped to a category without calling a model.

    Raises ValueError if the category is unmapped or a category has conflicting
    queues. Selecting a queue does not send a ticket to an external system.
    """
    queue_map = {}

    for case in load_cases():
        name = case["category"]
        queue = case["routed_queue"]

        if name in queue_map and queue_map[name] != queue:
            raise ValueError(f"Conflicting queues for category: {name}")

        queue_map[name] = queue

    if category not in queue_map:
        raise ValueError(f"No queue configured for category: {category}")

    return queue_map[category]


def assess_evidence(
    category: str,
    retrieved_cases: list[dict],
    confidence_threshold: float = 0.5,
) -> dict:
    """Return priority, voting diagnostics, confidence, and review reasons.

    Positive similarities clipped to [0, 1] vote within the predicted
    category; exact vote ties favor higher priority. Confidence multiplies
    mean similarity, category agreement, and winning-priority weight share.
    Review is required below the threshold, or whenever supporting evidence
    is absent; absent evidence also yields a medium priority placeholder.

    The provisional flag identifies that fallback, not every uncertain case.
    This function reads no original inquiry text and cannot independently
    interpret urgency or negation. Confidence is not a calibrated probability.
    Invalid thresholds, non-finite scores, and invalid voting priorities
    raise ValueError.
    """
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("Confidence threshold must be between 0 and 1.")

    votes = {"low": 0.0, "medium": 0.0, "high": 0.0}
    similarities = []
    supporting_ids = []

    for case in retrieved_cases:
        similarity = float(case["cosine_similarity"])

        if not isfinite(similarity):
            raise ValueError("Non-finite similarity score.")

        # Negative similarities contribute no positive support.
        weight = max(0.0, min(1.0, similarity))
        similarities.append(weight)

        if case["category"] == category and weight > 0:
            priority = case["priority"]
            if priority not in votes:
                raise ValueError(f"Invalid historical priority: {priority}")

            votes[priority] += weight
            supporting_ids.append(case["case_id"])

    case_count = len(retrieved_cases)
    supporting_weight = sum(votes.values())
    review_reasons = []

    if supporting_weight > 0:
        # If vote totals tie exactly, choose the higher priority.
        priority = max(
            votes,
            key=lambda label: (votes[label], PRIORITY_ORDER[label]),
        )
        priority_agreement = votes[priority] / supporting_weight
        priority_is_provisional = False
    else:
        # Required output uses low/medium/high, so use an explicit placeholder.
        priority = "medium"
        priority_agreement = 0.0
        priority_is_provisional = True
        review_reasons.append(
            "No positive retrieved evidence supports the predicted category; "
            "priority is a placeholder requiring human review."
        )

    similarity_strength = (
        sum(similarities) / case_count if case_count else 0.0
    )
    category_agreement = (
        len(supporting_ids) / case_count if case_count else 0.0
    )

    confidence = (
        similarity_strength
        * category_agreement
        * priority_agreement
    )

    if confidence < confidence_threshold:
        review_reasons.append("Confidence is below the selected threshold.")

    # Missing evidence requires review even if the threshold is set to zero.
    escalated = (
        priority_is_provisional
        or confidence < confidence_threshold
    )

    return {
        "priority": priority,
        "priority_is_provisional": priority_is_provisional,
        "priority_source_case_ids": supporting_ids,
        "priority_votes": votes,
        "confidence": confidence,
        "confidence_components": {
            "similarity_strength": similarity_strength,
            "category_agreement": category_agreement,
            "priority_agreement": priority_agreement,
        },
        "escalated": escalated,
        "review_reasons": review_reasons,
    }


def main():
    """Run classification, retrieval, and decisions as a standalone CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 1 <= args.top_k <= 10:
        parser.error("--top-k must be between 1 and 10.")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")

    query = input("Customer inquiry: ").strip()

    print("Classifying...", flush=True)
    classification = classify_inquiry(query)

    print("Retrieving historical cases...", flush=True)
    retrieved = retrieve_cases(query, top_k=args.top_k)

    decision = assess_evidence(
        classification.category,
        retrieved,
        confidence_threshold=args.threshold,
    )

    result = {
        "query": query,
        "category": classification.category,
        "classification_reason": classification.reason,
        "routed_queue": route_category(classification.category),
        **decision,
        "retrieved_past_cases": retrieved,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()