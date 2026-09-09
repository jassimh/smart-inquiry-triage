"""Demonstrate semantic ranking over three fictional texts without using the CSV index."""

from math import sqrt

from langchain_ollama import OllamaEmbeddings


def cosine_similarity(vector_a, vector_b):
    """Return the cosine similarity of two equal-dimensional nonzero vectors.

    Raises ValueError for mismatched dimensions or zero vector magnitude.
    The score compares vector directions; it is not a correctness probability.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding dimensions do not match.")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    length_a = sqrt(sum(a * a for a in vector_a))
    length_b = sqrt(sum(b * b for b in vector_b))

    if length_a == 0 or length_b == 0:
        raise ValueError("Cannot compare a zero-length vector.")

    return dot_product / (length_a * length_b)


def main():
    """Embed three fictional examples and a query, then print their similarity ranking."""
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://127.0.0.1:11434",
        client_kwargs={"timeout": 120},
    )

    # Fictional examples for this exercise.
    documents = [
        "My monthly vehicle payment was deducted twice.",
        "The vehicle configurator crashes when I change the paint color.",
        "Please remove me from your promotional email list.",
    ]

    query = "I was charged two times for the same monthly payment."

    print("Generating embeddings...", flush=True)

    document_vectors = embeddings.embed_documents([
        f"search_document: {text}" for text in documents
    ])

    query_vector = embeddings.embed_query(
        f"search_query: {query}"
    )

    print(f"\nNumber of document vectors: {len(document_vectors)}")
    print(f"Numbers per vector: {len(query_vector)}")
    print(f"First five query values: {query_vector[:5]}")

    ranked_results = sorted(
        [
            (cosine_similarity(query_vector, vector), text)
            for text, vector in zip(documents, document_vectors)
        ],
        key=lambda result: result[0],
        reverse=True,
    )

    print(f"\nQuery: {query}")
    print("\nDocuments ranked by similarity:")

    for score, text in ranked_results:
        print(f"{score:.4f} | {text}")


if __name__ == "__main__":
    main()