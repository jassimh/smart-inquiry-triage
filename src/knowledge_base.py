"""Build and query the persistent historical-case index using Nomic and Chroma."""

import argparse
import csv
from hashlib import sha256
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from src.inspect_data import main as validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "past_cases.csv"
INDEX_PATH = PROJECT_ROOT / ".chroma"
EMBEDDING_MODEL = "nomic-embed-text:latest"


class NomicEmbeddings(Embeddings):
    """Apply the appropriate Nomic prefix automatically."""

    def __init__(self):
        """Configure the local embedding client without generating vectors."""
        self.client = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://127.0.0.1:11434",
            client_kwargs={"timeout": 120},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text using Nomic's search_document prefix."""
        return self.client.embed_documents([
            f"search_document: {text}" for text in texts
        ])

    def embed_query(self, text: str) -> list[float]:
        """Return a query embedding using Nomic's search_query prefix."""
        return self.client.embed_query(f"search_query: {text}")


def load_cases():
    """Read the supplied CSV as dictionaries; this helper does not validate labels."""
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def get_store():
    """Open or create the local cosine-distance collection for the current CSV.

    The collection name includes a CSV hash and a manual embedding version.
    This function does not ingest cases or verify index completeness.
    """
    # A changed CSV gets a separate collection.
    data_version = sha256(CSV_PATH.read_bytes()).hexdigest()[:12]

    return Chroma(
        collection_name=f"past_cases_{data_version}_nomic_v1",
        embedding_function=NomicEmbeddings(),
        persist_directory=str(INDEX_PATH),
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def build_index():
    """Validate the CSV, embed inquiries, and persist cases under stable IDs.

    Stores labels and queues as metadata, processes batches of 32, and prints
    progress. Raises RuntimeError if the final stored IDs differ from the CSV.
    Reruns recompute embeddings and update records with the same IDs.
    """
    # Reuse the audit you already implemented.
    validate_data()

    cases = load_cases()
    store = get_store()
    batch_size = 32

    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]

        documents = [
            Document(
                page_content=case["inquiry_text"],
                metadata={
                    "case_id": case["case_id"],
                    "category": case["category"],
                    "priority": case["priority"],
                    "routed_queue": case["routed_queue"],
                },
            )
            for case in batch
        ]

        store.add_documents(
            documents=documents,
            ids=[case["case_id"] for case in batch],
        )

        print(
            f"Indexed {min(start + batch_size, len(cases))}"
            f"/{len(cases)} cases",
            flush=True,
        )

    stored_ids = set(store.get(include=[])["ids"])
    expected_ids = {case["case_id"] for case in cases}

    if stored_ids != expected_ids:
        raise RuntimeError("The stored case IDs do not match the CSV.")

    print(f"\nIndex ready: {len(stored_ids)} historical cases.")
    print(f"Stored locally in: {INDEX_PATH}")


def retrieve_cases(query: str, top_k: int = 3) -> list[dict]:
    """Return Top-K historical matches with text, metadata, and cosine scores.

    Requires nonblank query text and Top-K from 1 to 10. Searches across all
    categories and converts cosine distance to similarity with 1 - distance.
    Raises RuntimeError for a missing or incomplete case index; model and
    store failures propagate. Similarity ranks neighbours, not their urgency.
    """
    if not query.strip():
        raise ValueError("The inquiry must not be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    store = get_store()
    expected_ids = {case["case_id"] for case in load_cases()}
    stored_ids = set(store.get(include=[])["ids"])

    if not expected_ids or stored_ids != expected_ids:
        raise RuntimeError(
            "The index is missing or incomplete. "
            "Run: python -m src.knowledge_base --ingest"
        )

    matches = store.similarity_search_with_score(
        query,
        k=min(top_k, len(stored_ids)),
    )

    return [
        {
            **document.metadata,
            "inquiry_text": document.page_content,
            "cosine_distance": float(distance),
            "cosine_similarity": 1.0 - float(distance),
        }
        for document, distance in matches
    ]


def main():
    """Build the index with --ingest, or read an inquiry and print ranked matches."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.ingest:
        build_index()
        return

    query = input("Customer inquiry: ").strip()
    results = retrieve_cases(query, top_k=args.top_k)

    for rank, case in enumerate(results, start=1):
        print(
            f"\n{rank}. {case['case_id']} | "
            f"similarity={case['cosine_similarity']:.4f}"
        )
        print(case["inquiry_text"])
        print(
            f"Category: {case['category']} | "
            f"Priority: {case['priority']} | "
            f"Queue: {case['routed_queue']}"
        )


if __name__ == "__main__":
    main()