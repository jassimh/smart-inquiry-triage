import json
from pathlib import Path

import pandas as pd


# Find the project folder relative to this script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def main():
    # Load the historical cases and category definitions.
    cases = pd.read_csv(DATA_DIR / "past_cases.csv")

    with (DATA_DIR / "taxonomy.json").open(encoding="utf-8") as file:
        taxonomy = json.load(file)

    allowed_categories = {
        category["name"] for category in taxonomy["categories"]
    }
    allowed_priorities = {"low", "medium", "high"}
    required_columns = {
        "case_id", "inquiry_text", "category", "priority", "routed_queue"
    }

    # Check the structure before accessing individual columns.
    missing_columns = required_columns - set(cases.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if cases.empty:
        raise ValueError("The historical dataset is empty.")

    # Required fields must contain nonblank text.
    for column in sorted(required_columns):
        values = cases[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing or blank values in: {column}")

    if cases["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs found.")

    if not cases["category"].isin(allowed_categories).all():
        raise ValueError("A case contains a category outside the taxonomy.")

    if not cases["priority"].isin(allowed_priorities).all():
        raise ValueError("A case contains an invalid priority.")

    queue_counts = cases.groupby("category")["routed_queue"].nunique()
    if (queue_counts > 1).any():
        raise ValueError("A category maps to multiple queues; review routing.")

    print(f"Validation passed: {len(cases)} historical cases.")

    print("\nCases per category:")
    print(cases["category"].value_counts())

    print("\nCases per priority:")
    print(cases["priority"].value_counts())

    print("\nCategory-to-queue mapping:")
    print(
        cases[["category", "routed_queue"]]
        .drop_duplicates()
        .sort_values("category")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()