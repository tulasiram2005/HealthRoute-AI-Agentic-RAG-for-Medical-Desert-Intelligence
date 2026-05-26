"""LanceDB ingestion pipeline with JSON fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.rag.embeddings import EmbeddingModel
from src.integrations.lancedb_store import write_vector_store


def ingest_csv(data_path: str, output_dir: str = "artifacts/lancedb_fallback") -> list[dict]:
    """Load facility CSV, embed rich text, and persist searchable records."""

    df = pd.read_csv(data_path)
    embedder = EmbeddingModel()
    rows: list[dict] = []
    texts: list[str] = []
    for _, row in df.iterrows():
        text = " ".join(str(row.get(col, "") or "") for col in ["procedure", "equipment", "capability", "description"])
        texts.append(text)
        rows.append(
            {
                "id": str(row.get("id") or row.get("source_row_id")),
                "text": text,
                "metadata": {
                    "facility_id": str(row.get("id")),
                    "region": row.get("address_stateOrRegion"),
                    "district": row.get("address_district"),
                    "facilityTypeId": row.get("facilityTypeId"),
                    "specialties": row.get("specialties"),
                },
            }
        )
    vectors = embedder.embed(texts)
    for record, vector in zip(rows, vectors):
        record["vector"] = vector
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "facility_capabilities.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_vector_store(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ghana_facilities.csv")
    parser.add_argument("--output-dir", default="artifacts/lancedb_fallback")
    args = parser.parse_args()
    rows = ingest_csv(args.data, args.output_dir)
    print(f"Ingested {len(rows)} records")


if __name__ == "__main__":
    main()
