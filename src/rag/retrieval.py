"""Hybrid retrieval over fallback JSON records."""

from __future__ import annotations

import json
import math
from pathlib import Path

from src.rag.embeddings import EmbeddingModel


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / ((math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0))


class FacilityRetriever:
    """Hybrid vector and keyword search for facility capabilities."""

    def __init__(self, store_path: str = "artifacts/lancedb_fallback/facility_capabilities.json") -> None:
        self.store_path = Path(store_path)
        self.embedder = EmbeddingModel()
        self.records = json.loads(self.store_path.read_text(encoding="utf-8")) if self.store_path.exists() else []

    def search(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        """Return top-K retrieval results with source citations."""

        filters = filters or {}
        query_vector = self.embedder.embed([query])[0]
        query_terms = set(query.lower().split())
        results: list[dict] = []
        for record in self.records:
            metadata = record.get("metadata", {})
            if any(str(metadata.get(key)) != str(value) for key, value in filters.items() if value):
                continue
            vector_score = _cosine(query_vector, record.get("vector", []))
            text_terms = set(record.get("text", "").lower().split())
            keyword_score = len(query_terms & text_terms) / max(1, len(query_terms))
            score = (0.7 * vector_score) + (0.3 * keyword_score)
            results.append(
                {
                    "source_row_id": record["id"],
                    "similarity_score": round(score, 4),
                    "text_snippet": record["text"][:240],
                    "metadata": metadata,
                }
            )
        return sorted(results, key=lambda item: item["similarity_score"], reverse=True)[:top_k]

