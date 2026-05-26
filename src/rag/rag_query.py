"""RAG grounding helpers for agent and query workflows."""

from __future__ import annotations

from src.rag.retrieval import FacilityRetriever

_retriever: FacilityRetriever | None = None


def get_retriever() -> FacilityRetriever:
    """Return a cached facility retriever."""

    global _retriever
    if _retriever is None:
        _retriever = FacilityRetriever()
    return _retriever


def rag_ground_query(query: str, top_k: int = 5) -> dict:
    """Ground a query against the LanceDB/JSON vector store."""

    try:
        results = get_retriever().search(query, top_k=top_k)
    except Exception:
        results = []
    context = "\n\n".join(f"[{r['source_row_id']}] {r['text_snippet']}" for r in results)
    return {
        "context": context,
        "source_row_ids": [r["source_row_id"] for r in results],
        "scores": [r["similarity_score"] for r in results],
        "results": results,
    }
