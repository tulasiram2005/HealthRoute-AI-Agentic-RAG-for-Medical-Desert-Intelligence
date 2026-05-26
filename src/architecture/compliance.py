"""Challenge tech-stack compliance matrix."""

from __future__ import annotations


TECH_STACK_MATRIX = [
    {
        "requirement": "Agentic orchestration",
        "requested": "LangGraph primary, CrewAI roles",
        "implemented": "LangGraph StateGraph path with direct fallback; CrewAI role builder",
        "status": "Implemented with graceful fallback",
        "files": "src/agents/graph.py, src/agents/crew.py",
    },
    {
        "requirement": "ML lifecycle/tracing",
        "requested": "MLflow experiment tracking and traces",
        "implemented": "MLflowTraceLogger logs runs when MLflow is installed, JSON fallback otherwise",
        "status": "Implemented optional native integration",
        "files": "src/integrations/mlflow_tracing.py",
    },
    {
        "requirement": "RAG/vector store",
        "requested": "LanceDB",
        "implemented": "LanceDB writer when available; JSON vector fallback for offline demos",
        "status": "Implemented optional native integration",
        "files": "src/rag/ingest.py, src/integrations/lancedb_store.py",
    },
    {
        "requirement": "LLM",
        "requested": "Databricks-hosted LLM via MLflow AI Gateway or OpenAI-compatible endpoint",
        "implemented": "Databricks/OpenAI-compatible chat client with deterministic fallback",
        "status": "Credential-ready",
        "files": "src/integrations/databricks_llm.py",
    },
    {
        "requirement": "Text2SQL",
        "requested": "Databricks Genie",
        "implemented": "Genie-style adapter and deterministic query router for demo",
        "status": "Credential-ready",
        "files": "src/integrations/genie.py, src/planning/query_engine.py",
    },
    {
        "requirement": "Validation",
        "requested": "Pydantic v2 strict mode",
        "implemented": "Strict Pydantic models with enum/field validators",
        "status": "Implemented",
        "files": "src/models/",
    },
    {
        "requirement": "Backend/UI",
        "requested": "FastAPI and Streamlit",
        "implemented": "FastAPI planner API and Streamlit command center",
        "status": "Implemented",
        "files": "src/api/, src/ui/app.py",
    },
]


def compliance_score() -> dict:
    """Summarize implementation readiness."""

    implemented = sum("Implemented" in row["status"] or "Credential-ready" in row["status"] for row in TECH_STACK_MATRIX)
    return {"implemented": implemented, "total": len(TECH_STACK_MATRIX), "percent": round(100 * implemented / len(TECH_STACK_MATRIX))}

