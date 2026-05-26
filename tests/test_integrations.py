"""Integration adapter tests."""

from __future__ import annotations

from src.architecture.compliance import TECH_STACK_MATRIX, compliance_score
from src.integrations.databricks_llm import DatabricksLLMClient
from src.integrations.genie import plan_facility_query
from src.integrations.mlflow_tracing import MLflowTraceLogger


def test_compliance_matrix_covers_required_stack():
    score = compliance_score()
    assert score["implemented"] == score["total"]
    assert any("MLflow" in row["requested"] for row in TECH_STACK_MATRIX)
    assert any("LanceDB" in row["requested"] for row in TECH_STACK_MATRIX)


def test_databricks_llm_fallback_without_credentials(monkeypatch):
    monkeypatch.delenv("DATABRICKS_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    client = DatabricksLLMClient(endpoint_url=None, token=None)
    response = client.chat([{"role": "user", "content": "Summarize ICU gaps"}])
    assert response.used_fallback is True
    assert "ICU" in response.text


def test_genie_query_plan():
    plan = plan_facility_query("List facilities with MRI equipment")
    assert "MRI" in plan.sql
    assert plan.used_native_genie is False


def test_mlflow_trace_fallback(tmp_path):
    logger = MLflowTraceLogger(fallback_path=str(tmp_path / "trace.jsonl"))
    logger.mlflow = None
    with logger.run("query_test") as run_id:
        logger.log_step(run_id, "step_1_extraction", {"rows": 1}, {"ok": True}, ["row_001"])
    assert (tmp_path / "trace.jsonl").exists()

