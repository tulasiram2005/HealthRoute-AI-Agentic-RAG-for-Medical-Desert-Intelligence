"""Async LangGraph-compatible agent nodes."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, TypedDict

from src.extraction.parser import parse_facility_text
from src.models.extraction import ExtractionResult
from src.models.facility import Specialty
from src.planning.desert_scorer import compute_care_index
from src.planning.recommender import generate_recommendations
from src.validation.anomaly_detector import detect_anomalies


class AgentState(TypedDict, total=False):
    """State flowing through the IDP graph."""

    raw_text: str
    source_row_id: str
    source_row: dict[str, Any]
    extracted_data: dict[str, Any]
    validated_data: dict[str, Any]
    anomalies: list[Any]
    care_index: Any
    recommendations: list[str]
    citations: dict[str, list[str]]
    rag_context: dict[str, Any]
    mlflow_run_id: str
    error: str | None
    retries: int


def _cite(state: AgentState, step: str) -> None:
    citations = state.setdefault("citations", {})
    citations.setdefault(step, [])
    row = state.get("source_row_id", "row_unknown")
    if row not in citations[step]:
        citations[step].append(row)


async def extract_node(state: AgentState) -> AgentState:
    """Parse free-form source data into a structured facility profile."""

    try:
        row = dict(state.get("source_row") or {})
        row.setdefault("source_row_id", state.get("source_row_id", "row_unknown"))
        row.setdefault("description", state.get("raw_text", ""))
        result = parse_facility_text(row)
        rag_context = _rag_context_for_row(row)
        state["rag_context"] = rag_context
        for row_id in rag_context.get("source_row_ids", []):
            state.setdefault("citations", {}).setdefault("step_1_extraction_rag", [])
            if row_id not in state["citations"]["step_1_extraction_rag"]:
                state["citations"]["step_1_extraction_rag"].append(row_id)
        llm_data = await _llm_enhance_extraction(row, result, rag_context.get("context", ""))
        if llm_data:
            result = _merge_extraction(result, llm_data)
        state["source_row_id"] = result.source_row_id
        state["extracted_data"] = result.model_dump(mode="json")
        _cite(state, "step_1_extraction")
    except Exception as exc:
        state["error"] = f"extract_node failed: {exc}"
    return state


async def validate_node(state: AgentState) -> AgentState:
    """Use strict Pydantic validation result from extraction."""

    if not state.get("extracted_data"):
        state["error"] = state.get("error") or "Nothing extracted for validation."
        return state
    state["validated_data"] = state["extracted_data"]["extracted_facility"]
    _cite(state, "step_2_validation")
    return state


async def anomaly_node(state: AgentState) -> AgentState:
    """Detect suspicious and contradictory facility claims."""

    try:
        result = parse_facility_text(state.get("source_row") or {"description": state.get("raw_text", ""), "source_row_id": state.get("source_row_id", "row_unknown")})
        reports = detect_anomalies(result.extracted_facility, state.get("raw_text", ""))
        semantic = await _llm_semantic_anomaly_check(
            state.get("raw_text", ""),
            result.extracted_facility.name,
            result.extracted_facility.id,
            result.source_row_id,
        )
        state["anomalies"] = [report.model_dump(mode="json") for report in reports] + semantic
        _cite(state, "step_3_anomaly")
    except Exception as exc:
        state["error"] = f"anomaly_node failed: {exc}"
    return state


def _rag_context_for_row(row: dict) -> dict:
    """Retrieve similar facility chunks for extraction grounding."""

    try:
        from src.rag.rag_query import rag_ground_query

        query = " ".join(str(row.get(key, "") or "") for key in ["name", "description", "procedure", "equipment", "capability"])
        return rag_ground_query(query, top_k=5)
    except Exception:
        return {"context": "", "source_row_ids": [], "scores": [], "results": []}


async def _llm_enhance_extraction(row: dict, base_result: ExtractionResult, rag_context: str = "") -> dict | None:
    """Call Databricks/OpenAI-compatible LLM for enhanced extraction."""

    try:
        from src.agents.prompts import EXTRACT_SYSTEM_PROMPT, EXTRACT_USER_TEMPLATE
        from src.integrations.databricks_llm import DatabricksLLMClient

        client = DatabricksLLMClient()
        if client.token is None:
            return None
        user_msg = EXTRACT_USER_TEMPLATE.format(
            source_row_id=row.get("source_row_id", base_result.source_row_id),
            name=row.get("name", "Unknown"),
            description=row.get("description", ""),
            procedure=row.get("procedure", ""),
            equipment=row.get("equipment", ""),
            capability=row.get("capability", ""),
            rag_context=rag_context or "No retrieved context available.",
        )
        response = client.chat([{"role": "system", "content": EXTRACT_SYSTEM_PROMPT}, {"role": "user", "content": user_msg}])
        if response.used_fallback:
            return None
        text = _strip_json_fence(response.text)
        return json.loads(text)
    except Exception:
        return None


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) > 1:
            text = parts[1].removeprefix("json").strip()
    return text


def _merge_extraction(base_result: ExtractionResult, llm_data: dict) -> ExtractionResult:
    """Merge LLM findings into deterministic extraction without losing fallback data."""

    facility = base_result.extracted_facility
    data = facility.model_dump(mode="python")
    valid_specialties = {specialty.value for specialty in Specialty}
    for list_field in ["procedure", "equipment", "capability"]:
        additions = [str(item).strip() for item in llm_data.get(list_field, []) if str(item).strip()]
        if additions:
            data[list_field] = sorted(set(data.get(list_field, [])) | set(additions))
    if llm_data.get("specialties"):
        additions = [s for s in llm_data["specialties"] if s in valid_specialties]
        data["specialties"] = sorted(set(data.get("specialties", [])) | set(additions))
    for numeric in ["numberDoctors", "capacity"]:
        if isinstance(llm_data.get(numeric), int) and llm_data[numeric] >= 0:
            data[numeric] = max(data.get(numeric, 0), llm_data[numeric])
    if llm_data.get("facilityTypeId"):
        data["facilityTypeId"] = llm_data["facilityTypeId"]
    merged_facility = facility.__class__.model_validate(data)
    llm_confidence = llm_data.get("confidence_scores", {}) or {}
    merged_confidence = dict(base_result.confidence_scores)
    for key, value in llm_confidence.items():
        if isinstance(value, (int, float)):
            merged_confidence[key] = max(float(merged_confidence.get(key, 0.0)), min(1.0, float(value)))
    return ExtractionResult(
        source_row_id=base_result.source_row_id,
        extracted_facility=merged_facility,
        confidence_scores=merged_confidence,
        extraction_timestamp=base_result.extraction_timestamp or datetime.utcnow(),
        agent_step_id="extract_llm_enhanced",
    )


async def _llm_semantic_anomaly_check(facility_text: str, facility_name: str, facility_id: str, source_row_id: str) -> list[dict]:
    """LLM-powered semantic contradiction detection with deterministic fallback."""

    try:
        from src.agents.prompts import SEMANTIC_ANOMALY_SYSTEM_PROMPT
        from src.integrations.databricks_llm import DatabricksLLMClient

        client = DatabricksLLMClient()
        if client.token is None:
            return []
        response = client.chat(
            [
                {"role": "system", "content": SEMANTIC_ANOMALY_SYSTEM_PROMPT},
                {"role": "user", "content": f"Facility: {facility_name}\n\nData:\n{facility_text}"},
            ]
        )
        if response.used_fallback:
            return []
        text = _strip_json_fence(response.text)
        items = json.loads(text) if text.startswith("[") else []
        normalized = []
        for item in items:
            normalized.append(
                {
                    "facility_id": facility_id,
                    "anomaly_type": str(item.get("anomaly_type", "SEMANTIC_CONTRADICTION")),
                    "description": str(item.get("description", "")),
                    "evidence": str(item.get("evidence", "")),
                    "severity": str(item.get("severity", "medium")),
                    "source_row_id": source_row_id,
                    "confidence": float(item.get("confidence", 0.7)),
                }
            )
        return normalized
    except Exception:
        return []


async def score_node(state: AgentState) -> AgentState:
    """Compute Care Access Index for the source facility region."""

    try:
        result = parse_facility_text(state.get("source_row") or {"description": state.get("raw_text", ""), "source_row_id": state.get("source_row_id", "row_unknown")})
        reports = detect_anomalies(result.extracted_facility, state.get("raw_text", ""))
        care_index = compute_care_index([result.extracted_facility], reports, result.extracted_facility.address_stateOrRegion)
        state["care_index"] = care_index.model_dump(mode="json")
        _cite(state, "step_4_scoring")
    except Exception as exc:
        state["error"] = f"score_node failed: {exc}"
    return state


async def recommend_node(state: AgentState) -> AgentState:
    """Generate actionable recommendations."""

    try:
        result = parse_facility_text(state.get("source_row") or {"description": state.get("raw_text", ""), "source_row_id": state.get("source_row_id", "row_unknown")})
        care_index = compute_care_index([result.extracted_facility], [], result.extracted_facility.address_stateOrRegion)
        state["recommendations"] = generate_recommendations([care_index])
        _cite(state, "step_5_recommendation")
    except Exception as exc:
        state["error"] = f"recommend_node failed: {exc}"
    return state
