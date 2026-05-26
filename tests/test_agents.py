"""Agent pipeline content tests."""

from __future__ import annotations

import asyncio

from src.agents.graph import run_agent
from src.agents.nodes import anomaly_node, extract_node, recommend_node, score_node


def test_extract_node_returns_facility_name(sample_facility_row):
    state = asyncio.run(extract_node({"source_row": sample_facility_row, "citations": {}}))
    facility = state["extracted_data"]["extracted_facility"]
    assert facility["name"] == "Test Regional Hospital"
    assert "cardiology" in facility["specialties"]
    assert state["citations"]["step_1_extraction"] == ["row_test"]


def test_anomaly_node_detects_icu_without_power():
    row = {
        "source_row_id": "bad_power",
        "id": "bad_power",
        "name": "No Power ICU",
        "address_stateOrRegion": "Greater Accra",
        "facilityTypeId": "hospital",
        "operatorTypeId": "public",
        "numberDoctors": 3,
        "capacity": 20,
        "specialties": ["emergencyMedicine"],
        "equipment": ["oxygen"],
        "capability": ["ICU"],
    }
    state = asyncio.run(anomaly_node({"source_row": row, "source_row_id": "bad_power", "raw_text": "ICU with oxygen only", "citations": {}}))
    assert any(item["anomaly_type"] == "ICU_WITHOUT_POWER" for item in state["anomalies"])


def test_score_node_returns_valid_index(sample_facility_row):
    state = asyncio.run(score_node({"source_row": sample_facility_row, "source_row_id": "row_test", "citations": {}}))
    assert 0 <= state["care_index"]["score"] <= 100
    assert state["care_index"]["region"] == "Test Region"


def test_recommend_node_returns_non_empty_list(sample_facility_row):
    state = asyncio.run(recommend_node({"source_row": sample_facility_row, "source_row_id": "row_test", "citations": {}}))
    assert state["recommendations"]
    assert "Test Region" in state["recommendations"][0]


def test_full_pipeline_end_to_end(sample_facility_row):
    state = asyncio.run(run_agent(sample_facility_row))
    assert state["extracted_data"]["extracted_facility"]["name"] == "Test Regional Hospital"
    assert state["validated_data"]["name"] == "Test Regional Hospital"
    assert "care_index" in state
    assert state["recommendations"]


def test_citations_populated_at_all_5_steps(sample_facility_row):
    state = asyncio.run(run_agent(sample_facility_row))
    for step in [
        "step_1_extraction",
        "step_2_validation",
        "step_3_anomaly",
        "step_4_scoring",
        "step_5_recommendation",
    ]:
        assert state["citations"].get(step), f"Missing citations for {step}"
