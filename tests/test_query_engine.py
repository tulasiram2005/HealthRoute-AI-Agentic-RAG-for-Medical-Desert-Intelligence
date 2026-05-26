"""Grounded planner query tests."""

from __future__ import annotations

from src.models.facility import FacilityModel
from src.planning.query_engine import answer_query
from src.validation.anomaly_detector import detect_anomalies


def test_icu_count_query(sample_facility):
    result = answer_query("How many hospitals in Ghana have ICU capability?", [sample_facility], [])
    assert "1" in result["answer"]
    assert result["sources"] == ["row_test"]


def test_critical_anomaly_query(sample_bad_records):
    facilities = [FacilityModel.model_validate(__import__("src.extraction.parser", fromlist=["parse_facility_text"]).parse_facility_text(row).extracted_facility.model_dump(mode="json")) for row in sample_bad_records]
    anomalies = [report for facility in facilities for report in detect_anomalies(facility)]
    result = answer_query("Which facilities have been flagged as CRITICAL anomalies?", facilities, anomalies)
    assert result["data"]
    assert "critical" in result["answer"].lower()

