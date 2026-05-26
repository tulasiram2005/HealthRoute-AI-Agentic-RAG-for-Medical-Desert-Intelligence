"""Scenario modeling tests."""

from __future__ import annotations

from src.planning.scenario import rank_intervention_targets, simulate_specialist_deployment, verification_queue
from src.validation.anomaly_detector import detect_anomalies


def test_simulate_specialist_deployment(sample_facility):
    result = simulate_specialist_deployment([sample_facility], [], "pediatrics", 2)
    assert result["specialists_added"] == 2
    assert "estimated_total_patient_impact" in result


def test_rank_intervention_targets(sample_facility):
    targets = rank_intervention_targets([sample_facility], [], "pediatrics")
    assert targets
    assert targets[0].specialty == "pediatrics"


def test_verification_queue(sample_bad_records):
    from src.extraction.parser import parse_facility_text

    facilities = [parse_facility_text(row).extracted_facility for row in sample_bad_records]
    anomalies = [report for facility in facilities for report in detect_anomalies(facility)]
    queue = verification_queue(facilities, anomalies)
    assert queue
    assert queue[0]["risk_score"] >= queue[-1]["risk_score"]

