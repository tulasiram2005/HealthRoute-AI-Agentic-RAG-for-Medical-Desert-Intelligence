"""Anomaly detection tests."""

from __future__ import annotations

from src.extraction.parser import parse_facility_text
from src.validation.anomaly_detector import aggregate_risk_score, detect_anomalies


def test_known_bad_records_are_caught(sample_bad_records):
    reports = []
    for row in sample_bad_records:
        facility = parse_facility_text(row).extracted_facility
        reports.extend(detect_anomalies(facility))
    kinds = {report.anomaly_type for report in reports}
    assert {"ICU_WITHOUT_POWER", "SURGERY_NO_DOCTORS", "PHANTOM_EQUIPMENT", "SPECIALTY_WITHOUT_EQUIPMENT", "ZERO_CAPACITY_OPERATING"} <= kinds


def test_surgery_no_doctors_critical(sample_bad_records):
    facility = parse_facility_text(sample_bad_records[1]).extracted_facility
    reports = detect_anomalies(facility)
    assert any(report.anomaly_type == "SURGERY_NO_DOCTORS" and report.severity == "critical" for report in reports)


def test_clean_record_has_no_critical(sample_facility):
    reports = detect_anomalies(sample_facility)
    assert not any(report.severity == "critical" for report in reports)


def test_risk_score(sample_bad_records):
    facility = parse_facility_text(sample_bad_records[0]).extracted_facility
    assert aggregate_risk_score(detect_anomalies(facility)) > 0

