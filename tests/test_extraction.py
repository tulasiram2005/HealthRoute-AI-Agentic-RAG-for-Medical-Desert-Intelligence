"""Extraction tests."""

from __future__ import annotations

from src.extraction.parser import parse_facility_text


def test_parse_clear_text():
    result = parse_facility_text(
        {
            "source_row_id": "row_clear",
            "id": "fac_clear",
            "name": "Clear Hospital",
            "address_stateOrRegion": "Ashanti",
            "facilityTypeId": "hospital",
            "operatorTypeId": "public",
            "description": "Hospital with cardiology, ECG, echo, ICU and backup generator.",
            "numberDoctors": 5,
            "capacity": 100,
        }
    )
    assert "cardiology" in result.extracted_facility.specialties
    assert result.confidence_scores["overall"] > 0.7


def test_parse_malformed_empty_text():
    result = parse_facility_text({"source_row_id": "empty", "id": "empty", "name": "Empty", "address_stateOrRegion": "Northern", "facilityTypeId": "clinic", "operatorTypeId": "public"})
    assert result.extracted_facility.name == "Empty"


def test_parse_french_terms():
    result = parse_facility_text({"source_row_id": "fr", "id": "fr", "name": "Clinique", "address_stateOrRegion": "Northern", "facilityTypeId": "clinic", "operatorTypeId": "public", "description": "Service pediatrie avec ultrasound et oxygen."})
    assert "pediatrics" in result.extracted_facility.specialties

