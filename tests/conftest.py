"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.extraction.parser import parse_facility_text


@pytest.fixture
def sample_facility_row() -> dict:
    return {
        "source_row_id": "row_test",
        "id": "fac_test",
        "name": "Test Regional Hospital",
        "phone_numbers": ["+233300000000"],
        "address_stateOrRegion": "Test Region",
        "address_district": "Test District",
        "facilityTypeId": "hospital",
        "operatorTypeId": "public",
        "numberDoctors": 5,
        "capacity": 80,
        "specialties": ["cardiology", "generalSurgery"],
        "equipment": ["ECG", "echo", "backup generator", "operating theatre"],
        "capability": ["ICU", "surgery"],
        "procedure": ["general surgery"],
    }


@pytest.fixture
def sample_facility(sample_facility_row):
    return parse_facility_text(sample_facility_row).extracted_facility


@pytest.fixture
def sample_bad_records() -> list[dict]:
    return [
        {"source_row_id": "bad_1", "id": "bad_1", "name": "No Power ICU", "address_stateOrRegion": "A", "facilityTypeId": "hospital", "operatorTypeId": "public", "numberDoctors": 3, "capacity": 20, "specialties": ["emergencyMedicine"], "equipment": ["oxygen"], "capability": ["ICU"]},
        {"source_row_id": "bad_2", "id": "bad_2", "name": "No Doctor Surgery", "address_stateOrRegion": "A", "facilityTypeId": "hospital", "operatorTypeId": "public", "numberDoctors": 0, "capacity": 20, "specialties": ["generalSurgery"], "equipment": ["operating theatre"], "capability": ["surgery"]},
        {"source_row_id": "bad_3", "id": "bad_3", "name": "Tiny MRI", "address_stateOrRegion": "A", "facilityTypeId": "clinic", "operatorTypeId": "private", "numberDoctors": 1, "capacity": 5, "specialties": ["familyMedicine"], "equipment": ["MRI"], "capability": ["outpatient"]},
        {"source_row_id": "bad_4", "id": "bad_4", "name": "Cardiology No ECG", "address_stateOrRegion": "A", "facilityTypeId": "clinic", "operatorTypeId": "private", "numberDoctors": 1, "capacity": 10, "specialties": ["cardiology"], "equipment": ["oxygen"], "capability": ["outpatient"]},
        {"source_row_id": "bad_5", "id": "bad_5", "name": "Zero Capacity Ops", "address_stateOrRegion": "A", "facilityTypeId": "clinic", "operatorTypeId": "private", "numberDoctors": 1, "capacity": 0, "specialties": ["familyMedicine"], "equipment": ["oxygen"], "procedure": ["wound care"], "capability": ["outpatient"]},
    ]

