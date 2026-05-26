"""Model validation tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models.extraction import ExtractionResult
from src.models.facility import DoctorProfile, FacilityModel


def test_valid_facility(sample_facility):
    assert sample_facility.name == "Test Regional Hospital"
    assert "cardiology" in sample_facility.specialties


@pytest.mark.parametrize("phone", ["233300000000", "+0123", "abc"])
def test_invalid_e164(sample_facility_row, phone):
    sample_facility_row["phone_numbers"] = [phone]
    with pytest.raises(ValidationError):
        FacilityModel.model_validate(sample_facility_row)


def test_invalid_iso(sample_facility_row):
    sample_facility_row["address_countryCode"] = "ZZZ"
    with pytest.raises(ValidationError):
        FacilityModel.model_validate(sample_facility_row)


def test_invalid_specialty(sample_facility_row):
    sample_facility_row["specialties"] = ["neurosurgery"]
    with pytest.raises(ValidationError):
        FacilityModel.model_validate(sample_facility_row)


def test_invalid_enum(sample_facility_row):
    sample_facility_row["facilityTypeId"] = "spa"
    with pytest.raises(ValidationError):
        FacilityModel.model_validate(sample_facility_row)


def test_confidence_bounds(sample_facility):
    with pytest.raises(ValidationError):
        ExtractionResult(source_row_id="x", extracted_facility=sample_facility, confidence_scores={"bad": 1.5})


def test_doctor_profile():
    profile = DoctorProfile(specialty="pediatrics", years_experience=10, equipment_familiar_with=[], languages=["English"], available_weeks=4)
    assert profile.specialty == "pediatrics"

