"""Planning tests."""

from __future__ import annotations

from src.models.facility import DoctorProfile
from src.planning.desert_scorer import compute_care_index
from src.planning.matcher import match_doctor_to_facilities
from src.planning.recommender import generate_recommendations


def test_care_access_index_bounds(sample_facility):
    index = compute_care_index([sample_facility], [])
    assert 0 <= index.score <= 100


def test_recommendations_format(sample_facility):
    index = compute_care_index([sample_facility], [])
    recs = generate_recommendations([index])
    assert recs
    assert "Deploy" in recs[0]


def test_matching_top_three(sample_facility):
    profile = DoctorProfile(specialty="pediatrics", years_experience=6, equipment_familiar_with=["ECG"], languages=["English"], available_weeks=4)
    matches = match_doctor_to_facilities(profile, [sample_facility], [])
    assert len(matches) == 1
    assert matches[0].match_score >= 0

