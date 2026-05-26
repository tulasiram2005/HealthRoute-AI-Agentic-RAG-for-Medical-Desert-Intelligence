"""Care Access Index computation."""

from __future__ import annotations

from collections import defaultdict

from src.models.analysis import AnomalyReport, CareAccessIndex
from src.models.facility import FacilityModel, Specialty


def compute_care_index(
    facilities: list[FacilityModel],
    anomalies: list[AnomalyReport] | None = None,
    region: str | None = None,
    district: str | None = None,
) -> CareAccessIndex:
    """Compute a regional Care Access Index using weighted coverage signals."""

    selected = [facility for facility in facilities if (region is None or facility.address_stateOrRegion == region)]
    if district:
        selected = [facility for facility in selected if facility.address_district == district]
    if not selected:
        return CareAccessIndex(region=region or "Unknown", district=district or "All", score=0.0, specialty_gaps=[s.value for s in Specialty], flagged_facilities=[], recommendation_priority=5)

    covered = {specialty for facility in selected for specialty in facility.specialties}
    all_specialties = {specialty.value for specialty in Specialty}
    gaps = sorted(all_specialties - set(covered))
    specialty_score = 100 * len(covered) / len(all_specialties)
    equipment_score = min(100.0, sum(len(facility.equipment) for facility in selected) * 8)
    capacity_score = min(100.0, sum(facility.capacity for facility in selected) / 10)
    infra_hits = sum(any(term in " ".join(facility.equipment).lower() for term in ["generator", "backup power", "oxygen"]) for facility in selected)
    infra_score = 100 * infra_hits / len(selected)
    score = (0.4 * specialty_score) + (0.3 * equipment_score) + (0.2 * capacity_score) + (0.1 * infra_score)
    flagged = sorted({report.facility_id for report in (anomalies or []) if report.facility_id in {f.id for f in selected}})
    if score < 40:
        priority = 5
    elif score < 55:
        priority = 4
    elif score < 70:
        priority = 3
    elif score < 85:
        priority = 2
    else:
        priority = 1
    return CareAccessIndex(
        region=region or selected[0].address_stateOrRegion,
        district=district or "All",
        score=round(score, 2),
        specialty_gaps=gaps,
        flagged_facilities=flagged,
        recommendation_priority=priority,
    )


def compute_indices_by_region(facilities: list[FacilityModel], anomalies: list[AnomalyReport] | None = None) -> list[CareAccessIndex]:
    """Compute Care Access Index for every region."""

    regions = sorted({facility.address_stateOrRegion for facility in facilities})
    return [compute_care_index(facilities, anomalies, region=region) for region in regions]

