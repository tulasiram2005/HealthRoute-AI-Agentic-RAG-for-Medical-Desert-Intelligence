"""Doctor-facility matching engine."""

from __future__ import annotations

from src.models.analysis import AnomalyReport
from src.models.facility import DoctorFacilityMatch, DoctorProfile, FacilityModel


def match_doctor_to_facilities(
    profile: DoctorProfile,
    facilities: list[FacilityModel],
    anomalies: list[AnomalyReport] | None = None,
) -> list[DoctorFacilityMatch]:
    """Return top three facilities where a visiting doctor can maximize impact."""

    anomaly_facilities = {report.facility_id for report in anomalies or [] if report.severity == "critical"}
    region_counts: dict[str, int] = {}
    for facility in facilities:
        if profile.specialty in facility.specialties:
            region_counts[facility.address_stateOrRegion] = region_counts.get(facility.address_stateOrRegion, 0) + facility.numberDoctors

    matches: list[DoctorFacilityMatch] = []
    for facility in facilities:
        if profile.preferred_regions and facility.address_stateOrRegion not in profile.preferred_regions:
            continue
        region_doctors = region_counts.get(facility.address_stateOrRegion, 0)
        if profile.specialty not in facility.specialties:
            need = 100
            need_text = f"{facility.address_stateOrRegion} has no recorded {profile.specialty} coverage in this sample."
        elif region_doctors < 2:
            need = 75
            need_text = f"{facility.address_stateOrRegion} has limited {profile.specialty} coverage."
        else:
            need = 35
            need_text = f"{facility.address_stateOrRegion} already has some {profile.specialty} coverage."
        equipment_text = " ".join(facility.equipment).lower()
        compatible = sum(item.lower() in equipment_text for item in profile.equipment_familiar_with)
        infra = min(100, 35 + compatible * 20 + min(len(facility.equipment), 5) * 5)
        if any(term in equipment_text for term in ["generator", "backup power", "icu", "oxygen"]):
            infra += 10
        if facility.id in anomaly_facilities:
            infra -= 35
        score = max(0.0, min(100.0, (0.6 * need) + (0.4 * infra)))
        matches.append(
            DoctorFacilityMatch(
                doctor_specialty=profile.specialty,
                matched_facility_id=facility.id,
                match_score=round(score, 2),
                rationale=f"{need_text} Infrastructure score is {round(infra, 1)}.",
                need_rationale=need_text,
                infra_rationale=f"Compatible equipment matches: {compatible}; equipment count: {len(facility.equipment)}.",
                estimated_patient_impact=int(profile.available_weeks * (20 + profile.years_experience * 1.5)),
                citations=[facility.source_row_id or facility.id],
            )
        )
    return sorted(matches, key=lambda item: item.match_score, reverse=True)[:3]

