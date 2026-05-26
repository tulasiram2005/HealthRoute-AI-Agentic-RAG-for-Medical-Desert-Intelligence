"""Dashboard data helpers."""

from __future__ import annotations

from collections import Counter

from src.models.analysis import AnomalyReport, CareAccessIndex
from src.models.facility import FacilityModel, Specialty


def specialty_gap_matrix(facilities: list[FacilityModel]) -> dict[str, dict[str, bool]]:
    """Return region by specialty coverage booleans."""

    regions = sorted({facility.address_stateOrRegion for facility in facilities})
    matrix = {region: {specialty.value: False for specialty in Specialty} for region in regions}
    for facility in facilities:
        for specialty in facility.specialties:
            matrix[facility.address_stateOrRegion][specialty] = True
    return matrix


def specialty_gap_numeric(facilities: list[FacilityModel]) -> dict[str, dict[str, int]]:
    """Return region by specialty coverage as 0/1 values for heatmaps."""

    return {
        region: {specialty: int(covered) for specialty, covered in values.items()}
        for region, values in specialty_gap_matrix(facilities).items()
    }


def executive_metrics(
    facilities: list[FacilityModel],
    anomalies: list[AnomalyReport],
    indices: list[CareAccessIndex],
) -> dict[str, int | float]:
    """Compute top-line planner metrics for the home screen."""

    return {
        "facilities_analyzed": len(facilities),
        "regions_covered": len({facility.address_stateOrRegion for facility in facilities}),
        "medical_deserts": sum(index.score < 40 for index in indices),
        "critical_anomalies": sum(report.severity == "critical" for report in anomalies),
        "avg_care_index": round(sum(index.score for index in indices) / max(1, len(indices)), 1),
        "source_rows": len({facility.source_row_id for facility in facilities}),
    }


def anomaly_severity_counts(anomalies: list[AnomalyReport]) -> dict[str, int]:
    """Count anomalies by severity in stable order."""

    counts = Counter(report.severity for report in anomalies)
    return {severity: counts.get(severity, 0) for severity in ["critical", "high", "medium", "low"]}


def facility_readiness_rows(facilities: list[FacilityModel], anomalies: list[AnomalyReport]) -> list[dict]:
    """Create a scannable readiness table for planners."""

    critical_ids = {report.facility_id for report in anomalies if report.severity in {"critical", "high"}}
    rows = []
    for facility in facilities:
        infra_terms = " ".join(facility.equipment).lower()
        readiness = 50 + min(25, len(facility.equipment) * 4) + min(15, facility.numberDoctors * 2)
        if "generator" in infra_terms or "backup power" in infra_terms:
            readiness += 10
        if facility.id in critical_ids:
            readiness -= 25
        rows.append(
            {
                "facility": facility.name,
                "region": facility.address_stateOrRegion,
                "doctors": facility.numberDoctors,
                "beds": facility.capacity,
                "readiness_score": max(0, min(100, readiness)),
                "needs_verification": facility.id in critical_ids,
                "source_row_id": facility.source_row_id or facility.id,
            }
        )
    return sorted(rows, key=lambda row: row["readiness_score"], reverse=True)
