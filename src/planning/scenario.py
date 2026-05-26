"""Scenario modeling for NGO resource allocation."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.analysis import AnomalyReport
from src.models.facility import FacilityModel, Specialty
from src.planning.desert_scorer import compute_indices_by_region


@dataclass(frozen=True)
class InterventionTarget:
    """Ranked target for a clinical deployment."""

    region: str
    district: str
    specialty: str
    urgency_score: float
    expected_patient_impact: int
    rationale: str
    citations: list[str]


def rank_intervention_targets(
    facilities: list[FacilityModel],
    anomalies: list[AnomalyReport],
    specialty: str | None = None,
) -> list[InterventionTarget]:
    """Rank where a new specialist or verification visit would create the most impact."""

    indices = compute_indices_by_region(facilities, anomalies)
    targets: list[InterventionTarget] = []
    for index in indices:
        region_facilities = [f for f in facilities if f.address_stateOrRegion == index.region]
        critical_penalty = sum(1 for report in anomalies if report.facility_id in {f.id for f in region_facilities} and report.severity in {"critical", "high"})
        gaps = [specialty] if specialty else index.specialty_gaps[:3]
        for gap in gaps:
            if gap not in {s.value for s in Specialty}:
                continue
            urgency = round((100 - index.score) + critical_penalty * 8 + index.recommendation_priority * 4, 2)
            impact = int(max(2500, urgency * 950 + sum(f.capacity for f in region_facilities) * 3))
            targets.append(
                InterventionTarget(
                    region=index.region,
                    district=index.district,
                    specialty=gap,
                    urgency_score=urgency,
                    expected_patient_impact=impact,
                    rationale=f"{index.region} scores {index.score}/100 and lacks {gap}; {critical_penalty} high-risk validation signals need attention.",
                    citations=sorted({f.source_row_id or f.id for f in region_facilities}),
                )
            )
    return sorted(targets, key=lambda target: target.urgency_score, reverse=True)


def simulate_specialist_deployment(
    facilities: list[FacilityModel],
    anomalies: list[AnomalyReport],
    specialty: str,
    count: int,
) -> dict:
    """Estimate Care Access Index lift from adding N specialists to underserved regions."""

    before = compute_indices_by_region(facilities, anomalies)
    targets = rank_intervention_targets(facilities, anomalies, specialty)[: max(count, 1)]
    lift_by_region = {index.region: 0.0 for index in before}
    for target in targets[:count]:
        lift_by_region[target.region] += 6.5
    after = []
    for index in before:
        projected_score = min(100.0, round(index.score + lift_by_region.get(index.region, 0.0), 2))
        after.append(
            {
                "region": index.region,
                "before_score": index.score,
                "after_score": projected_score,
                "delta": round(projected_score - index.score, 2),
                "priority_before": index.recommendation_priority,
            }
        )
    return {
        "specialty": specialty,
        "specialists_added": count,
        "top_targets": [target.__dict__ for target in targets[:count]],
        "score_projection": sorted(after, key=lambda item: item["delta"], reverse=True),
        "estimated_total_patient_impact": sum(target.expected_patient_impact for target in targets[:count]),
    }


def verification_queue(facilities: list[FacilityModel], anomalies: list[AnomalyReport]) -> list[dict]:
    """Build a field-verification queue ranked by patient safety risk."""

    severity_points = {"critical": 100, "high": 70, "medium": 35, "low": 10}
    rows: list[dict] = []
    for facility in facilities:
        reports = [report for report in anomalies if report.facility_id == facility.id]
        if not reports:
            continue
        score = sum(severity_points[report.severity] for report in reports) + min(facility.capacity, 100)
        rows.append(
            {
                "facility_id": facility.id,
                "facility": facility.name,
                "region": facility.address_stateOrRegion,
                "risk_score": score,
                "anomaly_count": len(reports),
                "top_evidence": reports[0].evidence,
                "recommended_action": "Call facility administrator and request field verification before routing patients.",
                "citations": sorted({report.source_row_id for report in reports}),
            }
        )
    return sorted(rows, key=lambda row: row["risk_score"], reverse=True)

