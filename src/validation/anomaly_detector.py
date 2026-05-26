"""Rule-based and semantic anomaly detection for facility records."""

from __future__ import annotations

from datetime import datetime

from src.models.analysis import AnomalyReport
from src.models.facility import FacilityModel
from src.validation.rules import SEVERITY_WEIGHTS


def _contains(items: list[str], *needles: str) -> bool:
    text = " | ".join(items).lower()
    return any(needle.lower() in text for needle in needles)


def detect_anomalies(facility: FacilityModel, source_text: str = "") -> list[AnomalyReport]:
    """Detect contradictory or implausible medical claims."""

    source_row_id = facility.source_row_id or facility.id
    reports: list[AnomalyReport] = []

    def add(kind: str, description: str, evidence: str, severity: str, confidence: float = 1.0) -> None:
        reports.append(
            AnomalyReport(
                facility_id=facility.id,
                anomaly_type=kind,
                description=description,
                evidence=evidence,
                severity=severity,
                source_row_id=source_row_id,
                confidence=confidence,
            )
        )

    if _contains(facility.capability, "ICU") and not _contains(facility.equipment, "backup power", "generator"):
        add("ICU_WITHOUT_POWER", "ICU capability is claimed without backup power evidence.", "capability=ICU; equipment lacks generator", "high")
    if "generalSurgery" in facility.specialties and facility.numberDoctors == 0:
        add("SURGERY_NO_DOCTORS", "Surgical specialty listed but no doctors are recorded.", "generalSurgery with numberDoctors=0", "critical")
    if len(facility.capability) > 15 and len(facility.equipment) < 3:
        add("CAPABILITY_INFLATION", "Many capabilities are claimed with little supporting equipment.", f"{len(facility.capability)} capabilities; {len(facility.equipment)} equipment", "medium")
    if _contains(facility.equipment, "MRI", "CT") and facility.capacity < 10:
        add("PHANTOM_EQUIPMENT", "Advanced imaging listed for a very small facility.", "MRI/CT with capacity < 10", "high")
    if "cardiology" in facility.specialties and not _contains(facility.equipment, "ECG", "echo"):
        add("SPECIALTY_WITHOUT_EQUIPMENT", "Cardiology appears without ECG or echo equipment.", "cardiology with no ECG/echo", "medium")
    if facility.capacity == 0 and facility.procedure:
        add("ZERO_CAPACITY_OPERATING", "Procedures listed for a facility with zero capacity.", "capacity=0 with procedures", "high")
    if facility.yearEstablished is not None and (facility.yearEstablished > datetime.now().year or facility.yearEstablished < 1800):
        add("YEAR_ANOMALY", "Establishment year is outside the valid range.", str(facility.yearEstablished), "critical")
    if _contains(facility.capability, "NICU") and "pediatrics" not in facility.specialties:
        add("NICU_WITHOUT_PEDIATRICS", "NICU capability appears without pediatrics specialty.", "NICU with no pediatrics", "medium")
    if _contains(facility.capability, "Level I trauma") and facility.capacity < 50:
        add("TRAUMA_LEVEL_MISMATCH", "Level I trauma claim is implausible with fewer than 50 beds.", "Level I trauma with capacity < 50", "high")
    if source_text and "24/7" in source_text and facility.numberDoctors == 0:
        add("SEMANTIC_CONTRADICTION", "Round-the-clock service is claimed with no doctors recorded.", "24/7 claim and numberDoctors=0", "medium", 0.82)
    return reports


def aggregate_risk_score(reports: list[AnomalyReport]) -> int:
    """Aggregate anomaly reports into a 0-100 risk score."""

    return min(100, sum(SEVERITY_WEIGHTS[report.severity] for report in reports))

