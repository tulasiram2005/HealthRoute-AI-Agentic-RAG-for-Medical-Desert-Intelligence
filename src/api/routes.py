"""FastAPI routes for the Virtue Foundation IDP Agent."""

from __future__ import annotations

import csv
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

from src.agents.graph import run_agent
from src.agents.crew import run_crew_analysis
from src.extraction.parser import parse_facility_text
from src.models.facility import DoctorProfile
from src.planning.query_engine import answer_query
from src.planning.desert_scorer import compute_indices_by_region
from src.planning.matcher import match_doctor_to_facilities
from src.planning.scenario import rank_intervention_targets, simulate_specialist_deployment, verification_queue
from src.reporting.audit_report import build_audit_html
from src.validation.anomaly_detector import detect_anomalies

router = APIRouter()
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ghana_facilities.csv"


class QueryRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    query: str
    filters: dict | None = None


class ScenarioRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    specialty: str
    count: int = 1


class JourneyRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    condition: str
    patient_region: str


def load_rows() -> list[dict]:
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_facilities():
    return [parse_facility_text(row).extracted_facility for row in load_rows()]


@router.post("/api/v1/query")
async def query(request: QueryRequest) -> dict:
    facilities = load_facilities()
    reports = [report for facility in facilities for report in detect_anomalies(facility)]
    result = answer_query(request.query, facilities, reports)
    if result["sources"]:
        selected = next((row for row in load_rows() if (row.get("source_row_id") or row.get("id")) == result["sources"][0]), load_rows()[0])
        state = await run_agent(selected)
        result["citations"].update(state.get("citations", {}))
    result["mlflow_run_url"] = ""
    return result


@router.post("/api/v1/analyze-facility")
async def analyze_facility(payload: dict) -> dict:
    result = parse_facility_text(payload)
    anomalies = detect_anomalies(result.extracted_facility, payload.get("facility_text", ""))
    return {"extraction": result.model_dump(mode="json"), "anomalies": [a.model_dump(mode="json") for a in anomalies]}


@router.post("/api/v1/crew-analyze")
async def crew_analyze(payload: dict) -> dict:
    rows = load_rows()
    source_row_id = payload.get("source_row_id")
    row = next((item for item in rows if item.get("source_row_id") == source_row_id or item.get("id") == source_row_id), None)
    return run_crew_analysis(row or payload)


@router.post("/api/v1/patient-journey")
async def patient_journey(request: JourneyRequest) -> dict:
    from src.planning.patient_journey import simulate_patient_journey

    return simulate_patient_journey(request.condition, request.patient_region, load_facilities()).to_dict()


@router.get("/api/v1/care-index/{region}")
async def care_index(region: str) -> dict:
    facilities = load_facilities()
    indices = compute_indices_by_region(facilities)
    for index in indices:
        if index.region.lower() == region.lower():
            return index.model_dump(mode="json")
    raise HTTPException(status_code=404, detail="Unknown region")


@router.get("/api/v1/medical-deserts")
async def medical_deserts() -> list[dict]:
    facilities = load_facilities()
    return [item.model_dump(mode="json") for item in compute_indices_by_region(facilities) if item.score < 40]


@router.get("/api/v1/intervention-targets")
async def intervention_targets(specialty: str | None = None) -> list[dict]:
    facilities = load_facilities()
    reports = [report for facility in facilities for report in detect_anomalies(facility)]
    return [target.__dict__ for target in rank_intervention_targets(facilities, reports, specialty)[:10]]


@router.post("/api/v1/scenario")
async def scenario(request: ScenarioRequest) -> dict:
    facilities = load_facilities()
    reports = [report for facility in facilities for report in detect_anomalies(facility)]
    return simulate_specialist_deployment(facilities, reports, request.specialty, request.count)


@router.get("/api/v1/verification-queue")
async def field_verification_queue() -> list[dict]:
    facilities = load_facilities()
    reports = [report for facility in facilities for report in detect_anomalies(facility)]
    return verification_queue(facilities, reports)


@router.post("/api/v1/match-doctor")
async def match_doctor(profile: DoctorProfile) -> list[dict]:
    facilities = load_facilities()
    anomalies = [report for facility in facilities for report in detect_anomalies(facility)]
    return [match.model_dump(mode="json") for match in match_doctor_to_facilities(profile, facilities, anomalies)]


@router.get("/api/v1/anomalies")
async def anomalies(severity: str | None = None) -> list[dict]:
    reports = [report for facility in load_facilities() for report in detect_anomalies(facility)]
    if severity:
        reports = [report for report in reports if report.severity == severity]
    return [report.model_dump(mode="json") for report in reports]


@router.get("/api/v1/map-data")
async def map_data() -> dict:
    features = []
    for facility in load_facilities():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [facility.longitude or 0, facility.latitude or 0]},
                "properties": facility.model_dump(mode="json"),
            }
        )
    return {"type": "FeatureCollection", "features": features}


@router.get("/api/v1/export/report")
async def export_report() -> dict:
    facilities = load_facilities()
    reports = [report for facility in facilities for report in detect_anomalies(facility)]
    indices = compute_indices_by_region(facilities, reports)
    targets = rank_intervention_targets(facilities, reports)
    path = build_audit_html(facilities, reports, indices, targets)
    return {"message": "Audit report generated", "path": path}
