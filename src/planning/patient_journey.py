"""Patient journey simulation for routing impact analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import asin, cos, radians, sin, sqrt

from src.models.facility import FacilityModel

CONDITION_SPECIALTY_MAP = {
    "chest pain": ["cardiology", "emergencyMedicine"],
    "heart attack": ["cardiology", "emergencyMedicine"],
    "difficult labour": ["gynecologyAndObstetrics"],
    "broken leg": ["orthopedicSurgery", "emergencyMedicine"],
    "eye injury": ["ophthalmology", "emergencyMedicine"],
    "child fever": ["pediatrics", "familyMedicine"],
    "tooth pain": ["dentistry"],
    "general illness": ["familyMedicine", "internalMedicine"],
    "trauma": ["emergencyMedicine", "generalSurgery"],
    "newborn emergency": ["pediatrics", "gynecologyAndObstetrics"],
}

REGION_COORDS = {
    "Greater Accra": (5.60, -0.18),
    "Ashanti": (6.70, -1.63),
    "Northern": (9.40, -0.85),
    "Upper East": (10.78, -0.85),
    "Upper West": (10.06, -2.51),
    "Western": (4.95, -2.25),
    "Eastern": (6.10, -0.26),
    "Central": (5.13, -1.28),
    "Volta": (6.61, 0.47),
    "Bono": (7.34, -2.33),
    "Western North": (6.20, -2.49),
    "Ahafo": (6.92, -2.49),
    "Bono East": (7.59, -1.94),
    "Oti": (7.80, 0.30),
    "North East": (10.52, -0.37),
    "Savannah": (9.08, -1.82),
}


@dataclass(frozen=True)
class PatientJourneyResult:
    condition: str
    patient_region: str
    current_route: list[dict]
    optimized_route: list[dict]
    time_saved_minutes: int
    lives_at_risk: str
    recommendation: str
    citations: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_patient_journey(condition: str, patient_region: str, facilities: list[FacilityModel]) -> PatientJourneyResult:
    """Compare nearest-facility routing with capability-aware routing."""

    needed = CONDITION_SPECIALTY_MAP.get(condition.lower(), ["emergencyMedicine"])
    patient_lat, patient_lon = REGION_COORDS.get(patient_region, (7.95, -1.02))
    current = sorted(facilities, key=lambda f: _facility_distance(patient_lat, patient_lon, f))[:3]
    capable = [facility for facility in facilities if any(specialty in facility.specialties for specialty in needed)]
    optimized = sorted(capable, key=lambda f: _facility_distance(patient_lat, patient_lon, f))[:3]
    current_route = _to_route(current, needed, patient_lat, patient_lon)
    optimized_route = _to_route(optimized, needed, patient_lat, patient_lon) if optimized else current_route
    current_time = (current_route[0]["wait_minutes"] if current_route else 999) + (20 if current_route and not current_route[0]["has_specialty"] else 0)
    optimized_time = optimized_route[0]["wait_minutes"] if optimized_route else 999
    time_saved = max(0, current_time - optimized_time)
    high_severity = condition.lower() in {"heart attack", "trauma", "newborn emergency"}
    citations = [route["source_row_id"] for route in (optimized_route or current_route)[:3]]
    return PatientJourneyResult(
        condition=condition,
        patient_region=patient_region,
        current_route=current_route,
        optimized_route=optimized_route,
        time_saved_minutes=time_saved,
        lives_at_risk="HIGH" if high_severity else "MEDIUM",
        recommendation=f"Deploy {needed[0]} specialist to {patient_region} to save about {time_saved} minutes per {condition} patient.",
        citations=citations,
    )


def _to_route(facilities: list[FacilityModel], needed: list[str], patient_lat: float, patient_lon: float) -> list[dict]:
    route = []
    for facility in facilities:
        distance = _facility_distance(patient_lat, patient_lon, facility)
        route.append(
            {
                "facility": facility.name,
                "region": facility.address_stateOrRegion,
                "distance_km": round(distance, 1),
                "has_specialty": any(specialty in facility.specialties for specialty in needed),
                "capacity": facility.capacity,
                "wait_minutes": 30 + int(distance * 2),
                "source_row_id": facility.source_row_id or facility.id,
            }
        )
    return route


def _facility_distance(patient_lat: float, patient_lon: float, facility: FacilityModel) -> float:
    return _distance_km(patient_lat, patient_lon, facility.latitude or 0.0, facility.longitude or 0.0)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(a))
