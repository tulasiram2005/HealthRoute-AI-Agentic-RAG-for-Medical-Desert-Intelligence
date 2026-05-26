"""Deterministic grounded query engine for hackathon must-have questions."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import asin, cos, radians, sin, sqrt
from typing import Any

from src.models.analysis import AnomalyReport
from src.models.facility import FacilityModel, Specialty
from src.planning.desert_scorer import compute_indices_by_region
from src.planning.matcher import match_doctor_to_facilities


def _source(facility: FacilityModel) -> str:
    return facility.source_row_id or facility.id


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def answer_query(query: str, facilities: list[FacilityModel], anomalies: list[AnomalyReport]) -> dict[str, Any]:
    """Answer common NGO planner questions with row-level citations."""

    q = query.lower()
    try:
        from src.rag.rag_query import rag_ground_query

        rag = rag_ground_query(query, top_k=5)
    except Exception:
        rag = {"source_row_ids": [], "scores": [], "results": [], "context": ""}
    sources: list[str] = []
    confidence = 0.86
    payload: Any = None

    if "how many" in q and "icu" in q:
        matches = [f for f in facilities if "icu" in " ".join(f.capability).lower()]
        sources = [_source(f) for f in matches]
        answer = f"{len(matches)} facilities in the sample report ICU capability."
        payload = [{"facility": f.name, "region": f.address_stateOrRegion, "source_row_id": _source(f)} for f in matches]

    elif "no emergency medicine" in q:
        regions = sorted({f.address_stateOrRegion for f in facilities})
        covered = {f.address_stateOrRegion for f in facilities if "emergencyMedicine" in f.specialties}
        missing = [region for region in regions if region not in covered]
        answer = f"Regions with no recorded emergency medicine specialists: {', '.join(missing) or 'none in this sample'}."
        sources = [_source(f) for f in facilities if f.address_stateOrRegion in missing]
        payload = missing

    elif "care access index" in q:
        indices = compute_indices_by_region(facilities, anomalies)
        matched = next((i for i in indices if i.region.lower() in q), indices[0])
        answer = f"{matched.region} has a Care Access Index of {matched.score}/100 with priority {matched.recommendation_priority}."
        sources = [_source(f) for f in facilities if f.address_stateOrRegion == matched.region]
        payload = matched.model_dump(mode="json")

    elif "mri" in q:
        matches = [f for f in facilities if "mri" in " ".join(f.equipment).lower()]
        answer = f"{len(matches)} facilities list MRI equipment."
        sources = [_source(f) for f in matches]
        payload = [{"facility": f.name, "equipment": f.equipment, "source_row_id": _source(f)} for f in matches]

    elif "most suspicious" in q:
        counts = Counter(report.facility_id for report in anomalies)
        if counts:
            facility_id, _ = counts.most_common(1)[0]
            facility = next(f for f in facilities if f.id == facility_id)
            evidence = [r.evidence for r in anomalies if r.facility_id == facility_id]
            answer = f"{facility.name} has the most suspicious capability claims based on {len(evidence)} anomaly signals."
            sources = [_source(facility)]
            payload = {"facility": facility.name, "evidence": evidence}
        else:
            answer = "No suspicious capability claims were detected in the current sample."
            payload = {}

    elif "underrepresented" in q:
        northern = [f for f in facilities if "north" in f.address_stateOrRegion.lower() or f.address_stateOrRegion in {"Northern", "Upper East", "Upper West"}]
        covered = Counter(s for f in northern for s in f.specialties)
        ranking = [s.value for s in Specialty if covered[s.value] == 0]
        answer = f"Most underrepresented specialties in Northern Ghana: {', '.join(ranking[:5])}."
        sources = [_source(f) for f in northern]
        payload = ranking

    elif "within 100km of accra" in q or "within 100 km of accra" in q:
        accra = (5.6037, -0.1870)
        matches = []
        for facility in facilities:
            if facility.latitude is None or facility.longitude is None or "pediatrics" not in facility.specialties:
                continue
            distance = _distance_km(accra[0], accra[1], facility.latitude, facility.longitude)
            if distance <= 100:
                matches.append({"facility": facility.name, "distance_km": round(distance, 1), "source_row_id": _source(facility)})
        answer = f"{len(matches)} pediatric facilities are within 100 km of Accra in the sample."
        sources = [m["source_row_id"] for m in matches]
        payload = matches

    elif "critical anomalies" in q:
        critical = [r for r in anomalies if r.severity == "critical"]
        answer = f"{len(critical)} critical anomalies are currently flagged."
        sources = [r.source_row_id for r in critical]
        payload = [r.model_dump(mode="json") for r in critical]

    elif "best facility" in q and ("cardiac" in q or "cardiology" in q):
        candidates = [f for f in facilities if "cardiology" in f.specialties or "cardiac" in " ".join(f.capability).lower()]
        best = max(candidates or facilities, key=lambda f: f.numberDoctors + f.capacity / 50 + len(f.equipment) * 2)
        answer = f"{best.name} is the strongest cardiac-care candidate because it combines specialist coverage, capacity, and relevant equipment."
        sources = [_source(best)]
        payload = best.model_dump(mode="json")

    elif "deployment plan" in q or "orthopedic" in q:
        from src.models.facility import DoctorProfile

        profile = DoctorProfile(specialty="orthopedicSurgery", years_experience=8, equipment_familiar_with=["x-ray", "operating theatre"], languages=["English"], available_weeks=12)
        matches = match_doctor_to_facilities(profile, facilities, anomalies)
        answer = "Recommended 3-month deployment: place orthopedic surgeons at the top matched facilities, then rotate follow-up clinics into adjacent underserved districts."
        sources = [source for match in matches for source in match.citations]
        payload = [match.model_dump(mode="json") for match in matches]

    else:
        tokens = set(q.split())
        scored = []
        for facility in facilities:
            text = " ".join([facility.name, facility.address_stateOrRegion, *facility.specialties, *facility.equipment, *facility.capability]).lower()
            overlap = len(tokens & set(text.split()))
            if overlap:
                scored.append((overlap, facility))
        matches = [facility for _, facility in sorted(scored, key=lambda item: item[0], reverse=True)[:5]]
        answer = f"Found {len(matches)} relevant facilities for this question."
        sources = [_source(f) for f in matches]
        payload = [{"facility": f.name, "region": f.address_stateOrRegion, "source_row_id": _source(f)} for f in matches]
        confidence = 0.72

    citations = {"query_engine": sorted(set(sources))}
    if rag.get("source_row_ids"):
        citations["rag_retrieval"] = rag["source_row_ids"]
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "sources": sorted(set(sources)),
        "data": payload,
        "rag": {
            "context": rag.get("context", ""),
            "source_row_ids": rag.get("source_row_ids", []),
            "scores": rag.get("scores", []),
            "results": rag.get("results", []),
        },
    }


def regional_specialty_counts(facilities: list[FacilityModel]) -> dict[str, dict[str, int]]:
    """Count specialty coverage by region."""

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {specialty.value: 0 for specialty in Specialty})
    for facility in facilities:
        for specialty in facility.specialties:
            counts[facility.address_stateOrRegion][specialty] += max(1, facility.numberDoctors)
    return dict(counts)
