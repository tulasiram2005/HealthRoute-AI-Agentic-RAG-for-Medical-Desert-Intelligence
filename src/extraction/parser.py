"""Free-form facility text parser."""

from __future__ import annotations

from datetime import datetime

from src.extraction.confidence import confidence_for_terms, field_confidence
from src.extraction.entity_extractor import (
    CAPABILITY_KEYWORDS,
    EQUIPMENT_KEYWORDS,
    extract_specialties,
    extract_terms,
)
from src.models.extraction import ExtractionResult
from src.models.facility import FacilityModel


def _split_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


def _int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float_or_none(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def parse_facility_text(row: dict) -> ExtractionResult:
    """Parse a source row into a strict FacilityModel with confidence scores."""

    source_row_id = str(row.get("source_row_id") or row.get("id") or "row_unknown")
    text = " ".join(
        str(row.get(field, "") or "")
        for field in ["description", "procedure", "equipment", "capability", "notes"]
    )
    specialties = row.get("specialties") or extract_specialties(text)
    specialties = _split_list(specialties) if specialties else extract_specialties(text)
    equipment = row.get("equipment")
    equipment_list = _split_list(equipment) if equipment else extract_terms(text, EQUIPMENT_KEYWORDS)
    capability = row.get("capability")
    capability_list = _split_list(capability) if capability else extract_terms(text, CAPABILITY_KEYWORDS)
    procedure = row.get("procedure")
    procedure_list = _split_list(procedure)

    facility = FacilityModel(
        id=str(row.get("id") or source_row_id),
        name=str(row.get("name") or "Unknown Facility"),
        phone_numbers=_split_list(row.get("phone_numbers")),
        email=row.get("email"),
        websites=_split_list(row.get("websites")),
        officialWebsite=row.get("officialWebsite"),
        yearEstablished=_int(row.get("yearEstablished")) if row.get("yearEstablished") else None,
        acceptsVolunteers=_bool(row.get("acceptsVolunteers", False)),
        address_line1=row.get("address_line1"),
        address_city=row.get("address_city"),
        address_stateOrRegion=str(row.get("address_stateOrRegion") or row.get("region") or "Unknown"),
        address_district=row.get("address_district") or row.get("district"),
        address_countryCode=str(row.get("address_countryCode") or "GH"),
        latitude=_float_or_none(row.get("latitude")),
        longitude=_float_or_none(row.get("longitude")),
        facilityTypeId=row.get("facilityTypeId") or "clinic",
        operatorTypeId=row.get("operatorTypeId") or "public",
        affiliationTypeIds=_split_list(row.get("affiliationTypeIds")),
        description=row.get("description"),
        area=row.get("area"),
        numberDoctors=_int(row.get("numberDoctors")),
        capacity=_int(row.get("capacity")),
        specialties=specialties,
        procedure=procedure_list,
        equipment=equipment_list,
        capability=capability_list,
        source_row_id=source_row_id,
    )
    matched = list(specialties) + equipment_list + capability_list
    confidence = confidence_for_terms(text, matched)
    specialty_confidence = max(confidence_for_terms(text, list(specialties)), field_confidence(row, "specialties", specialties))
    equipment_confidence = max(confidence_for_terms(text, equipment_list), field_confidence(row, "equipment", equipment_list))
    capability_confidence = max(confidence_for_terms(text, capability_list), field_confidence(row, "capability", capability_list))
    return ExtractionResult(
        source_row_id=source_row_id,
        extracted_facility=facility,
        confidence_scores={
            "specialties": specialty_confidence,
            "equipment": equipment_confidence,
            "capability": capability_confidence,
            "overall": max(confidence, round((specialty_confidence + equipment_confidence + capability_confidence) / 3, 3)),
        },
        extraction_timestamp=datetime.utcnow(),
        agent_step_id="extract",
    )
