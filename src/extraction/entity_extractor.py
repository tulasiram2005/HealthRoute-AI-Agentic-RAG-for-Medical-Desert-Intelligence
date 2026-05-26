"""Rule-assisted entity extraction for medical notes."""

from __future__ import annotations

from src.models.facility import Specialty


SPECIALTY_KEYWORDS = {
    Specialty.internalMedicine.value: ["internal medicine", "physician", "medicine"],
    Specialty.familyMedicine.value: ["family medicine", "primary care"],
    Specialty.pediatrics.value: ["pediatric", "paediatric", "children", "pediatrie"],
    Specialty.cardiology.value: ["cardiology", "cardiac", "heart", "echo", "ecg"],
    Specialty.generalSurgery.value: ["surgery", "surgical", "operating theatre", "theatre"],
    Specialty.emergencyMedicine.value: ["emergency", "trauma", "er", "casualty"],
    Specialty.gynecologyAndObstetrics.value: ["obstetric", "gynecology", "maternity", "labour"],
    Specialty.orthopedicSurgery.value: ["orthopedic", "orthopaedic", "fracture", "bone"],
    Specialty.dentistry.value: ["dental", "dentist", "oral"],
    Specialty.ophthalmology.value: ["ophthalmology", "eye", "cataract"],
}

EQUIPMENT_KEYWORDS = [
    "MRI", "CT", "ECG", "echo", "ultrasound", "x-ray", "ventilator", "generator",
    "backup power", "incubator", "oxygen", "ambulance", "lab", "operating theatre",
]

CAPABILITY_KEYWORDS = [
    "ICU", "NICU", "surgery", "cardiac surgery", "trauma", "maternity", "dialysis",
    "emergency", "orthopedic", "pediatric", "outpatient", "inpatient",
]


def extract_specialties(text: str) -> list[str]:
    """Return exact-match specialties inferred from free text."""

    lowered = text.lower()
    found: list[str] = []
    for specialty, keywords in SPECIALTY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            found.append(specialty)
    return found


def extract_terms(text: str, vocabulary: list[str]) -> list[str]:
    """Extract vocabulary terms using case-insensitive containment."""

    lowered = text.lower()
    return [term for term in vocabulary if term.lower() in lowered]

