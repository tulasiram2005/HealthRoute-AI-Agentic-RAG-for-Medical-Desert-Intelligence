"""Strict Pydantic models for Virtue Foundation facility data."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


E164_RE = re.compile(r"^\+[1-9]\d{1,14}$")
ISO_ALPHA2 = {
    "GH", "US", "GB", "CA", "FR", "DE", "NG", "CI", "TG", "BF", "KE", "ZA",
    "IN", "AU", "BR", "MX", "ES", "IT", "NL", "SE", "NO", "DK", "FI",
}


class FacilityType(str, Enum):
    """Allowed Virtue facility types."""

    hospital = "hospital"
    pharmacy = "pharmacy"
    doctor = "doctor"
    clinic = "clinic"
    dentist = "dentist"


class OperatorType(str, Enum):
    """Allowed operator ownership models."""

    public = "public"
    private = "private"


class AffiliationType(str, Enum):
    """Allowed facility affiliation types."""

    faith_tradition = "faith-tradition"
    philanthropy_legacy = "philanthropy-legacy"
    community = "community"
    academic = "academic"
    government = "government"


class Specialty(str, Enum):
    """Case-sensitive specialty taxonomy from the challenge prompt."""

    internalMedicine = "internalMedicine"
    familyMedicine = "familyMedicine"
    pediatrics = "pediatrics"
    cardiology = "cardiology"
    generalSurgery = "generalSurgery"
    emergencyMedicine = "emergencyMedicine"
    gynecologyAndObstetrics = "gynecologyAndObstetrics"
    orthopedicSurgery = "orthopedicSurgery"
    dentistry = "dentistry"
    ophthalmology = "ophthalmology"


class FacilityModel(BaseModel):
    """Unified facility record with strict schema validation."""

    model_config = ConfigDict(strict=True, use_enum_values=True)

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    phone_numbers: list[str] = Field(default_factory=list)
    email: Optional[str] = None
    websites: list[str] = Field(default_factory=list)
    officialWebsite: Optional[str] = None
    facebookLink: Optional[str] = None
    twitterLink: Optional[str] = None
    linkedinLink: Optional[str] = None
    instagramLink: Optional[str] = None
    logo: Optional[str] = None
    yearEstablished: Optional[int] = None
    acceptsVolunteers: bool = False
    address_line1: Optional[str] = None
    address_line2: Optional[str] = None
    address_line3: Optional[str] = None
    address_city: Optional[str] = None
    address_stateOrRegion: str
    address_district: Optional[str] = None
    address_zipOrPostcode: Optional[str] = None
    address_country: str = "Ghana"
    address_countryCode: str = "GH"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    facilityTypeId: FacilityType
    operatorTypeId: OperatorType
    affiliationTypeIds: list[AffiliationType] = Field(default_factory=list)
    description: Optional[str] = None
    area: Optional[str] = None
    numberDoctors: int = Field(default=0, ge=0)
    capacity: int = Field(default=0, ge=0)
    specialties: list[Specialty] = Field(default_factory=list)
    procedure: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    capability: list[str] = Field(default_factory=list)
    source_row_id: Optional[str] = None

    @field_validator("phone_numbers")
    @classmethod
    def validate_phone_numbers(cls, values: list[str]) -> list[str]:
        for value in values:
            if not E164_RE.match(value):
                raise ValueError(f"phone number is not E164: {value}")
        return values

    @field_validator("facilityTypeId", mode="before")
    @classmethod
    def normalize_facility_type(cls, value):
        if isinstance(value, FacilityType):
            return value
        return FacilityType(value)

    @field_validator("operatorTypeId", mode="before")
    @classmethod
    def normalize_operator_type(cls, value):
        if isinstance(value, OperatorType):
            return value
        return OperatorType(value)

    @field_validator("affiliationTypeIds", mode="before")
    @classmethod
    def normalize_affiliations(cls, values):
        return [value if isinstance(value, AffiliationType) else AffiliationType(value) for value in (values or [])]

    @field_validator("specialties", mode="before")
    @classmethod
    def normalize_specialties(cls, values):
        return [value if isinstance(value, Specialty) else Specialty(value) for value in (values or [])]

    @field_validator("address_countryCode")
    @classmethod
    def validate_country_code(cls, value: str) -> str:
        if value.upper() not in ISO_ALPHA2:
            raise ValueError("address_countryCode must be ISO alpha-2")
        return value.upper()

    @field_validator("yearEstablished")
    @classmethod
    def validate_year(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return value
        current_year = datetime.now().year
        if value < 1800 or value > current_year:
            raise ValueError("yearEstablished must be between 1800 and current year")
        return value


class DoctorProfile(BaseModel):
    """Profile used by the doctor-facility matching agent."""

    model_config = ConfigDict(strict=True, use_enum_values=True)

    specialty: Specialty
    sub_specialty: Optional[str] = None
    years_experience: int = Field(..., ge=0, le=70)
    equipment_familiar_with: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    available_weeks: int = Field(..., ge=1, le=52)
    preferred_regions: Optional[list[str]] = None

    @field_validator("specialty", mode="before")
    @classmethod
    def normalize_specialty(cls, value):
        if isinstance(value, Specialty):
            return value
        return Specialty(value)


class DoctorFacilityMatch(BaseModel):
    """Top match between a visiting doctor and a facility."""

    model_config = ConfigDict(strict=True)

    doctor_specialty: str
    matched_facility_id: str
    match_score: float = Field(..., ge=0.0, le=100.0)
    rationale: str
    citations: list[str] = Field(default_factory=list)
    need_rationale: str = ""
    infra_rationale: str = ""
    estimated_patient_impact: int = Field(default=0, ge=0)


if __name__ == "__main__":
    facility = FacilityModel(
        id="fac_001",
        name="Korle Bu Teaching Hospital",
        phone_numbers=["+233302665401"],
        address_stateOrRegion="Greater Accra",
        address_district="Accra Metropolitan",
        facilityTypeId=FacilityType.hospital,
        operatorTypeId=OperatorType.public,
        affiliationTypeIds=[AffiliationType.academic, AffiliationType.government],
        numberDoctors=120,
        capacity=2000,
        specialties=[Specialty.cardiology, Specialty.generalSurgery],
        equipment=["ECG", "echo", "backup generator"],
        capability=["ICU", "cardiac surgery"],
    )
    print(facility.model_dump())
