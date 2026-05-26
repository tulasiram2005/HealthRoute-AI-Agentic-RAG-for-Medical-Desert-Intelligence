"""Analysis output models for anomalies, access scoring, and citations."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Severity(str, Enum):
    """Anomaly severity taxonomy."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class AnomalyReport(BaseModel):
    """Evidence-backed validation anomaly."""

    model_config = ConfigDict(strict=True, use_enum_values=True)

    facility_id: str
    anomaly_type: str
    description: str
    evidence: str
    severity: Severity
    source_row_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value):
        if isinstance(value, Severity):
            return value
        return Severity(value)


class CareAccessIndex(BaseModel):
    """Regional medical desert score."""

    model_config = ConfigDict(strict=True)

    region: str
    district: str
    score: float = Field(..., ge=0.0, le=100.0)
    specialty_gaps: list[str] = Field(default_factory=list)
    flagged_facilities: list[str] = Field(default_factory=list)
    recommendation_priority: int = Field(..., ge=1, le=5)
