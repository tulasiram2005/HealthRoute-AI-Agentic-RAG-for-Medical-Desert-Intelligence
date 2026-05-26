"""Extraction result model."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .facility import FacilityModel


class ExtractionResult(BaseModel):
    """Structured facts extracted from a single source row."""

    model_config = ConfigDict(strict=True)

    source_row_id: str
    extracted_facility: FacilityModel
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_step_id: str = "extract"

    @field_validator("confidence_scores")
    @classmethod
    def validate_confidence(cls, values: dict[str, float]) -> dict[str, float]:
        for key, value in values.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"confidence for {key} must be 0.0-1.0")
        return values

