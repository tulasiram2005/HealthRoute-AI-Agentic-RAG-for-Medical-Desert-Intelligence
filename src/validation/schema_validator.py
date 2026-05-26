"""Strict schema validation helpers."""

from __future__ import annotations

from pydantic import ValidationError

from src.models.facility import FacilityModel


def validate_facility_payload(payload: dict) -> tuple[FacilityModel | None, list[str]]:
    """Validate a facility payload and return model plus human-readable errors."""

    try:
        return FacilityModel.model_validate(payload), []
    except ValidationError as exc:
        return None, [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in exc.errors()]

