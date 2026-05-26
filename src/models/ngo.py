"""NGO and partner organization models."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NGOModel(BaseModel):
    """Strict NGO schema aligned with Virtue Foundation fields."""

    model_config = ConfigDict(strict=True)

    id: str
    name: str
    phone_numbers: list[str] = Field(default_factory=list)
    email: Optional[str] = None
    websites: list[str] = Field(default_factory=list)
    officialWebsite: Optional[str] = None
    description: Optional[str] = None
    address_countryCode: str = "GH"
    focus_areas: list[str] = Field(default_factory=list)


class OtherOrganizationModel(NGOModel):
    """Other partner organization that shares the NGO contact shape."""

    organization_type: str = "other"

