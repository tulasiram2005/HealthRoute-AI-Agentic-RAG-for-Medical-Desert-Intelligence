"""SQLite-backed Databricks Genie-style Text2SQL adapter."""

from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE facility_capabilities (
  source_row_id TEXT,
  id TEXT,
  name TEXT,
  address_stateOrRegion TEXT,
  address_district TEXT,
  facilityTypeId TEXT,
  operatorTypeId TEXT,
  numberDoctors INTEGER,
  capacity INTEGER,
  specialties TEXT,
  procedure TEXT,
  equipment TEXT,
  capability TEXT,
  latitude REAL,
  longitude REAL,
  acceptsVolunteers BOOLEAN,
  yearEstablished INTEGER
);
"""


@dataclass(frozen=True)
class GeniePlan:
    """Text2SQL execution result used by the UI and API."""

    question: str
    sql: str
    explanation: str
    used_native_genie: bool = False
    results: list[dict[str, Any]] | None = None
    row_count: int = 0
    source_row_ids: list[str] | None = None


class SQLiteGenieAdapter:
    """In-memory SQLite query router for Ghana facility analytics."""

    def __init__(self, data_path: str | Path | None = None) -> None:
        self.data_path = Path(data_path) if data_path else Path(__file__).resolve().parents[2] / "data" / "ghana_facilities.csv"
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self._load()

    def _load(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        with self.data_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            self.conn.execute(
                """
                INSERT INTO facility_capabilities VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("source_row_id"),
                    row.get("id"),
                    row.get("name"),
                    row.get("address_stateOrRegion"),
                    row.get("address_district"),
                    row.get("facilityTypeId"),
                    row.get("operatorTypeId"),
                    _int(row.get("numberDoctors")),
                    _int(row.get("capacity")),
                    row.get("specialties"),
                    row.get("procedure"),
                    row.get("equipment"),
                    row.get("capability"),
                    _float(row.get("latitude")),
                    _float(row.get("longitude")),
                    str(row.get("acceptsVolunteers", "")).lower() == "true",
                    _int(row.get("yearEstablished")),
                ),
            )
        self.conn.commit()

    def plan(self, question: str) -> GeniePlan:
        sql, used_llm = self._llm_sql(question)
        if not sql:
            sql = self._deterministic_sql(question)
        try:
            rows = self._execute(sql)
        except sqlite3.Error:
            sql = self._deterministic_sql(question)
            rows = self._execute(sql)
            used_llm = False
        source_row_ids = [str(row.get("source_row_id")) for row in rows if row.get("source_row_id")]
        return GeniePlan(
            question=question,
            sql=sql,
            explanation="SQLite Genie-style Text2SQL plan executed against the Ghana facility table.",
            used_native_genie=used_llm,
            results=rows,
            row_count=len(rows),
            source_row_ids=source_row_ids,
        )

    def _execute(self, sql: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(sql).fetchall()
        return [dict(row) for row in rows]

    def _llm_sql(self, question: str) -> tuple[str | None, bool]:
        try:
            from src.integrations.databricks_llm import DatabricksLLMClient

            client = DatabricksLLMClient()
            if client.token is None:
                return None, False
            response = client.chat(
                [
                    {
                        "role": "system",
                        "content": f"You are a SQL expert. Given this schema, generate a SQLite-compatible SELECT query. Return ONLY SQL.\n{SCHEMA_SQL}",
                    },
                    {"role": "user", "content": question},
                ]
            )
            sql = response.text.strip().strip("`")
            if response.used_fallback or not _is_safe_select(sql):
                return None, False
            return sql, True
        except Exception:
            return None, False

    def _deterministic_sql(self, question: str) -> str:
        q = question.lower()
        where: list[str] = []
        if "icu" in q:
            where.append("capability LIKE '%ICU%'")
        if "mri" in q:
            where.append("equipment LIKE '%MRI%'")
        if "hospital" in q:
            where.append("facilityTypeId = 'hospital'")
        region = _region_from_query(q)
        if region:
            where.append(f"address_stateOrRegion = '{region}'")
        specialty = _specialty_from_no_query(q)
        if specialty:
            where.append(f"(specialties IS NULL OR specialties NOT LIKE '%{specialty}%')")
        clause = " WHERE " + " AND ".join(where) if where else ""
        if "how many" in q or "count" in q:
            return f"SELECT COUNT(*) AS count FROM facility_capabilities{clause}"
        return f"SELECT * FROM facility_capabilities{clause} LIMIT 25"


def plan_facility_query(question: str) -> GeniePlan:
    """Create and execute a Genie-style query plan."""

    return SQLiteGenieAdapter().plan(question)


def _is_safe_select(sql: str) -> bool:
    return bool(re.match(r"^\s*select\b", sql, re.IGNORECASE)) and ";" not in sql.rstrip(";")


def _int(value) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _region_from_query(q: str) -> str | None:
    regions = [
        "Greater Accra",
        "Ashanti",
        "Northern",
        "Upper East",
        "Upper West",
        "Western",
        "Eastern",
        "Central",
        "Volta",
        "Bono",
        "Western North",
        "Ahafo",
        "Bono East",
        "Oti",
        "North East",
        "Savannah",
    ]
    return next((region for region in regions if region.lower() in q), None)


def _specialty_from_no_query(q: str) -> str | None:
    if " no " not in f" {q} ":
        return None
    specialties = [
        "internalMedicine",
        "familyMedicine",
        "pediatrics",
        "cardiology",
        "generalSurgery",
        "emergencyMedicine",
        "gynecologyAndObstetrics",
        "orthopedicSurgery",
        "dentistry",
        "ophthalmology",
    ]
    normalized = q.replace(" ", "").lower()
    return next((specialty for specialty in specialties if specialty.lower() in normalized), None)
