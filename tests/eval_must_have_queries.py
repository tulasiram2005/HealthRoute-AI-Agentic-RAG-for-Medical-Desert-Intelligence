"""Evaluation tests for the 10 must-have hackathon queries."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.extraction.parser import parse_facility_text
from src.planning.query_engine import answer_query
from src.validation.anomaly_detector import detect_anomalies

DATA_PATH = Path("data/ghana_facilities.csv")


@pytest.fixture(scope="module")
def loaded_system():
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    facilities = [parse_facility_text(row).extracted_facility for row in rows]
    anomalies = [report for facility in facilities for report in detect_anomalies(facility)]
    return facilities, anomalies


MUST_HAVE_QUERIES = [
    {
        "query": "How many hospitals in Ghana have ICU capability?",
        "assert": lambda r: isinstance(r["answer"], str) and any(c.isdigit() for c in r["answer"]),
        "assert_msg": "Answer must contain a number",
        "min_confidence": 0.8,
        "min_sources": 1,
    },
    {
        "query": "Which regions have no emergency medicine specialists?",
        "assert": lambda r: isinstance(r.get("data"), list),
        "assert_msg": "Must return list of regions",
        "min_confidence": 0.8,
        "min_sources": 0,
    },
    {
        "query": "What is the Care Access Index for Ashanti region?",
        "assert": lambda r: "Ashanti" in r["answer"] and any(c.isdigit() for c in r["answer"]),
        "assert_msg": "Answer must mention Ashanti and a score",
        "min_confidence": 0.8,
        "min_sources": 1,
    },
    {
        "query": "List all facilities with MRI equipment",
        "assert": lambda r: isinstance(r.get("data"), list),
        "assert_msg": "Must return facility list",
        "min_confidence": 0.7,
        "min_sources": 0,
    },
    {
        "query": "Which facility has the most suspicious capability claims?",
        "assert": lambda r: isinstance(r["answer"], str) and len(r["answer"]) > 10,
        "assert_msg": "Must return a facility name or no anomalies",
        "min_confidence": 0.8,
        "min_sources": 0,
    },
    {
        "query": "What specialties are most underrepresented in Northern Ghana?",
        "assert": lambda r: isinstance(r.get("data"), list),
        "assert_msg": "Must return specialty list",
        "min_confidence": 0.7,
        "min_sources": 0,
    },
    {
        "query": "Find all pediatric facilities within 100km of Accra",
        "assert": lambda r: isinstance(r.get("data"), list),
        "assert_msg": "Must return facility list with distances",
        "min_confidence": 0.7,
        "min_sources": 0,
    },
    {
        "query": "Which facilities have been flagged as CRITICAL anomalies?",
        "assert": lambda r: isinstance(r.get("data"), list),
        "assert_msg": "Must return list, possibly empty",
        "min_confidence": 0.8,
        "min_sources": 0,
    },
    {
        "query": "What is the best facility for cardiac surgery in Ghana?",
        "assert": lambda r: isinstance(r["answer"], str) and len(r["answer"]) > 20,
        "assert_msg": "Must return facility name with justification",
        "min_confidence": 0.7,
        "min_sources": 1,
    },
    {
        "query": "Generate a 3-month deployment plan for 2 orthopedic surgeons",
        "assert": lambda r: isinstance(r.get("data"), list) and len(r["data"]) > 0,
        "assert_msg": "Must return deployment plan with at least 1 match",
        "min_confidence": 0.7,
        "min_sources": 1,
    },
]


@pytest.mark.parametrize("case", MUST_HAVE_QUERIES, ids=[case["query"][:40] for case in MUST_HAVE_QUERIES])
def test_must_have_query(case, loaded_system):
    facilities, anomalies = loaded_system
    result = answer_query(case["query"], facilities, anomalies)

    assert isinstance(result, dict), "answer_query must return dict"
    assert "answer" in result, "result must have answer key"
    assert "citations" in result, "result must have citations key"
    assert "confidence" in result, "result must have confidence key"
    assert "sources" in result, "result must have sources key"
    assert case["assert"](result), f"Content assertion failed: {case['assert_msg']}\nGot: {result['answer']}"
    assert result["confidence"] >= case["min_confidence"]
    assert len(result["sources"]) >= case["min_sources"]


def test_all_10_queries_pass(loaded_system):
    facilities, anomalies = loaded_system
    for case in MUST_HAVE_QUERIES:
        result = answer_query(case["query"], facilities, anomalies)
        assert result["answer"], f"Empty answer for: {case['query']}"
