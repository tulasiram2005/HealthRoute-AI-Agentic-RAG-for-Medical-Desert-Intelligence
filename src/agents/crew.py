"""CrewAI multi-agent execution with deterministic fallback."""

from __future__ import annotations

import os
from typing import Any

from src.extraction.parser import parse_facility_text
from src.planning.desert_scorer import compute_care_index
from src.planning.recommender import generate_recommendations
from src.validation.anomaly_detector import detect_anomalies

ROLE_DESCRIPTIONS = {
    "Extractor Agent": "Parse raw facility text and extract structured medical facts.",
    "Validator Agent": "Verify strict schema and healthcare business rules.",
    "Analyst Agent": "Identify medical deserts, capability gaps, and suspicious claims.",
    "Planner Agent": "Generate ranked deployment recommendations for NGO planners.",
    "Reporter Agent": "Compile findings into a citeable evidence report.",
}

COLORS = {
    "Extractor Agent": "green",
    "Validator Agent": "blue",
    "Analyst Agent": "orange",
    "Planner Agent": "purple",
    "Reporter Agent": "teal",
}


def build_crew():
    """Return a configured CrewAI crew when installed, otherwise role metadata."""

    if os.getenv("VIRTUE_ENABLE_NATIVE_CREWAI", "").lower() not in {"1", "true", "yes"}:
        return ROLE_DESCRIPTIONS
    try:
        from crewai import Agent, Crew, Process, Task
    except Exception:
        return ROLE_DESCRIPTIONS
    agents = [Agent(role=name, goal=goal, backstory="Healthcare coordination specialist.", verbose=True) for name, goal in ROLE_DESCRIPTIONS.items()]
    tasks = [
        Task(description=f"{agent.role}: {agent.goal}", expected_output="Structured JSON findings with citations.", agent=agent)
        for agent in agents
    ]
    return Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=True)


def run_crew_analysis(row: dict[str, Any]) -> dict[str, Any]:
    """Run five-agent analysis, using deterministic tools when CrewAI is unavailable."""

    extraction = parse_facility_text(row)
    facility = extraction.extracted_facility
    anomalies = detect_anomalies(facility, " ".join(str(row.get(k, "") or "") for k in ["description", "procedure", "equipment", "capability"]))
    care_index = compute_care_index([facility], anomalies, facility.address_stateOrRegion)
    recommendations = generate_recommendations([care_index])
    findings = [
        {
            "agent": "Extractor Agent",
            "color": COLORS["Extractor Agent"],
            "finding": f"Extracted {facility.name} with {len(facility.specialties)} specialties, {len(facility.equipment)} equipment items, and confidence {extraction.confidence_scores.get('overall', 0):.0%}.",
            "data": extraction.model_dump(mode="json"),
        },
        {
            "agent": "Validator Agent",
            "color": COLORS["Validator Agent"],
            "finding": "Strict Pydantic validation passed; business-rule anomalies are passed to the analyst.",
            "data": facility.model_dump(mode="json"),
        },
        {
            "agent": "Analyst Agent",
            "color": COLORS["Analyst Agent"],
            "finding": f"Detected {len(anomalies)} anomaly signal(s) and computed regional access score {care_index.score}/100.",
            "data": {"anomalies": [a.model_dump(mode="json") for a in anomalies], "care_index": care_index.model_dump(mode="json")},
        },
        {
            "agent": "Planner Agent",
            "color": COLORS["Planner Agent"],
            "finding": recommendations[0] if recommendations else "No deployment recommendation was needed.",
            "data": {"recommendations": recommendations},
        },
        {
            "agent": "Reporter Agent",
            "color": COLORS["Reporter Agent"],
            "finding": f"Compiled citeable report for source row {extraction.source_row_id}.",
            "data": {"citations": [extraction.source_row_id]},
        },
    ]
    crew_output = None
    if os.getenv("VIRTUE_ENABLE_NATIVE_CREWAI", "").lower() in {"1", "true", "yes"}:
        try:
            crew = build_crew()
            if not isinstance(crew, dict):
                crew_output = str(crew.kickoff(inputs={"facility": facility.model_dump(mode="json")}))
        except Exception as exc:
            crew_output = f"Deterministic fallback used because CrewAI execution was unavailable: {exc}"
    return {
        "crew_output": crew_output or "Deterministic five-agent fallback completed.",
        "agent_findings": findings,
        "citation_map": {finding["agent"]: [extraction.source_row_id] for finding in findings},
    }
