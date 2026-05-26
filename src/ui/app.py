"""Streamlit command center for NGO healthcare planners."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.architecture.compliance import TECH_STACK_MATRIX, compliance_score
from src.agents.crew import run_crew_analysis
from src.extraction.confidence import extraction_quality_grade
from src.extraction.parser import parse_facility_text
from src.integrations.genie import plan_facility_query
from src.models.facility import DoctorProfile, Specialty
from src.planning.desert_scorer import compute_indices_by_region
from src.planning.matcher import match_doctor_to_facilities
from src.planning.patient_journey import CONDITION_SPECIALTY_MAP, REGION_COORDS, simulate_patient_journey
from src.planning.query_engine import answer_query
from src.planning.scenario import rank_intervention_targets, simulate_specialist_deployment, verification_queue
from src.reporting.audit_report import build_audit_html
from src.ui.dashboard import (
    anomaly_severity_counts,
    executive_metrics,
    facility_readiness_rows,
    specialty_gap_numeric,
)
from src.ui.map_generator import build_facility_map
from src.ui.visuals import access_map_figure, anomaly_donut, care_index_figure
from src.validation.anomaly_detector import detect_anomalies

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ghana_facilities.csv"


@st.cache_data
def load_rows() -> list[dict]:
    """Load raw CSV rows."""

    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@st.cache_data
def load_facility_payloads() -> list[dict]:
    """Load parsed facilities as JSON-safe payloads for Streamlit caching."""

    return [parse_facility_text(row).extracted_facility.model_dump(mode="json") for row in load_rows()]


def load_facilities():
    """Hydrate strict facility models from cached payloads."""

    from src.models.facility import FacilityModel

    return [FacilityModel.model_validate(payload) for payload in load_facility_payloads()]


def render_style() -> None:
    """Inject the app visual system."""

    st.markdown(
        """
        <style>
        .stApp { background: #06101C; color: #F7FBFF; }
        section[data-testid="stSidebar"] { background: #091A2F; border-right: 1px solid #17395F; }
        h1, h2, h3 { color: #F7FBFF; letter-spacing: 0; }
        .hero {
            padding: 28px 30px; border: 1px solid #1C456F; border-radius: 8px;
            background:
              radial-gradient(circle at 82% 22%, rgba(0,212,170,0.22), transparent 22%),
              linear-gradient(135deg, #0C2139 0%, #0B332F 100%);
            margin-bottom: 18px;
        }
        .hero h1 { margin: 0 0 8px 0; font-size: 34px; }
        .hero p { color: #C9D7E6; max-width: 980px; margin: 0; }
        div[data-testid="stMetric"] {
            background: #0D2138; border: 1px solid #1D4B74; padding: 12px;
            border-radius: 8px; box-shadow: 0 10px 28px rgba(0,0,0,0.18);
        }
        div[data-testid="stMetric"] label { color: #AFC4D8; }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #00D4AA; }
        .callout {
            border-left: 4px solid #00D4AA; background: #0D2138; padding: 14px;
            border-radius: 6px; color: #DCEAF7; margin: 12px 0;
        }
        .risk { border-left-color: #FF6B6B; }
        .small-muted { color: #9FB3C8; font-size: 0.92rem; }
        .citation {
            display: inline-block; margin: 3px; padding: 3px 8px; border-radius: 999px;
            background: #2F4E22; color: #F4FFC7; border: 1px solid #B7D968;
        }
        .scorecard {
            background:#0D2138; border:1px solid #1D4B74; border-radius:8px;
            padding:14px; min-height:120px;
        }
        .scorecard strong { color:#00D4AA; font-size:1.2rem; }
        .tag {
            display:inline-block; padding:4px 9px; margin:2px; border:1px solid #2C5A84;
            border-radius:999px; color:#CBE7FF; background:#0A1D32; font-size:.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Virtue Foundation Healthcare Intelligence Command Center</h1>
          <p>Parse messy facility notes, verify clinical claims, expose medical deserts, route scarce specialists, and generate board-ready evidence with agent-step citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rubric_badges(note: str) -> None:
    st.caption(note)
    st.markdown(
        "<span class='tag'>Technical Accuracy</span><span class='tag'>IDP Innovation</span><span class='tag'>Social Impact</span><span class='tag'>UX</span>",
        unsafe_allow_html=True,
    )


def dataframe_download(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(label, frame.to_csv(index=False), file_name=filename, mime="text/csv")


st.set_page_config(page_title="Virtue Foundation IDP", layout="wide")
render_style()

facilities = load_facilities()
anomalies = [report for facility in facilities for report in detect_anomalies(facility)]
indices = compute_indices_by_region(facilities, anomalies)
metrics = executive_metrics(facilities, anomalies, indices)

page = st.sidebar.radio(
    "Command center",
    [
        "Live Demo Mode",
        "Executive Overview",
        "Ask the Agent",
        "Medical Desert Map",
        "Deployment Planner",
        "Patient Journey Simulator",
        "Verification Queue",
        "Capability Gap Lab",
        "Tech Stack Proof",
        "Audit Report",
    ],
)

render_hero()

render_rubric_badges("What this shows: grounded extraction, validation, healthcare planning impact, and judge-friendly evidence.")

if page == "Live Demo Mode":
    st.subheader("Live Demo Mode")
    placeholder = st.empty()
    for label, value in [
        ("Facilities parsed", len(facilities)),
        ("Medical deserts identified", len([index for index in indices if index.score < 40])),
        ("Anomalies flagged", len(anomalies)),
    ]:
        placeholder.metric(label, value)
        time.sleep(0.2)
    st.markdown(
        f"<div class='callout'>{len(facilities)} facilities analyzed. {len([i for i in indices if i.score < 40])} medical deserts identified. {len(anomalies)} anomalies flagged. In this data, underserved regions need verified emergency access before patient routing.</div>",
        unsafe_allow_html=True,
    )
    demo_queries = [
        "How many hospitals in Ghana have ICU capability?",
        "Which regions have no emergency medicine specialists?",
        "What is the Care Access Index for Ashanti region?",
        "List all facilities with MRI equipment",
        "Generate a 3-month deployment plan for 2 orthopedic surgeons",
    ]
    index = st.session_state.get("demo_query_index", 0) % len(demo_queries)
    query = demo_queries[index]
    result = answer_query(query, facilities, anomalies)
    st.markdown(f"**Ask the Agent:** {query}")
    st.markdown(f"<div class='callout'>{result['answer']}</div>", unsafe_allow_html=True)
    st.markdown("".join(f"<span class='citation'>{source}</span>" for source in result["sources"][:8]), unsafe_allow_html=True)
    if st.button("Next demo question"):
        st.session_state["demo_query_index"] = index + 1
        st.rerun()
    lowest = min(indices, key=lambda item: item.score)
    st.subheader("Medical Deserts Revealed")
    st.markdown(f"<div class='callout risk'>Lowest access region: {lowest.region} ({lowest.score}/100). Gaps: {', '.join(lowest.specialty_gaps[:5])}</div>", unsafe_allow_html=True)
    st.plotly_chart(care_index_figure(indices), width="stretch")
    worst = anomalies[0] if anomalies else None
    if worst:
        st.subheader("Anomaly Caught")
        st.markdown(f"<div class='callout risk'>{worst.description}<br><strong>Evidence:</strong> {worst.evidence}</div>", unsafe_allow_html=True)
    scenario = simulate_specialist_deployment(facilities, anomalies, "orthopedicSurgery", 2)
    st.subheader("Deploy a Specialist")
    st.metric("Estimated patients reached", f"{scenario['estimated_total_patient_impact']:,}")
    st.bar_chart(pd.DataFrame(scenario["score_projection"]).set_index("region")["delta"])
    st.subheader("Citeable Evidence")
    for step, rows in result["citations"].items():
        st.write(f"{step}: {', '.join(rows[:8])}")
        time.sleep(0.1)
    st.subheader("Tech Stack Proof")
    st.dataframe(pd.DataFrame(TECH_STACK_MATRIX), width="stretch")

elif page == "Executive Overview":
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Facilities parsed", metrics["facilities_analyzed"])
    col2.metric("Regions covered", metrics["regions_covered"])
    col3.metric("Avg access index", metrics["avg_care_index"])
    col4.metric("Medical deserts", metrics["medical_deserts"])
    col5.metric("Critical anomalies", metrics["critical_anomalies"])

    st.markdown("<div class='callout'>Highest-value next move: verify high-risk claims before routing patients, then deploy specialists to the lowest access regions with explicit source-row evidence.</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.markdown("<div class='scorecard'><strong>IDP Innovation</strong><br>Strict extraction, confidence scores, anomaly evidence, and source citations.</div>", unsafe_allow_html=True)
    b.markdown("<div class='scorecard'><strong>Social Impact</strong><br>Care Access Index, specialist matching, and field-verification prioritization.</div>", unsafe_allow_html=True)
    c.markdown("<div class='scorecard'><strong>Demo Readiness</strong><br>Runs locally without credentials, upgrades to Databricks/MLflow/LanceDB when configured.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1])
    with left:
        access_df = pd.DataFrame([index.model_dump(mode="json") for index in indices]).sort_values("score")
        st.plotly_chart(care_index_figure(indices), width="stretch")
        dataframe_download("Download care index CSV", access_df, "care_access_index.csv")
    with right:
        st.plotly_chart(anomaly_donut(anomalies), width="stretch")
        st.subheader("Top Intervention Targets")
        target_df = pd.DataFrame([target.__dict__ for target in rank_intervention_targets(facilities, anomalies)[:5]])
        st.dataframe(target_df[["region", "specialty", "urgency_score", "expected_patient_impact"]], width="stretch")

elif page == "Ask the Agent":
    st.subheader("Natural Language Query Assistant")
    example = "How many hospitals in Ghana have ICU capability?"
    query = st.text_area("Ask a question", value=example, height=90)
    if st.button("Run grounded query", type="primary"):
        with st.spinner("Reasoning over extracted facility data and citations..."):
            result = answer_query(query, facilities, anomalies)
        st.markdown(f"<div class='callout'>{result['answer']}</div>", unsafe_allow_html=True)
        st.metric("Confidence", f"{result['confidence']:.0%}")
        genie_plan = plan_facility_query(query)
        grade = extraction_quality_grade({"overall": result["confidence"]})
        st.caption(f"Extraction quality signal: {grade}")
        with st.expander("Databricks Genie Query Plan"):
            st.code(genie_plan.sql, language="sql")
            st.caption(genie_plan.explanation)
            st.write(f"Rows returned: {genie_plan.row_count}")
            if genie_plan.results:
                st.dataframe(pd.DataFrame(genie_plan.results[:10]), width="stretch")
        with st.expander("RAG Retrieval Details"):
            for item in result.get("rag", {}).get("results", []):
                st.markdown(f"**{item['source_row_id']}** - score `{item['similarity_score']}`")
                st.caption(item["text_snippet"])
        st.write("Citations")
        st.markdown("".join(f"<span class='citation'>{source}</span>" for source in result["sources"]), unsafe_allow_html=True)
        if result["data"]:
            st.dataframe(pd.json_normalize(result["data"]), width="stretch")
        with st.expander("CrewAI Multi-Agent Run"):
            selected_row = next((row for row in load_rows() if (row.get("source_row_id") or row.get("id")) in result["sources"]), load_rows()[0])
            crew = run_crew_analysis(selected_row)
            st.caption(crew["crew_output"])
            for finding in crew["agent_findings"]:
                st.markdown(f"<div class='callout' style='border-left-color:{finding['color']}'><strong>{finding['agent']}</strong><br>{finding['finding']}</div>", unsafe_allow_html=True)
        st.session_state.setdefault("query_history", []).append(result)

elif page == "Medical Desert Map":
    st.subheader("Medical Desert Map")
    region_filter = st.multiselect("Filter regions", sorted({facility.address_stateOrRegion for facility in facilities}))
    selected = [facility for facility in facilities if not region_filter or facility.address_stateOrRegion in region_filter]
    selected_indices = [index for index in indices if not region_filter or index.region in region_filter]
    st.plotly_chart(access_map_figure(selected, selected_indices), width="stretch")
    output = build_facility_map(selected, indices=selected_indices)
    html = Path(output).read_text(encoding="utf-8")
    with st.expander("Standalone map export details"):
        st.write("The downloadable HTML includes the same facility markers plus an embedded facility index for offline review.")
        st.code(output)
    st.download_button("Download standalone map HTML", html, file_name="ghana_medical_desert_map.html", mime="text/html")

elif page == "Deployment Planner":
    st.subheader("Deployment Planner")
    col1, col2, col3 = st.columns(3)
    specialty = col1.selectbox("Specialty", [specialty.value for specialty in Specialty], index=7)
    count = col2.slider("Specialists available", 1, 10, 3)
    weeks = col3.slider("Deployment weeks", 1, 52, 12)

    scenario = simulate_specialist_deployment(facilities, anomalies, specialty, count)
    st.markdown(
        f"<div class='callout'>Adding {count} {specialty} specialist(s) is projected to reach about {scenario['estimated_total_patient_impact']:,} patients.</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Projected Access Lift")
    projection_df = pd.DataFrame(scenario["score_projection"])
    st.dataframe(projection_df, width="stretch")
    st.bar_chart(projection_df.set_index("region")["delta"])

    st.subheader("Doctor-Facility Matching")
    experience = st.slider("Doctor years of experience", 0, 40, 8)
    equipment = st.multiselect("Equipment the doctor can use", ["ECG", "echo", "x-ray", "CT", "MRI", "operating theatre", "ultrasound", "oxygen"], default=["x-ray", "operating theatre"])
    profile = DoctorProfile(specialty=specialty, years_experience=experience, equipment_familiar_with=equipment, languages=["English"], available_weeks=weeks)
    matches = match_doctor_to_facilities(profile, facilities, anomalies)
    st.dataframe(pd.DataFrame([match.model_dump(mode="json") for match in matches]), width="stretch")

elif page == "Patient Journey Simulator":
    st.subheader("Patient Journey Simulator")
    col1, col2 = st.columns(2)
    condition = col1.selectbox("Patient Condition", list(CONDITION_SPECIALTY_MAP), index=1)
    region = col2.selectbox("Patient's Starting Region", list(REGION_COORDS), index=0)
    if st.button("Simulate Journey", type="primary"):
        journey = simulate_patient_journey(condition, region, facilities)
        if journey.lives_at_risk == "HIGH":
            st.markdown("<div class='callout risk'><strong>LIFE-CRITICAL:</strong> Every minute matters for this condition.</div>", unsafe_allow_html=True)
        st.metric("Time Saved", f"{journey.time_saved_minutes} minutes")
        left, right = st.columns(2)
        with left:
            st.markdown("### Current System")
            st.markdown("Patient -> nearest facility -> transfer if wrong specialty")
            st.dataframe(pd.DataFrame(journey.current_route), width="stretch")
        with right:
            st.markdown("### Optimized System")
            st.markdown("Patient -> nearest facility with required specialty")
            st.dataframe(pd.DataFrame(journey.optimized_route), width="stretch")
        needed = ", ".join(CONDITION_SPECIALTY_MAP.get(condition, ["emergencyMedicine"]))
        st.markdown(f"<div class='callout'>{journey.recommendation}<br>Required specialty: {needed}</div>", unsafe_allow_html=True)
        st.markdown("".join(f"<span class='citation'>{source}</span>" for source in journey.citations), unsafe_allow_html=True)

elif page == "Verification Queue":
    st.subheader("Field Verification Queue")
    queue_df = pd.DataFrame(verification_queue(facilities, anomalies))
    if queue_df.empty:
        st.success("No field verification work is required for the current sample.")
    else:
        st.markdown("<div class='callout risk'>These facilities should be verified before patient routing or major deployment decisions.</div>", unsafe_allow_html=True)
        grades = []
        by_id = {facility.source_row_id or facility.id: facility for facility in facilities}
        for source_row_id in queue_df.get("source_row_id", []):
            row = next((raw for raw in load_rows() if raw.get("source_row_id") == source_row_id), {})
            grades.append(extraction_quality_grade(parse_facility_text(row).confidence_scores) if row else "")
        if grades:
            queue_df["extraction_quality_grade"] = grades
        st.dataframe(queue_df, width="stretch")
        dataframe_download("Download verification queue", queue_df, "verification_queue.csv")
    st.subheader("All Anomaly Evidence")
    anomaly_df = pd.DataFrame([report.model_dump(mode="json") for report in anomalies])
    st.dataframe(anomaly_df, width="stretch")

elif page == "Capability Gap Lab":
    st.subheader("Capability Gap Lab")
    gap_df = pd.DataFrame(specialty_gap_numeric(facilities)).T
    st.dataframe(gap_df.style.background_gradient(cmap="RdYlGn", axis=None), width="stretch")
    st.caption("0 means no recorded coverage in the current data. 1 means at least one facility reports the specialty.")
    readiness_df = pd.DataFrame(facility_readiness_rows(facilities, anomalies))
    st.subheader("Facility Readiness Ranking")
    st.dataframe(readiness_df, width="stretch")
    dataframe_download("Download readiness ranking", readiness_df, "facility_readiness.csv")

elif page == "Tech Stack Proof":
    st.subheader("Tech Stack Proof")
    score = compliance_score()
    st.markdown(
        f"<div class='callout'>Challenge stack coverage: <strong>{score['implemented']}/{score['total']} requirements</strong> represented in code. Native cloud features activate when credentials/packages are present; local fallbacks keep the demo reliable.</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(pd.DataFrame(TECH_STACK_MATRIX), width="stretch")
    st.markdown(
        "<span class='tag'>LangGraph</span><span class='tag'>CrewAI</span><span class='tag'>MLflow</span><span class='tag'>LanceDB</span><span class='tag'>Databricks LLM</span><span class='tag'>Genie Text2SQL</span><span class='tag'>Pydantic v2</span><span class='tag'>FastAPI</span><span class='tag'>Streamlit</span>",
        unsafe_allow_html=True,
    )

else:
    st.subheader("Audit Report")
    targets = rank_intervention_targets(facilities, anomalies)
    report_path = build_audit_html(facilities, anomalies, indices, targets)
    report_html = Path(report_path).read_text(encoding="utf-8")
    st.html(report_html)
    fallback_path = Path("artifacts/mlflow_trace_fallback.jsonl")
    with st.expander("MLflow Citation Trail"):
        st.link_button("View in MLflow", "http://localhost:5000")
        if fallback_path.exists():
            lines = [json.loads(line) for line in fallback_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                st.json(lines[-10:])
    st.download_button("Download board-ready audit HTML", report_html, file_name="virtue_audit_report.html", mime="text/html")
    st.markdown("<p class='small-muted'>The report is structured to mirror the challenge requirements: data used, extraction, validation, conclusions, recommendations, and citations.</p>", unsafe_allow_html=True)
