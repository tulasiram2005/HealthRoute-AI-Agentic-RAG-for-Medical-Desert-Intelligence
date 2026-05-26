# Virtue Foundation IDP Agent

Agentic Intelligent Document Parsing project for the Databricks x Accenture healthcare coordination challenge. It extracts structured facts from messy medical facility text, validates claims, scores medical deserts, recommends interventions, and exposes both API and Streamlit planner UI.

## What Is Included

- LangGraph-compatible five-step agent pipeline with deterministic fallback
- CrewAI role definitions for extractor, validator, analyst, planner, and reporter agents
- Strict Pydantic v2 models for facilities, NGOs, extractions, anomalies, care index, and doctor matching
- Rule-based anomaly detection for ICU, surgery, imaging, capacity, NICU, trauma, and specialty contradictions
- Care Access Index and medical desert prioritization
- Doctor-facility matching engine with citations and patient impact estimates
- Grounded query engine for the hackathon must-have questions
- Specialist deployment scenario modeling with projected access-index lift
- Field verification queue for suspicious facility claims before patient routing
- Facility readiness ranking and specialty heatmap for non-technical planners
- Board-ready HTML audit report with source rows, evidence, and intervention rationale
- Offline-first Plotly Ghana impact map that does not depend on web map tiles
- Tech Stack Proof page showing how the requested challenge stack is implemented
- Credential-ready Databricks LLM, Databricks Genie-style, MLflow, and LanceDB adapters
- LanceDB-ready RAG ingestion with JSON fallback and hybrid retrieval
- FastAPI backend with query, facility analysis, care index, anomaly, map, matching, scenario, verification, and report endpoints
- Streamlit command center with executive overview, agent Q&A, medical desert map, deployment planner, verification queue, gap lab, and audit report
- Sample Ghana facility CSV and pytest suite

## Quickstart

```bash
cd virtue_foundation_idp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a lighter first run, install only the core dependencies:

```bash
pip install pydantic fastapi "uvicorn[standard]" pandas streamlit folium pytest pytest-asyncio pytest-cov
```

Copy `.env.example` if you want native Databricks/MLflow integrations. Without credentials, the project uses transparent local fallbacks so the demo does not fail on stage.

## Run Tests

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## Ingest Data

```bash
python src/rag/ingest.py --data data/ghana_facilities.csv
```

This writes a local fallback store to `artifacts/lancedb_fallback/facility_capabilities.json`. If LanceDB and sentence-transformers are installed, the project is ready to swap the fallback persistence for a LanceDB table.

## Start Backend

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open:

- API docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

Example:

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"Find hospitals with ICU capability","filters":{}}'
```

Useful demo endpoints:

```bash
curl http://localhost:8000/api/v1/intervention-targets
curl http://localhost:8000/api/v1/verification-queue
curl -X POST http://localhost:8000/api/v1/scenario \
  -H "Content-Type: application/json" \
  -d '{"specialty":"orthopedicSurgery","count":3}'
curl http://localhost:8000/api/v1/export/report
```

## Start Streamlit UI

```bash
streamlit run src/ui/app.py
```

Open http://localhost:8501.

## Optional MLflow UI

```bash
mlflow ui --port 5000
```

The `CitationTracker` class in `src/agents/citations.py` produces agent-step citation JSON compatible with MLflow artifact logging.

## Project Structure

```text
src/
  agents/       LangGraph nodes, graph runner, CrewAI roles, citations
  api/          FastAPI app and routes
  extraction/   Text parsing and confidence scoring
  models/       Strict Pydantic schemas
  planning/     Care index, recommendations, doctor matching
  rag/          Ingestion, embeddings, retrieval
  ui/           Streamlit UI and Folium map generation
  validation/   Schema and anomaly validation
tests/          Pytest suite
data/           Sample Ghana facility CSV
```

## Notes For Hackathon Demo

Set Databricks, OpenAI-compatible, or MLflow AI Gateway credentials in your environment and replace the deterministic parser calls inside `src/agents/nodes.py` with structured LLM calls. The surrounding validation, citations, scoring, API, UI, and tests are already shaped for that production path.

## Winning Demo Flow

1. Start on the Executive Overview and show the one-screen story: facilities parsed, deserts, anomalies, and average access.
2. Ask: "How many hospitals in Ghana have ICU capability?" and point to source-row citation chips.
3. Open Medical Desert Map and show the visible Plotly Ghana access-risk map.
4. Open the Verification Queue to show patient-safety reasoning before routing patients.
5. Run the Deployment Planner with `orthopedicSurgery` and 3 specialists to show projected access lift and patient impact.
6. Open Tech Stack Proof and explain the Databricks/MLflow/LanceDB/Genie upgrade path.
7. Open the Audit Report and download the board-ready HTML artifact.

## Honest Production Notes

This is now a high-quality hackathon prototype, not a fully deployed production system. It has the right architecture, strict validation, traceability, UI story, and local reliability. To make it production-grade, connect the adapters in `src/integrations/` to your Databricks workspace, replace the deterministic extraction fallback with structured LLM calls, and ingest the full Virtue Foundation dataset.
# HealthRoute-AI-Agentic-RAG-for-Medical-Desert-Intelligence
