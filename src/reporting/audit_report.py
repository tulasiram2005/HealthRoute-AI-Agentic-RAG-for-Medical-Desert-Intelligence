"""Board-ready audit report generation."""

from __future__ import annotations

from html import escape
from pathlib import Path

from src.models.analysis import AnomalyReport, CareAccessIndex
from src.models.facility import FacilityModel
from src.planning.scenario import InterventionTarget, verification_queue


def build_audit_html(
    facilities: list[FacilityModel],
    anomalies: list[AnomalyReport],
    indices: list[CareAccessIndex],
    targets: list[InterventionTarget],
    output_path: str = "artifacts/virtue_audit_report.html",
) -> str:
    """Generate a self-contained audit trail report for NGO stakeholders."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    worst_regions = sorted(indices, key=lambda item: item.score)[:5]
    queue = verification_queue(facilities, anomalies)[:5]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Virtue Foundation IDP Audit Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #10223B; margin: 36px; line-height: 1.45; }}
    h1, h2 {{ color: #0A1628; }}
    .metric {{ display: inline-block; min-width: 170px; padding: 14px; margin: 8px; border: 1px solid #D5E3F0; border-radius: 8px; }}
    .metric strong {{ display: block; font-size: 28px; color: #007C6C; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
    th, td {{ border: 1px solid #D5E3F0; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #EFF8F6; }}
    .critical {{ color: #A40000; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Virtue Foundation IDP Audit Report</h1>
  <p>This report shows the data used, extraction outcomes, validation findings, and recommended action path for medical desert planning.</p>
  <div class="metric"><strong>{len(facilities)}</strong> Facilities parsed</div>
  <div class="metric"><strong>{len(anomalies)}</strong> Validation signals</div>
  <div class="metric"><strong>{sum(1 for i in indices if i.score < 40)}</strong> Medical deserts</div>
  <div class="metric"><strong>{sum(t.expected_patient_impact for t in targets[:3])}</strong> Est. patients impacted</div>

  <h2>Lowest Access Regions</h2>
  <table><tr><th>Region</th><th>Score</th><th>Priority</th><th>Specialty gaps</th><th>Citations</th></tr>
  {''.join(f'<tr><td>{escape(i.region)}</td><td>{i.score}</td><td>{i.recommendation_priority}</td><td>{escape(", ".join(i.specialty_gaps[:5]))}</td><td>{escape(", ".join(i.flagged_facilities))}</td></tr>' for i in worst_regions)}
  </table>

  <h2>Top Intervention Targets</h2>
  <table><tr><th>Region</th><th>Specialty</th><th>Urgency</th><th>Patient impact</th><th>Rationale</th><th>Source rows</th></tr>
  {''.join(f'<tr><td>{escape(t.region)}</td><td>{escape(t.specialty)}</td><td>{t.urgency_score}</td><td>{t.expected_patient_impact}</td><td>{escape(t.rationale)}</td><td>{escape(", ".join(t.citations))}</td></tr>' for t in targets[:8])}
  </table>

  <h2>Field Verification Queue</h2>
  <table><tr><th>Facility</th><th>Region</th><th>Risk score</th><th>Evidence</th><th>Action</th></tr>
  {''.join(f'<tr><td>{escape(row["facility"])}</td><td>{escape(row["region"])}</td><td class="critical">{row["risk_score"]}</td><td>{escape(row["top_evidence"])}</td><td>{escape(row["recommended_action"])}</td></tr>' for row in queue)}
  </table>
  <h2>Traceability</h2>
  <p>Every recommendation above links to source row IDs. In production, the same shape is logged as MLflow child-run artifacts for extraction, validation, anomaly detection, scoring, and recommendation steps.</p>
</body>
</html>"""
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path

