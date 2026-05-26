"""Offline-first Ghana impact map generation."""

from __future__ import annotations

from pathlib import Path

from src.models.analysis import CareAccessIndex
from src.models.facility import FacilityModel


def _access_lookup(indices: list[CareAccessIndex] | None) -> dict[str, float]:
    return {index.region: index.score for index in indices or []}


def build_facility_map(
    facilities: list[FacilityModel],
    output_path: str = "artifacts/ghana_medical_desert_map.html",
    indices: list[CareAccessIndex] | None = None,
) -> str:
    """Create a standalone, visible map without relying on online map tiles."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    scores = _access_lookup(indices)
    try:
        import plotly.express as px
        import pandas as pd
    except Exception:
        html = _fallback_map_html(facilities, scores)
        Path(output_path).write_text(html, encoding="utf-8")
        return output_path

    rows = []
    for index, facility in enumerate(facilities):
        lat = facility.latitude if facility.latitude is not None else 7.9465 + (index * 0.06)
        lon = facility.longitude if facility.longitude is not None else -1.0232 + (index * 0.06)
        access_score = scores.get(facility.address_stateOrRegion, 50.0)
        rows.append(
            {
                "facility": facility.name,
                "region": facility.address_stateOrRegion,
                "district": facility.address_district or "Unknown",
                "lat": lat,
                "lon": lon,
                "beds": facility.capacity,
                "doctors": facility.numberDoctors,
                "type": facility.facilityTypeId,
                "specialties": ", ".join(facility.specialties[:4]) or "No specialty listed",
                "capabilities": ", ".join(facility.capability[:4]) or "No capability listed",
                "access_score": access_score,
                "source_row_id": facility.source_row_id or facility.id,
            }
        )
    df = pd.DataFrame(rows)
    fig = px.scatter_geo(
        df,
        lat="lat",
        lon="lon",
        color="access_score",
        size="beds",
        hover_name="facility",
        hover_data={
            "region": True,
            "district": True,
            "type": True,
            "doctors": True,
            "specialties": True,
            "capabilities": True,
            "source_row_id": True,
            "lat": False,
            "lon": False,
        },
        color_continuous_scale=["#FF4D4D", "#FFD166", "#00D4AA"],
        range_color=[0, 100],
        projection="natural earth",
        title="Ghana Healthcare Access and Facility Capability Map",
        height=660,
    )
    fig.update_geos(
        visible=True,
        resolution=50,
        showcountries=True,
        countrycolor="#7A8EA3",
        showsubunits=True,
        subunitcolor="#243B55",
        showland=True,
        landcolor="#0D2138",
        showocean=True,
        oceancolor="#07111F",
        lataxis_range=[4, 12],
        lonaxis_range=[-4, 2],
    )
    fig.update_layout(
        paper_bgcolor="#07111F",
        plot_bgcolor="#07111F",
        font={"color": "#F7FBFF"},
        title_font={"size": 22, "color": "#F7FBFF"},
        coloraxis_colorbar={"title": "Care Access"},
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
    )
    html = fig.to_html(full_html=True, include_plotlyjs=True)
    html += "<section id='facility-index' style='font-family:Arial;padding:18px;background:#07111F;color:#F7FBFF'>"
    html += "<h2>Facility Index</h2><ul>"
    html += "".join(
        f"<li><strong>{row['facility']}</strong> - {row['region']} - {row['source_row_id']} - access {row['access_score']}</li>"
        for row in rows
    )
    html += "</ul></section>"
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def _fallback_map_html(facilities: list[FacilityModel], scores: dict[str, float]) -> str:
    points = []
    for index, facility in enumerate(facilities):
        x = 18 + (index % 5) * 15
        y = 14 + (index // 5) * 15
        score = scores.get(facility.address_stateOrRegion, 50)
        color = "#00D4AA" if score >= 70 else "#FFD166" if score >= 40 else "#FF4D4D"
        points.append(
            f"<circle cx='{x}' cy='{y}' r='4' fill='{color}'><title>{facility.name} - {facility.address_stateOrRegion}</title></circle>"
        )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Ghana Medical Desert Map</title></head>
<body style="background:#07111F;color:#F7FBFF;font-family:Arial">
<h1>Ghana Healthcare Access Map</h1>
<svg viewBox="0 0 100 80" style="width:100%;max-width:900px;background:#0D2138;border:1px solid #1D4B74;border-radius:8px">
<path d="M44 4 L62 10 L70 26 L66 48 L58 72 L36 76 L22 58 L18 36 L28 14 Z" fill="#102B46" stroke="#6B86A2" stroke-width="1.5"/>
{''.join(points)}
</svg>
<ul id="facility-index">{''.join(f'<li>{f.name} - {f.address_stateOrRegion}</li>' for f in facilities)}</ul>
</body></html>"""
