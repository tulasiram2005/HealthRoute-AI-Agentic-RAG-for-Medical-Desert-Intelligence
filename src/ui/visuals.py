"""Plotly visualizations for the Streamlit command center."""

from __future__ import annotations

import pandas as pd

from src.models.analysis import AnomalyReport, CareAccessIndex
from src.models.facility import FacilityModel


def care_index_figure(indices: list[CareAccessIndex]):
    """Create a polished horizontal Care Access Index chart."""

    import plotly.express as px

    df = pd.DataFrame([index.model_dump(mode="json") for index in indices]).sort_values("score")
    fig = px.bar(
        df,
        x="score",
        y="region",
        orientation="h",
        color="score",
        color_continuous_scale=["#FF4D4D", "#FFD166", "#00D4AA"],
        range_color=[0, 100],
        text="score",
        title="Care Access Index by Region",
    )
    return dark_layout(fig)


def anomaly_donut(anomalies: list[AnomalyReport]):
    """Create anomaly severity donut."""

    import plotly.express as px

    rows = [report.model_dump(mode="json") for report in anomalies]
    df = pd.DataFrame(rows) if rows else pd.DataFrame({"severity": ["none"], "count": [0]})
    if "count" not in df.columns:
        df = df.groupby("severity", as_index=False).size().rename(columns={"size": "count"})
    fig = px.pie(
        df,
        names="severity",
        values="count",
        hole=0.58,
        color="severity",
        color_discrete_map={"critical": "#FF4D4D", "high": "#FF8A4D", "medium": "#FFD166", "low": "#6CA6FF", "none": "#00D4AA"},
        title="Validation Risk Mix",
    )
    return dark_layout(fig)


def access_map_figure(facilities: list[FacilityModel], indices: list[CareAccessIndex]):
    """Create an inline Plotly map figure."""

    import plotly.express as px

    score_by_region = {index.region: index.score for index in indices}
    rows = []
    for index, facility in enumerate(facilities):
        rows.append(
            {
                "facility": facility.name,
                "region": facility.address_stateOrRegion,
                "district": facility.address_district or "Unknown",
                "lat": facility.latitude if facility.latitude is not None else 7.9465 + (index * 0.06),
                "lon": facility.longitude if facility.longitude is not None else -1.0232 + (index * 0.06),
                "beds": max(5, facility.capacity),
                "access_score": score_by_region.get(facility.address_stateOrRegion, 50),
                "source": facility.source_row_id or facility.id,
                "specialties": ", ".join(facility.specialties[:3]) or "No listed specialty",
            }
        )
    fig = px.scatter_geo(
        pd.DataFrame(rows),
        lat="lat",
        lon="lon",
        color="access_score",
        size="beds",
        hover_name="facility",
        hover_data=["region", "district", "specialties", "source"],
        color_continuous_scale=["#FF4D4D", "#FFD166", "#00D4AA"],
        range_color=[0, 100],
        projection="natural earth",
        title="Facility Footprint and Access Risk",
        height=640,
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
    return dark_layout(fig)


def dark_layout(fig):
    """Apply shared visual polish to Plotly figures."""

    fig.update_layout(
        paper_bgcolor="#07111F",
        plot_bgcolor="#07111F",
        font={"color": "#F7FBFF"},
        title_font={"color": "#F7FBFF", "size": 21},
        margin={"l": 8, "r": 8, "t": 56, "b": 8},
        coloraxis_colorbar={"title": ""},
    )
    return fig

