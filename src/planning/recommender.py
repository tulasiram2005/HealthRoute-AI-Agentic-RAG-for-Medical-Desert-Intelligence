"""Deployment recommendation engine."""

from __future__ import annotations

from src.models.analysis import CareAccessIndex


def generate_recommendations(indices: list[CareAccessIndex]) -> list[str]:
    """Generate plain-language NGO deployment recommendations."""

    recommendations: list[str] = []
    for index in sorted(indices, key=lambda item: (item.recommendation_priority * -1, item.score)):
        if not index.specialty_gaps:
            continue
        specialty = index.specialty_gaps[0]
        patients = max(5000, int((100 - index.score) * 1200))
        recommendations.append(
            f"Deploy {specialty} to {index.district}, {index.region} -- serves approximately {patients} patients -- priority {index.recommendation_priority}"
        )
    return recommendations[:10]

