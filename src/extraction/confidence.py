"""Confidence scoring helpers for extraction quality."""

from __future__ import annotations


def confidence_for_terms(source_text: str, extracted_terms: list[str]) -> float:
    """Score extraction confidence using term presence, specificity, and source quality."""

    if not extracted_terms or not source_text:
        return 0.0

    text_lower = source_text.lower()
    scores: list[float] = []
    for term in extracted_terms:
        term_lower = str(term).lower()
        presence = 1.0 if term_lower in text_lower else 0.0
        specificity = min(1.0, len(term_lower.split()) / 4)
        context_words = ["has", "provides", "offers", "equipped", "available", "capacity", "with"]
        context_bonus = 0.15 if any(word in text_lower for word in context_words) else 0.0
        negation_words = ["no ", "not ", "without ", "lacks ", "does not "]
        negation = -0.3 if any(f"{neg}{term_lower}" in text_lower for neg in negation_words) else 0.0
        term_score = (presence * 0.6) + (specificity * 0.25) + context_bonus + negation
        scores.append(max(0.0, min(1.0, term_score)))
    return round(sum(scores) / len(scores), 3)


def field_confidence(row: dict, field_name: str, extracted_value) -> float:
    """Estimate whether an extracted value came from a direct field or inference."""

    if row.get(field_name) and extracted_value:
        return 0.95
    if extracted_value and not row.get(field_name):
        return 0.65
    return 0.1


def extraction_quality_grade(confidence_scores: dict[str, float]) -> str:
    """Convert confidence scores into a planner-friendly quality grade."""

    overall = confidence_scores.get("overall", 0.0)
    if overall >= 0.85:
        return "A - High quality: data well-supported by source records"
    if overall >= 0.70:
        return "B - Good quality: most data verified in source"
    if overall >= 0.50:
        return "C - Fair quality: some data inferred, recommend review"
    return "D - Low quality: significant data gaps, field verification recommended"
