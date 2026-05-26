"""Map rendering tests."""

from __future__ import annotations

from pathlib import Path

from src.ui.map_generator import build_facility_map


def test_map_export(sample_facility, tmp_path):
    output = build_facility_map([sample_facility], str(tmp_path / "map.html"))
    assert Path(output).exists()
    assert "Test Regional Hospital" in Path(output).read_text(encoding="utf-8")

