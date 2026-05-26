"""Citation tracking and lightweight MLflow integration."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class CitationTracker:
    """Maintains step-to-source-row mappings across an agent run."""

    def __init__(self) -> None:
        self._citations: dict[str, set[str]] = defaultdict(set)

    def add(self, step: str, rows: list[str] | str) -> None:
        """Add one or more row IDs to a step."""

        if isinstance(rows, str):
            rows = [rows]
        for row in rows:
            self._citations[step].add(str(row))

    def as_dict(self) -> dict[str, list[str]]:
        """Return sorted JSON-serializable citations."""

        return {step: sorted(rows) for step, rows in self._citations.items()}

    def response(self, answer: str, confidence: float, run_id: str | None = None) -> dict[str, Any]:
        """Build the API citation response shape."""

        citations = self.as_dict()
        rows = sorted({row for step_rows in citations.values() for row in step_rows})
        return {
            "answer": answer,
            "citations": citations,
            "mlflow_run_url": f"http://localhost:5000/#/runs/{run_id}" if run_id else "",
            "total_rows_referenced": len(rows),
            "confidence": confidence,
        }

    def human_report(self, row_names: dict[str, str] | None = None) -> str:
        """Render a readable citation report."""

        row_names = row_names or {}
        lines: list[str] = []
        for idx, (step, rows) in enumerate(self.as_dict().items(), start=1):
            rendered = [f"#{row} ({row_names.get(row, 'unknown facility')})" for row in rows]
            lines.append(f"Step {idx} {step} used rows: {', '.join(rendered)}")
        return "\n".join(lines)

    def save_json(self, path: str | Path) -> None:
        """Persist citations as a JSON artifact."""

        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

