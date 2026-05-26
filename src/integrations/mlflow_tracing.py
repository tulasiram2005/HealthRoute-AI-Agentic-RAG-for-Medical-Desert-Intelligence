"""MLflow tracing with JSON fallback artifacts."""

from __future__ import annotations

import json
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class MLflowTraceLogger:
    """Logs agent steps to MLflow when available, otherwise to local JSONL."""

    def __init__(self, experiment_name: str = "virtue_foundation_idp_agent", fallback_path: str = "artifacts/mlflow_trace_fallback.jsonl") -> None:
        self.experiment_name = experiment_name
        self.fallback_path = Path(fallback_path)
        try:
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.fallback_path = Path(tempfile.gettempdir()) / "virtue_mlflow_trace_fallback.jsonl"
        try:
            import mlflow

            self.mlflow = mlflow
            self.mlflow.set_experiment(experiment_name)
        except Exception:
            self.mlflow = None

    @contextmanager
    def run(self, name: str, tags: dict[str, str] | None = None) -> Iterator[str]:
        """Create a parent run context."""

        run_id = str(uuid.uuid4())
        if self.mlflow is None:
            self._write({"event": "start_run", "name": name, "run_id": run_id, "tags": tags or {}})
            try:
                yield run_id
            finally:
                self._write({"event": "end_run", "run_id": run_id})
            return
        with self.mlflow.start_run(run_name=name, tags=tags or {}) as active:
            yield active.info.run_id

    def log_step(self, run_id: str, step: str, inputs: dict[str, Any], outputs: dict[str, Any], citations: list[str]) -> None:
        """Log one agent step."""

        if self.mlflow is None:
            self._write({"event": "step", "run_id": run_id, "step": step, "inputs": inputs, "outputs": outputs, "citations": citations})
            return
        active = self.mlflow.active_run()
        if active and active.info.run_id == run_id:
            self._log_nested_step(step, inputs, outputs, citations)
            return
        with self.mlflow.start_run(run_id=run_id, nested=True):
            self._log_nested_step(step, inputs, outputs, citations)

    def _log_nested_step(self, step: str, inputs: dict[str, Any], outputs: dict[str, Any], citations: list[str]) -> None:
        """Log a child step under whichever parent run is currently active."""

        with self.mlflow.start_run(run_name=step, nested=True, tags={"citations": json.dumps(citations)}):
            self.mlflow.log_dict(inputs, "inputs.json")
            self.mlflow.log_dict(outputs, "outputs.json")
            self.mlflow.log_metric("rows_referenced", len(set(citations)))

    def _write(self, payload: dict[str, Any]) -> None:
        try:
            with self.fallback_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "virtue_mlflow_trace_fallback.jsonl"
            with fallback.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
            self.fallback_path = fallback
