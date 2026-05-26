"""Databricks/OpenAI-compatible LLM client."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import request


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM response."""

    text: str
    provider: str
    model: str
    used_fallback: bool


class DatabricksLLMClient:
    """Credential-ready client for Databricks-hosted or OpenAI-compatible chat endpoints."""

    def __init__(self, endpoint_url: str | None = None, token: str | None = None, model: str | None = None) -> None:
        self.endpoint_url = endpoint_url or os.getenv("DATABRICKS_LLM_ENDPOINT") or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        self.token = token or os.getenv("DATABRICKS_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("DATABRICKS_LLM_MODEL") or "databricks-meta-llama-3-1-70b-instruct"

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.0) -> LLMResponse:
        """Call the configured endpoint, or return a transparent deterministic fallback."""

        if not self.endpoint_url or not self.token:
            content = messages[-1]["content"] if messages else ""
            return LLMResponse(
                text=f"Local fallback reasoning: {content[:500]}",
                provider="local-fallback",
                model=self.model,
                used_fallback=True,
            )
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint_url,
            data=data,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return LLMResponse(text=text, provider="databricks-openai-compatible", model=self.model, used_fallback=False)

