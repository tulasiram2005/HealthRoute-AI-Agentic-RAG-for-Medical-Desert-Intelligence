"""Embedding wrapper with sentence-transformers fallback."""

from __future__ import annotations

import hashlib
import math


class EmbeddingModel:
    """CPU-friendly embedding model with deterministic hash fallback."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
        except Exception:
            self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as float vectors."""

        if self._model is not None:
            return [list(map(float, vector)) for vector in self._model.encode(texts)]
        return [self._hash_embed(text) for text in texts]

    @staticmethod
    def _hash_embed(text: str, dim: int = 64) -> list[float]:
        values = [0.0] * dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = digest[0] % dim
            values[idx] += 1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [round(value / norm, 6) for value in values]

