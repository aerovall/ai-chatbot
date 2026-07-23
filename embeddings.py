"""Semantic embedding support for meaning-based answer caching.

Wraps the Voyage AI embeddings API so the bot can tell when two differently
worded questions mean the same thing (e.g. "which firm pays out fastest?" and
"who has the quickest withdrawals?").

The client is intentionally fail-safe: if the ``voyageai`` package or an API key
is unavailable, or a request errors, embedding calls return ``None`` and the
caller transparently falls back to exact-match caching. Semantic caching is a
cost optimisation, never a hard dependency.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("tickshift.embeddings")


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Return the cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in the range [-1.0, 1.0]; ``0.0`` if either vector is
        empty, zero-length or a length mismatch.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


class EmbeddingClient:
    """Thin wrapper around the Voyage AI embeddings API."""

    def __init__(self, api_key: str, model: str = "voyage-3.5") -> None:
        """Create the embedding client.

        Args:
            api_key: Voyage AI API key.
            model: Voyage embedding model name.

        Raises:
            RuntimeError: If the ``voyageai`` package is not installed.
        """
        try:
            import voyageai  # Imported lazily so the bot runs without it.
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise RuntimeError(
                "The 'voyageai' package is required for semantic caching. "
                "Install it with `pip install voyageai`."
            ) from exc

        self._model = model
        self._client = voyageai.Client(api_key=api_key)
        logger.info("Voyage embedding client initialised (model=%s).", model)

    def embed(self, text: str, input_type: str = "query") -> Optional[List[float]]:
        """Embed a single piece of text, returning ``None`` on any failure.

        Args:
            text: The text to embed.
            input_type: Voyage input type hint (``"query"`` or ``"document"``).

        Returns:
            The embedding vector, or ``None`` if embedding failed.
        """
        text = (text or "").strip()
        if not text:
            return None
        try:
            result = self._client.embed(
                [text], model=self._model, input_type=input_type
            )
            return list(result.embeddings[0])
        except Exception:  # noqa: BLE001 - never let embedding errors bubble up.
            logger.exception("Voyage embedding request failed.")
            return None

    def health_check(self) -> bool:
        """Return whether the embedding API is reachable and configured."""
        return self.embed("health check", input_type="query") is not None
