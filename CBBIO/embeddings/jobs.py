"""Compatibility wrapper for :mod:`CBBIO.embeddings.utils.jobs`."""

from __future__ import annotations

from .utils.jobs import (
    EmbeddingJobResult,
    ProgressCallback,
    run_embedding_generation,
)

__all__ = [
    "EmbeddingJobResult",
    "ProgressCallback",
    "run_embedding_generation",
]
