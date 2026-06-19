"""Compatibility wrapper for :mod:`CBBIO.embeddings.models.prott5`."""

from __future__ import annotations

from .embeddings.models.prott5 import (
    ProtT5Preprocessor,
    ProtT5TokenizerAdapter,
    ProtT5ModelAdapter,
    ProtT5Postprocessor,
    ProtT5EmbeddingGenerator,
)

__all__ = [
    "ProtT5Preprocessor",
    "ProtT5TokenizerAdapter",
    "ProtT5ModelAdapter",
    "ProtT5Postprocessor",
    "ProtT5EmbeddingGenerator",
]
