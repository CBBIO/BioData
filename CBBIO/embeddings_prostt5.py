"""Compatibility wrapper for :mod:`CBBIO.embeddings.models.prostt5`."""

from __future__ import annotations

from .embeddings.models.prostt5 import (
    ProstT5Preprocessor,
    ProstT5TokenizerAdapter,
    ProstT5ModelAdapter,
    ProstT5Postprocessor,
    ProstT5EmbeddingGenerator,
)

__all__ = [
    "ProstT5Preprocessor",
    "ProstT5TokenizerAdapter",
    "ProstT5ModelAdapter",
    "ProstT5Postprocessor",
    "ProstT5EmbeddingGenerator",
]
