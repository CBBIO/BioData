"""Compatibility wrapper for :mod:`CBBIO.embeddings.models.ankh3`."""

from __future__ import annotations

from .embeddings.models.ankh3 import (
    Ankh3Preprocessor,
    Ankh3TokenizerAdapter,
    Ankh3ModelAdapter,
    Ankh3Postprocessor,
    Ankh3EmbeddingGenerator,
)

__all__ = [
    "Ankh3Preprocessor",
    "Ankh3TokenizerAdapter",
    "Ankh3ModelAdapter",
    "Ankh3Postprocessor",
    "Ankh3EmbeddingGenerator",
]
