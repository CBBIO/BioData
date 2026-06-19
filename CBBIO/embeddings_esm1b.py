"""Compatibility wrapper for :mod:`CBBIO.embeddings.models.esm1b`."""

from __future__ import annotations

from .embeddings.models.esm1b import (
    ESM1B_MODEL_ALIASES,
    Esm1bPreprocessor,
    Esm1bTokenizerAdapter,
    Esm1bModelAdapter,
    Esm1bPostprocessor,
    Esm1bEmbeddingGenerator,
)

__all__ = [
    "ESM1B_MODEL_ALIASES",
    "Esm1bPreprocessor",
    "Esm1bTokenizerAdapter",
    "Esm1bModelAdapter",
    "Esm1bPostprocessor",
    "Esm1bEmbeddingGenerator",
]
