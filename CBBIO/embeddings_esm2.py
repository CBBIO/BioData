"""Compatibility wrapper for :mod:`CBBIO.embeddings.models.esm2`."""

from __future__ import annotations

from .embeddings.models.esm2 import (
    ESM2_HF_MODEL_NAMES,
    ESM2_LAYER_COUNTS,
    Esm2Preprocessor,
    Esm2TokenizerAdapter,
    Esm2ModelAdapter,
    Esm2Postprocessor,
    Esm2EmbeddingGenerator,
)

__all__ = [
    "ESM2_HF_MODEL_NAMES",
    "ESM2_LAYER_COUNTS",
    "Esm2Preprocessor",
    "Esm2TokenizerAdapter",
    "Esm2ModelAdapter",
    "Esm2Postprocessor",
    "Esm2EmbeddingGenerator",
]
