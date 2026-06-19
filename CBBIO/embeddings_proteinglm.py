"""Compatibility wrapper for :mod:`CBBIO.embeddings.models.proteinglm`."""

from __future__ import annotations

from .embeddings.models.proteinglm import (
    PROTEINGLM_HF_MODEL_NAMES,
    ProteinGlmPreprocessor,
    ProteinGlmTokenizerAdapter,
    ProteinGlmModelAdapter,
    ProteinGlmPostprocessor,
    ProteinGlmEmbeddingGenerator,
    proteinglm_sample_spans_from_attention_mask,
)

__all__ = [
    "PROTEINGLM_HF_MODEL_NAMES",
    "ProteinGlmPreprocessor",
    "ProteinGlmTokenizerAdapter",
    "ProteinGlmModelAdapter",
    "ProteinGlmPostprocessor",
    "ProteinGlmEmbeddingGenerator",
    "proteinglm_sample_spans_from_attention_mask",
]
