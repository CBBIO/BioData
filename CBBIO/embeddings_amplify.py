"""Compatibility wrapper for :mod:`CBBIO.embeddings.models.amplify`."""

from __future__ import annotations

from .embeddings.models.amplify import (
    AMPLIFY_HF_MODEL_NAMES,
    AmplifyPreprocessor,
    AmplifyTokenizerAdapter,
    AmplifyModelAdapter,
    AmplifyPostprocessor,
    AmplifyEmbeddingGenerator,
)

__all__ = [
    "AMPLIFY_HF_MODEL_NAMES",
    "AmplifyPreprocessor",
    "AmplifyTokenizerAdapter",
    "AmplifyModelAdapter",
    "AmplifyPostprocessor",
    "AmplifyEmbeddingGenerator",
]
