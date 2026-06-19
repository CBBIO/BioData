"""Compatibility wrapper for :mod:`CBBIO.embeddings.utils.transformers`."""

from __future__ import annotations

from .utils.transformers import (
    load_esm_tokenizer,
    suppress_esm_tokenizer_class_warning,
)

__all__ = [
    "load_esm_tokenizer",
    "suppress_esm_tokenizer_class_warning",
]
