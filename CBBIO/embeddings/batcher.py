"""Compatibility wrapper for :mod:`CBBIO.embeddings.utils.batcher`."""

from __future__ import annotations

from .utils.batcher import (
    FastaBatcher,
    IterableBatcher,
)

__all__ = [
    "FastaBatcher",
    "IterableBatcher",
]
