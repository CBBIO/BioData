"""Compatibility wrapper for :mod:`CBBIO.embeddings.factory`."""

from __future__ import annotations

from .embeddings.factory import (
    Generator,
    available_generator_classes,
    available_generator_models,
)

__all__ = [
    "Generator",
    "available_generator_classes",
    "available_generator_models",
]
