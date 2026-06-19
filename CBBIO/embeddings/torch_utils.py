"""Compatibility wrapper for :mod:`CBBIO.embeddings.utils.torch`."""

from __future__ import annotations

from .utils.torch import (
    BasePreprocessor,
    DefaultPostprocessor,
    normalize_torch_dtype_name,
    resolve_torch_dtype,
    move_model_to_device,
    normalize_requested_layers,
    framework_versions,
    extract_name_or_path,
    extract_revision,
    infer_total_layers_from_model,
)

__all__ = [
    "BasePreprocessor",
    "DefaultPostprocessor",
    "normalize_torch_dtype_name",
    "resolve_torch_dtype",
    "move_model_to_device",
    "normalize_requested_layers",
    "framework_versions",
    "extract_name_or_path",
    "extract_revision",
    "infer_total_layers_from_model",
]
