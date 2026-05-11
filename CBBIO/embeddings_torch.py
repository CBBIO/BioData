"""Shared PyTorch helpers for embedding generators."""

from __future__ import annotations

from importlib import metadata as importlib_metadata
import re
from typing import Any, Sequence, cast

from .embeddings import (
    EmbeddingBackendError,
    EmbeddingDependencyError,
    EmbeddingInputError,
    EmbeddingPayload,
    PostprocessorAdapter,
    PreprocessorAdapter,
    as_float_matrix,
    validate_sequence,
)


def normalize_torch_dtype_name(dtype: str | None) -> str | None:
    if dtype is None:
        return None
    value = str(dtype).strip().lower()
    aliases = {
        "fp32": "float32",
        "float": "float32",
        "float32": "float32",
        "fp16": "float16",
        "half": "float16",
        "float16": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    normalized = aliases.get(value)
    if normalized is None:
        raise EmbeddingInputError("dtype must be one of: float32, float16, bfloat16.")
    return normalized


def resolve_torch_dtype(dtype: str) -> Any:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise EmbeddingDependencyError(
            "PyTorch is required to set model dtype. Install with: pip install torch"
        ) from exc

    dtype_obj = getattr(torch, dtype, None)
    if dtype_obj is None:
        raise EmbeddingBackendError(f"Current PyTorch build does not expose torch.{dtype}.")
    return dtype_obj


def move_model_to_device(
    model: Any,
    *,
    device: str,
    dtype: Any | None = None,
    dtype_name: str | None = None,
) -> None:
    to_fn = getattr(model, "to", None)
    if not callable(to_fn):
        return

    if dtype is None:
        to_fn(str(device))
        return

    try:
        to_fn(device=str(device), dtype=dtype)
        return
    except TypeError:
        pass

    try:
        to_fn(str(device), dtype=dtype)
        return
    except TypeError:
        pass

    to_fn(str(device))
    _apply_model_dtype_method(model, dtype_name=dtype_name)


def _apply_model_dtype_method(model: Any, *, dtype_name: str | None) -> None:
    if dtype_name == "float16":
        method_name = "half"
    elif dtype_name == "float32":
        method_name = "float"
    elif dtype_name == "bfloat16":
        method_name = "bfloat16"
    else:
        raise EmbeddingBackendError("Could not apply unknown torch dtype to model.")

    method = getattr(model, method_name, None)
    if not callable(method):
        raise EmbeddingBackendError(
            f"Model does not support dtype conversion through .to(..., dtype=...) or .{method_name}()."
        )
    method()


class BasePreprocessor(PreprocessorAdapter):
    """Shared preprocessor for protein sequence inputs.

    Handles: strip/upper/space-removal, UZOB→X substitution, optional
    inter-residue spacing, and an optional prefix string.
    """

    def __init__(
        self,
        *,
        context: str,
        spacing: bool = False,
        prefix: str | None = None,
        prefix_space: bool = True,
    ) -> None:
        self._context = context
        self._spacing = spacing
        self._prefix = prefix
        self._prefix_space = prefix_space

    def preprocess(self, raw_sequence: str) -> str:
        sequence = str(raw_sequence).strip().upper().replace(" ", "")
        validate_sequence(sequence, context=self._context)
        replaced = re.sub(r"[UZOB]", "X", sequence)
        if self._spacing:
            replaced = " ".join(list(replaced))
        if self._prefix is not None:
            sep = " " if self._prefix_space else ""
            replaced = f"{self._prefix}{sep}{replaced}"
        return replaced


class DefaultPostprocessor(PostprocessorAdapter):
    """Shared postprocessor for all models that return a standard {layers: {int: tensor}} dict."""

    def postprocess(self, model_output: Any) -> EmbeddingPayload:
        name = self.__class__.__name__
        if not isinstance(model_output, dict):
            raise EmbeddingBackendError(f"{name} expects a dict payload from model adapter.")
        model_output_map = cast(dict[str, Any], model_output)
        layers_obj_raw = model_output_map.get("layers")
        if not isinstance(layers_obj_raw, dict):
            raise EmbeddingBackendError(f"{name} expects a 'layers' dict in model output.")
        layers_obj = cast(dict[int, Any], layers_obj_raw)
        if not layers_obj:
            raise EmbeddingBackendError(f"{name} received no layers.")
        first_key = sorted(layers_obj.keys())[0]
        return as_float_matrix(layers_obj[first_key])


def normalize_requested_layers(layer_index: int | Sequence[int] | None) -> list[int] | None:
    """Normalise layer_index into a flat list, or None for all layers."""
    if layer_index is None:
        return None
    if isinstance(layer_index, int):
        return [int(layer_index)]
    return [int(value) for value in layer_index]


def framework_versions(*packages: str) -> dict[str, str]:
    """Return {package: version} for each package that is installed."""
    versions: dict[str, str] = {}
    for package_name in packages:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def extract_name_or_path(obj: Any) -> str | None:
    """Return the name_or_path attribute of a HuggingFace config/model, or None."""
    value = getattr(obj, "name_or_path", None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def extract_revision(obj: Any) -> str | None:
    """Return the _commit_hash from a HuggingFace model's config, or None."""
    config = getattr(obj, "config", None)
    if config is None:
        return None
    revision = getattr(config, "_commit_hash", None)
    if revision is None:
        return None
    text = str(revision).strip()
    return text or None


def infer_total_layers_from_model(model: Any) -> int | None:
    """Infer hidden-state count (encoder blocks + 1) from a HuggingFace model config."""
    config = getattr(model, "config", None)
    if config is None:
        return None
    num_layers = getattr(config, "num_layers", None)
    if isinstance(num_layers, int) and num_layers >= 1:
        return int(num_layers) + 1
    num_hidden = getattr(config, "num_hidden_layers", None)
    if isinstance(num_hidden, int) and num_hidden >= 1:
        return int(num_hidden) + 1
    return None


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
