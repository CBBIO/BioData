"""Utility helpers shared by search backends."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, cast

from ..BioData import DriverDependencyError
from ..types import DistanceMetric


def _coerce_vector_row(value: Any) -> Any:
    to_list_attr = getattr(value, "to_list", None)
    if callable(to_list_attr):
        try:
            return to_list_attr()
        except Exception:
            pass

    tolist_attr = getattr(value, "tolist", None)
    if callable(tolist_attr):
        try:
            return tolist_attr()
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        return list(value)

    try:
        return list(cast(Iterable[Any], value))
    except TypeError:
        return value


def _as_numpy_matrix(values: Any) -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise DriverDependencyError("NumPy is required for GPU search backends. Install with: pip install numpy") from exc

    matrix = np.asarray([_coerce_vector_row(value) for value in values], dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("Expected a 2D embedding matrix with at least one column.")
    return matrix


def _prepare_index_vectors(vectors: Any, *, metric: DistanceMetric) -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise DriverDependencyError("NumPy is required for GPU search backends. Install with: pip install numpy") from exc

    matrix = np.asarray(vectors, dtype=np.float32)
    if metric != "cosine":
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _normalize_distance(*, metric: DistanceMetric, value: Any, l2_squared: bool) -> float:
    distance = float(value)
    if metric == "inner_product":
        return -distance
    if metric == "cosine":
        return 1.0 - distance
    if not l2_squared:
        return distance
    if distance < 0.0:
        return 0.0
    return distance ** 0.5 if distance > 0.0 else 0.0


def _import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise DriverDependencyError("Torch is required for torch_gpu search backend. Install with: pip install torch") from exc
    return torch


def _import_faiss(*, allow_missing: bool = False) -> Any:
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        if allow_missing:
            return None
        raise DriverDependencyError("FAISS is required for faiss_gpu search backend. Install with: pip install faiss-cpu/faiss-gpu") from exc
    return faiss


def _preferred_torch_device(device: Optional[str]) -> Optional[str]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return None

    requested = str(device or "").strip().lower()
    if requested:
        if requested.startswith("cuda") and bool(getattr(torch.cuda, "is_available", lambda: False)()):
            return requested
        if requested == "mps":
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if bool(getattr(mps_backend, "is_available", lambda: False)()):
                return "mps"
        return None

    if bool(getattr(torch.cuda, "is_available", lambda: False)()):
        return "cuda:0"
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if bool(getattr(mps_backend, "is_available", lambda: False)()):
        return "mps"
    return None


def _preferred_faiss_device(device: Optional[str]) -> Optional[str]:
    requested = str(device or "").strip().lower()
    if requested == "mps":
        return None
    if requested.startswith("cuda"):
        return requested
    faiss = _import_faiss(allow_missing=True)
    if faiss is None:
        return None
    get_num_gpus = getattr(faiss, "get_num_gpus", None)
    if callable(get_num_gpus):
        try:
            if int(get_num_gpus()) > 0:
                return "cuda:0"
        except Exception:
            return None
    return None


def _cuda_device_index(device: str) -> int:
    lowered = str(device).strip().lower()
    if not lowered.startswith("cuda"):
        return 0
    _, _, suffix = lowered.partition(":")
    if not suffix:
        return 0
    try:
        return max(0, int(suffix))
    except ValueError:
        return 0


def _torch_normalize(tensor: Any, *, torch: Any) -> Any:
    norms = torch.linalg.norm(tensor, dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    return tensor / norms


def _tensor_to_list(value: Any) -> List[Any]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        result = value.tolist()
        if isinstance(result, list):
            return result
    if isinstance(value, list):
        return value
    return list(value)


__all__ = [
    "_coerce_vector_row",
    "_as_numpy_matrix",
    "_prepare_index_vectors",
    "_normalize_distance",
    "_import_torch",
    "_import_faiss",
    "_preferred_torch_device",
    "_preferred_faiss_device",
    "_cuda_device_index",
    "_torch_normalize",
    "_tensor_to_list",
]
