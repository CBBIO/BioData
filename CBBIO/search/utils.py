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
        return list(cast(Iterable[object], value))

    try:
        return list(cast(Iterable[object], value))
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


def _import_cupy(*, allow_missing: bool = False) -> Any:
    try:
        import cupy  # type: ignore
    except ModuleNotFoundError as exc:
        if allow_missing:
            return None
        raise DriverDependencyError("CuPy is required for cuvs_gpu search backend. Install with: pip install cupy-cuda12x") from exc
    return cupy


def _import_cuvs(*, allow_missing: bool = False) -> Any:
    try:
        import cuvs  # type: ignore
    except ModuleNotFoundError as exc:
        if allow_missing:
            return None
        raise DriverDependencyError("cuVS is required for cuvs_gpu search backend. Install with: pip install cuvs-cu12") from exc
    return cuvs


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
            if int(cast(Any, get_num_gpus)()) > 0:
                return "cuda:0"
        except Exception:
            return None
    return None


def _preferred_cuvs_device(device: Optional[str]) -> Optional[str]:
    requested = str(device or "").strip().lower()
    if requested == "mps":
        return None

    cupy = _import_cupy(allow_missing=True)
    if cupy is None or _import_cuvs(allow_missing=True) is None:
        return None

    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        return None
    if device_count < 1:
        return None

    if requested.startswith("cuda"):
        return requested
    return "cuda:0"


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
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        result = tolist()
        if isinstance(result, list):
            return list(cast(Iterable[Any], result))
    if isinstance(value, list):
        return list(cast(Iterable[Any], value))
    return list(cast(Iterable[Any], value))


as_numpy_matrix = _as_numpy_matrix
prepare_index_vectors = _prepare_index_vectors
normalize_distance = _normalize_distance
import_torch = _import_torch
import_faiss = _import_faiss
import_cupy = _import_cupy
import_cuvs = _import_cuvs
preferred_torch_device = _preferred_torch_device
preferred_faiss_device = _preferred_faiss_device
preferred_cuvs_device = _preferred_cuvs_device
cuda_device_index = _cuda_device_index
torch_normalize = _torch_normalize
tensor_to_list = _tensor_to_list


__all__ = [
    "_coerce_vector_row",
    "_as_numpy_matrix",
    "_prepare_index_vectors",
    "_normalize_distance",
    "_import_torch",
    "_import_faiss",
    "_import_cupy",
    "_import_cuvs",
    "_preferred_torch_device",
    "_preferred_faiss_device",
    "_preferred_cuvs_device",
    "_cuda_device_index",
    "_torch_normalize",
    "_tensor_to_list",
    "as_numpy_matrix",
    "prepare_index_vectors",
    "normalize_distance",
    "import_torch",
    "import_faiss",
    "import_cupy",
    "import_cuvs",
    "preferred_torch_device",
    "preferred_faiss_device",
    "preferred_cuvs_device",
    "cuda_device_index",
    "torch_normalize",
    "tensor_to_list",
]
