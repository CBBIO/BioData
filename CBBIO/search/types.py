"""Search subsystem state and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from ..types import DistanceMetric


ResolvedSearchBackend = Union[Literal["pgvector"], Literal["faiss_cpu"], Literal["faiss_gpu"], Literal["cuvs_gpu"], Literal["torch_gpu"]]

DEFAULT_BACKEND_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "cuda": {
        "faiss_gpu_min_batch": 8,
        "cuvs_gpu_min_batch": 8,
        "torch_gpu_min_batch": 16,
        "resident_gpu_min_batch": 1,
    },
    "mps": {
        "faiss_gpu_min_batch": 1_000_000,
        "cuvs_gpu_min_batch": 1_000_000,
        "torch_gpu_min_batch": 8,
        "resident_gpu_min_batch": 1,
    },
    "cpu": {
        "faiss_gpu_min_batch": 1_000_000,
        "cuvs_gpu_min_batch": 1_000_000,
        "torch_gpu_min_batch": 1_000_000,
        "resident_gpu_min_batch": 1_000_000,
    },
}


@dataclass
class BackendAvailability:
    faiss_gpu: bool
    torch_gpu: bool
    preferred_device: Optional[str]
    torch_device: Optional[str]
    faiss_device: Optional[str]
    hardware_class: str
    faiss_cpu: bool = False
    cuvs_gpu: bool = False
    cuvs_device: Optional[str] = None


@dataclass
class ResolvedBackend:
    backend: ResolvedSearchBackend
    device: Optional[str]
    ann_requested: bool
    ann_used: bool
    degraded: bool
    reason: str
    batch_size: int
    resident: bool
    hardware_class: str
    chunk_size: Optional[int] = None
    estimated_bytes: Optional[int] = None
    free_bytes: Optional[int] = None


@dataclass
class GpuSearchState:
    backend: ResolvedSearchBackend
    embedding_type_id: int
    layer_index: int
    metric: DistanceMetric
    device: str
    ann_enabled: bool
    protein_ids: List[str]
    protein_rows: Dict[str, List[int]]
    vectors: Any
    faiss_index: Any = None
    faiss_resources: Any = None
    cuvs_index: Any = None


_BackendAvailability = BackendAvailability
_ResolvedBackend = ResolvedBackend
_GpuSearchState = GpuSearchState


__all__ = [
    "ResolvedSearchBackend",
    "DEFAULT_BACKEND_THRESHOLDS",
    "BackendAvailability",
    "ResolvedBackend",
    "GpuSearchState",
    "_BackendAvailability",
    "_ResolvedBackend",
    "_GpuSearchState",
]
