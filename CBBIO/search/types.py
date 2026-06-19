"""Search subsystem state and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from ..types import DistanceMetric


ResolvedSearchBackend = Literal["pgvector"] | Literal["faiss_cpu"] | Literal["faiss_gpu"] | Literal["cuvs_gpu"] | Literal["torch_gpu"]

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
    """Detected availability of local search backends and devices."""
    faiss_gpu: bool
    torch_gpu: bool
    preferred_device: str | None
    torch_device: str | None
    faiss_device: str | None
    hardware_class: str
    faiss_cpu: bool = False
    cuvs_gpu: bool = False
    cuvs_device: str | None = None


@dataclass
class ResolvedBackend:
    """Effective search backend choice for one workload."""
    backend: ResolvedSearchBackend
    device: str | None
    ann_requested: bool
    ann_used: bool
    degraded: bool
    reason: str
    batch_size: int
    resident: bool
    hardware_class: str
    chunk_size: int | None = None
    estimated_bytes: int | None = None
    free_bytes: int | None = None


@dataclass
class GpuSearchState:
    """Loaded accelerated search state for one embedding workload."""
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
]
