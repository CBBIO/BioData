"""Search subsystem state and defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

from ..types import DistanceMetric


ResolvedSearchBackend = Union[Literal["pgvector"], Literal["faiss_gpu"], Literal["torch_gpu"]]

DEFAULT_BACKEND_THRESHOLDS: Dict[str, Dict[str, int]] = {
    "cuda": {
        "faiss_gpu_min_batch": 8,
        "torch_gpu_min_batch": 16,
        "resident_gpu_min_batch": 1,
    },
    "mps": {
        "faiss_gpu_min_batch": 1_000_000,
        "torch_gpu_min_batch": 8,
        "resident_gpu_min_batch": 1,
    },
    "cpu": {
        "faiss_gpu_min_batch": 1_000_000,
        "torch_gpu_min_batch": 1_000_000,
        "resident_gpu_min_batch": 1_000_000,
    },
}


@dataclass
class _BackendAvailability:
    faiss_gpu: bool
    torch_gpu: bool
    preferred_device: Optional[str]
    torch_device: Optional[str]
    faiss_device: Optional[str]
    hardware_class: str


@dataclass
class _ResolvedBackend:
    backend: ResolvedSearchBackend
    device: Optional[str]
    ann_requested: bool
    ann_used: bool
    degraded: bool
    reason: str
    batch_size: int
    resident: bool
    hardware_class: str


@dataclass
class _GpuSearchState:
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


__all__ = [
    "ResolvedSearchBackend",
    "DEFAULT_BACKEND_THRESHOLDS",
    "_BackendAvailability",
    "_ResolvedBackend",
    "_GpuSearchState",
]
