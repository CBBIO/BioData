"""Shared domain types for CBBIO modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypeAlias


ProteinID: TypeAlias = str
GOID: TypeAlias = str
StructureID: TypeAlias = str

DistanceMetric: TypeAlias = Literal["l2", "cosine", "inner_product"]
SimilarityMethod: TypeAlias = Literal["resnik", "lin", "schlicker", "wang"]
EmbeddingModel: TypeAlias = int | str
EmbeddingVector: TypeAlias = Sequence[float]
SearchBackend: TypeAlias = Literal["auto", "gpu", "pgvector", "faiss_cpu", "faiss_gpu", "cuvs_gpu", "torch_gpu"]


@dataclass(frozen=True)
class EmbeddingType:
    id: int
    name: str
    model_name: str | None
    task_name: str | None
    description: str | None


@dataclass(frozen=True)
class Neighbor:
    protein_id: str
    layer_index: int
    distance: float


@dataclass(frozen=True)
class GOAnnotation:
    go_id: str
    category: str
    description: str
    evidence_code: str


__all__ = [
    "ProteinID",
    "GOID",
    "StructureID",
    "DistanceMetric",
    "SimilarityMethod",
    "EmbeddingModel",
    "EmbeddingVector",
    "SearchBackend",
    "EmbeddingType",
    "Neighbor",
    "GOAnnotation",
]
