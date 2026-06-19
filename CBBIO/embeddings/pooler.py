"""Compatibility wrapper for :mod:`CBBIO.embeddings.utils.pooler`."""

from __future__ import annotations

from .utils.pooler import (
    ClsPooler,
    EmbeddingPooler,
    IdentityPooler,
    MeanPooler,
    PoolerInput,
    PoolerName,
    as_mean_pooled_vector,
    mean_pool_embedding_record,
    mean_pool_matrix,
    materialize_embedding_payload,
    pooler_factory,
    resolve_pooler,
)

__all__ = [
    "ClsPooler",
    "EmbeddingPooler",
    "IdentityPooler",
    "MeanPooler",
    "PoolerInput",
    "PoolerName",
    "as_mean_pooled_vector",
    "mean_pool_embedding_record",
    "mean_pool_matrix",
    "materialize_embedding_payload",
    "pooler_factory",
    "resolve_pooler",
]
