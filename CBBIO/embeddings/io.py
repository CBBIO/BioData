"""Compatibility wrapper for :mod:`CBBIO.embeddings.utils.io`."""

from __future__ import annotations

from .utils.io import (
    load_embedding_records,
    H5EmbeddingReader,
    load_embedding_records_pickle,
    load_embedding_records_h5,
    load_embedding_records_npy,
    save_embedding_records_pickle,
    save_embedding_records_pickle_shards,
    save_embedding_records_npy,
    save_embedding_records_npy_shards,
    save_embedding_records_h5,
    generate_fasta_pickle_shards,
    generate_fasta_npy_shards,
    generate_fasta_h5,
    mean_pool_embedding_record,
)

__all__ = [
    "load_embedding_records",
    "H5EmbeddingReader",
    "load_embedding_records_pickle",
    "load_embedding_records_h5",
    "load_embedding_records_npy",
    "save_embedding_records_pickle",
    "save_embedding_records_pickle_shards",
    "save_embedding_records_npy",
    "save_embedding_records_npy_shards",
    "save_embedding_records_h5",
    "generate_fasta_pickle_shards",
    "generate_fasta_npy_shards",
    "generate_fasta_h5",
    "mean_pool_embedding_record",
]
