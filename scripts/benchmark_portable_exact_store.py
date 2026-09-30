#!/usr/bin/env python3
"""Benchmark exact local-search backends from a portable exact store.

The command never opens PostgreSQL. It selects deterministic query vectors from
an exact store, then measures a cold and warm call for each exact backend.
"""

from __future__ import annotations

import argparse
import json
import logging
from math import sqrt
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from CBBIO import BioDataClient, DistanceMetric, IndexManager, Neighbor, SearchBackend


LOGGER = logging.getLogger(__name__)
DEFAULT_BACKENDS: tuple[SearchBackend, ...] = ("faiss_cpu", "cuvs_gpu")
_SUPPORTED_BACKENDS = {"faiss_cpu", "cuvs_gpu", "cuvs_streaming", "torch_gpu"}


def _parse_args() -> argparse.Namespace:
    """Parse command-line options for the offline exact-store benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark exact FAISS, cuVS, or Torch search using a portable exact store.",
    )
    parser.add_argument("--index-root", type=Path, required=True, help="Copied .biodata/indexes directory.")
    parser.add_argument("--database-label", default="biodata-nas", help="Artifact label. Default: biodata-nas.")
    parser.add_argument("--embedding-type-id", type=int, default=3, help="Embedding type. Default: 3.")
    parser.add_argument("--layer-index", type=int, default=0, help="Embedding layer. Default: 0.")
    parser.add_argument(
        "--metric",
        choices=("cosine", "l2", "inner_product"),
        default="cosine",
        help="Exact distance metric. Default: cosine.",
    )
    parser.add_argument("--query-count", type=int, default=10, help="Queries selected from the store. Default: 10.")
    parser.add_argument("--query-seed", type=int, default=20260919, help="Row sampling seed. Default: 20260919.")
    parser.add_argument("--top-k", type=int, default=100, help="Distinct protein neighbors. Default: 100.")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=sorted(_SUPPORTED_BACKENDS),
        default=list(DEFAULT_BACKENDS),
        help="Exact backends to test. Default: faiss_cpu cuvs_gpu.",
    )
    parser.add_argument("--device", default="cuda:0", help="GPU device. Default: cuda:0.")
    parser.add_argument("--faiss-threads", type=int, default=None, help="Optional FAISS CPU worker count.")
    parser.add_argument(
        "--cuvs-streaming-block-size",
        type=int,
        default=None,
        help="Vectors per cuVS streaming block; default: automatic free-VRAM sizing.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON results path.")
    args = parser.parse_args()
    if args.query_count < 1:
        parser.error("--query-count must be positive")
    if args.query_seed < 0:
        parser.error("--query-seed must be non-negative")
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.faiss_threads is not None and args.faiss_threads < 1:
        parser.error("--faiss-threads must be positive")
    if args.cuvs_streaming_block_size is not None and args.cuvs_streaming_block_size < 1:
        parser.error("--cuvs-streaming-block-size must be positive")
    return args


def _import_numpy() -> Any:
    """Load NumPy with a clear error for incomplete cluster environments."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for the portable exact-store benchmark.") from exc
    return np


def _configure_faiss_threads(thread_count: int | None) -> None:
    """Set FAISS CPU workers when explicitly requested."""
    if thread_count is None:
        return
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("FAISS is required when --faiss-threads is specified.") from exc
    faiss.omp_set_num_threads(thread_count)


def _protein_ids_for_rows(metadata_path: Path, row_indices: Sequence[int]) -> dict[int, str]:
    """Read protein IDs for selected exact-store row indices."""
    placeholders = ", ".join("?" for _ in row_indices)
    with sqlite3.connect(metadata_path) as connection:
        rows = connection.execute(
            f"SELECT row_index, protein_id FROM vector_rows WHERE row_index IN ({placeholders})",
            tuple(row_indices),
        ).fetchall()
    protein_ids = {int(row_index): str(protein_id) for row_index, protein_id in rows}
    missing_rows = sorted(set(row_indices).difference(protein_ids))
    if missing_rows:
        raise RuntimeError(f"Exact-store metadata is missing row indices: {missing_rows[:5]}.")
    return protein_ids


def _query_rows(
    manager: IndexManager,
    *,
    database_label: str,
    embedding_type_id: int,
    layer_index: int,
    query_count: int,
    query_seed: int,
) -> tuple[dict[str, Any], list[dict[str, int | str]]]:
    """Select deterministic, uniquely labelled external queries from an artifact."""
    inspection = manager.load_exact_store(
        database_label=database_label,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
    )
    manifest = inspection.manifest
    if manifest is None:
        raise RuntimeError("The loaded exact store has no manifest.")
    if query_count > manifest.vector_count:
        raise ValueError(f"query_count={query_count} exceeds store size {manifest.vector_count}.")
    np = _import_numpy()
    row_indices = sorted(
        int(value)
        for value in np.random.default_rng(query_seed).choice(
            manifest.vector_count,
            size=query_count,
            replace=False,
        )
    )
    protein_ids = _protein_ids_for_rows(inspection.artifact.metadata_path, row_indices)
    matrix = np.memmap(
        inspection.artifact.vectors_path,
        dtype=np.float16,
        mode="r",
        shape=(manifest.vector_count, manifest.dimension),
    )
    embeddings: dict[str, Any] = {}
    metadata: list[dict[str, int | str]] = []
    for row_index in row_indices:
        protein_id = protein_ids[row_index]
        # A row-qualified label keeps duplicate stored protein IDs separate as external queries.
        query_id = f"{protein_id}@row-{row_index}"
        embeddings[query_id] = np.asarray(matrix[row_index], dtype=np.float32)
        metadata.append({"query_id": query_id, "protein_id": protein_id, "row_index": row_index})
    return embeddings, metadata


def _serialize_neighbors(
    results: Mapping[str, Sequence[Neighbor]],
) -> dict[str, list[dict[str, float | int | str]]]:
    """Convert public neighbor values into JSON-compatible data."""
    return {
        query_id: [
            {
                "protein_id": neighbor.protein_id,
                "layer_index": neighbor.layer_index,
                "distance": float(neighbor.distance),
            }
            for neighbor in neighbors
        ]
        for query_id, neighbors in results.items()
    }


def _neighbors_from_serialized(
    groups: Mapping[str, Sequence[Mapping[str, float | int | str]]],
) -> dict[str, list[Neighbor]]:
    """Reconstruct neighbor objects for cross-backend comparisons."""
    return {
        query_id: [
            Neighbor(
                protein_id=str(neighbor["protein_id"]),
                layer_index=int(neighbor["layer_index"]),
                distance=float(neighbor["distance"]),
            )
            for neighbor in neighbors
        ]
        for query_id, neighbors in groups.items()
    }


def _quality_summary(
    reference: Mapping[str, Sequence[Neighbor]],
    observed: Mapping[str, Sequence[Neighbor]],
    *,
    top_k: int,
) -> dict[str, float]:
    """Compare ranking overlap and shared-neighbor distances with FAISS CPU."""
    rows: list[dict[str, float]] = []
    for query_id, reference_neighbors in reference.items():
        observed_neighbors = observed[query_id]
        reference_by_protein = {neighbor.protein_id: float(neighbor.distance) for neighbor in reference_neighbors}
        observed_by_protein = {neighbor.protein_id: float(neighbor.distance) for neighbor in observed_neighbors}
        shared_proteins = set(reference_by_protein).intersection(observed_by_protein)
        squared_errors = [
            (observed_by_protein[protein_id] - reference_by_protein[protein_id]) ** 2
            for protein_id in shared_proteins
        ]
        row = {
            "shared_neighbors": float(len(shared_proteins)),
            "rmsd_distance": sqrt(sum(squared_errors) / len(squared_errors)) if squared_errors else float("nan"),
        }
        for cutoff in (10, 50, 100):
            if cutoff > top_k:
                continue
            expected = {neighbor.protein_id for neighbor in reference_neighbors[:cutoff]}
            actual = {neighbor.protein_id for neighbor in observed_neighbors[:cutoff]}
            row[f"recall_at_{cutoff}"] = len(expected.intersection(actual)) / cutoff
        rows.append(row)

    summary = {
        "mean_shared_neighbors": sum(row["shared_neighbors"] for row in rows) / len(rows),
        "mean_rmsd_distance": sum(row["rmsd_distance"] for row in rows) / len(rows),
    }
    for cutoff in (10, 50, 100):
        key = f"recall_at_{cutoff}"
        if key not in rows[0]:
            continue
        values = [row[key] for row in rows]
        summary[f"mean_recall_at_{cutoff}"] = sum(values) / len(values)
        if cutoff == 100:
            summary["min_recall_at_100"] = min(values)
    return summary


def _run_backend(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    query_embeddings: Mapping[str, Any],
    embedding_type_id: int,
    layer_index: int,
    metric: DistanceMetric,
    top_k: int,
    device: str,
    cuvs_streaming_block_size: int | None,
) -> tuple[dict[str, Any], dict[str, list[dict[str, float | int | str]]]]:
    """Measure first and second calls for one backend."""
    kwargs: dict[str, Any] = {
        "embedding_type_id": embedding_type_id,
        "layer_index": layer_index,
        "k": top_k,
        "metric": metric,
        "backend": backend,
        "use_ann": False,
    }
    if backend in {"cuvs_gpu", "cuvs_streaming", "torch_gpu"}:
        kwargs["device"] = device
    if backend == "cuvs_streaming":
        kwargs["cuvs_streaming_block_size"] = cuvs_streaming_block_size
    try:
        cold_started_at = time.perf_counter()
        cold_results = client.find_nearest_neighbors_for_embeddings(query_embeddings, **kwargs)
        cold_seconds = time.perf_counter() - cold_started_at
        warm_started_at = time.perf_counter()
        warm_results = client.find_nearest_neighbors_for_embeddings(query_embeddings, **kwargs)
        warm_seconds = time.perf_counter() - warm_started_at
    except Exception as exc:  # Preserve successful results when a backend is unavailable.
        LOGGER.exception("Backend %s failed.", backend)
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}, {}
    if _serialize_neighbors(cold_results) != _serialize_neighbors(warm_results):
        LOGGER.warning("Backend %s returned different cold and warm result payloads.", backend)
    return {
        "status": "ok",
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "materialization_seconds_estimate": max(0.0, cold_seconds - warm_seconds),
        "query_count": len(query_embeddings),
        "top_k": top_k,
        "diagnostics": client.last_search_diagnostics,
    }, _serialize_neighbors(warm_results)


def _main() -> int:
    """Run the portable exact-store benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    _configure_faiss_threads(args.faiss_threads)
    metric = cast(DistanceMetric, args.metric)
    manager = IndexManager(args.index_root)
    query_embeddings, query_metadata = _query_rows(
        manager,
        database_label=args.database_label,
        embedding_type_id=args.embedding_type_id,
        layer_index=args.layer_index,
        query_count=args.query_count,
        query_seed=args.query_seed,
    )
    inspection = manager.load_exact_store(
        database_label=args.database_label,
        embedding_type_id=args.embedding_type_id,
        layer_index=args.layer_index,
    )
    if inspection.manifest is None:
        raise RuntimeError("The loaded exact store has no manifest.")
    LOGGER.info(
        "Loaded portable exact store: %s vectors, dimension %d, %d queries.",
        f"{inspection.manifest.vector_count:,}",
        inspection.manifest.dimension,
        len(query_embeddings),
    )

    client = BioDataClient(index_manager=manager, index_database_label=args.database_label)
    backend_results: dict[str, dict[str, Any]] = {}
    neighbor_results: dict[str, dict[str, list[dict[str, float | int | str]]]] = {}
    for backend_name in args.backends:
        backend = cast(SearchBackend, backend_name)
        LOGGER.info("Running %s cold and warm calls.", backend)
        result, neighbors = _run_backend(
            client,
            backend=backend,
            query_embeddings=query_embeddings,
            embedding_type_id=args.embedding_type_id,
            layer_index=args.layer_index,
            metric=metric,
            top_k=args.top_k,
            device=args.device,
            cuvs_streaming_block_size=args.cuvs_streaming_block_size,
        )
        backend_results[backend] = result
        if neighbors:
            neighbor_results[backend] = neighbors

    reference = neighbor_results.get("faiss_cpu")
    if reference is not None:
        reference_neighbors = _neighbors_from_serialized(reference)
        for backend, neighbors in neighbor_results.items():
            backend_results[backend]["comparison_to_faiss_cpu"] = _quality_summary(
                reference_neighbors,
                _neighbors_from_serialized(neighbors),
                top_k=args.top_k,
            )

    payload = {
        "artifact": {
            "directory": str(inspection.artifact.directory),
            "vector_count": inspection.manifest.vector_count,
            "dimension": inspection.manifest.dimension,
            "source_revision": inspection.manifest.source_revision,
        },
        "parameters": {
            "database_label": args.database_label,
            "embedding_type_id": args.embedding_type_id,
            "layer_index": args.layer_index,
            "metric": metric,
            "query_count": args.query_count,
            "query_seed": args.query_seed,
            "top_k": args.top_k,
            "device": args.device,
            "cuvs_streaming_block_size": args.cuvs_streaming_block_size,
        },
        "queries": query_metadata,
        "backends": backend_results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    LOGGER.info("Benchmark summary:\n%s", rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(f"{rendered}\n", encoding="utf-8")
        LOGGER.info("Wrote JSON results to %s", args.json_out)
    return 0 if all(result["status"] == "ok" for result in backend_results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
