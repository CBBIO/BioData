"""Search backend orchestration for BioData neighbor lookup."""

from __future__ import annotations

import uuid
import warnings
from typing import Any, Dict, Generator, List, Mapping, Sequence, Set, cast

from ..BioData import BioDataError, NotFoundError, cursor, embedding_dimension, metric_opclass, metric_operator
from ..types import DistanceMetric, Neighbor, SearchBackend
from .engines import (
    build_cuvs_streaming_search_state,
    build_faiss_streaming_search_state,
    build_search_state,
    neighbors_from_candidate_rows,
    search_cuvs_state,
    search_faiss_state,
    search_state,
    search_torch_state,
)
from .index_manager import IndexKey, SearchIndexError
from .types import (
    DEFAULT_BACKEND_THRESHOLDS,
    BackendAvailability,
    GpuSearchState,
    ResolvedBackend,
    ResolvedSearchBackend,
)
from .utils import (
    as_numpy_matrix,
    cuda_device_index,
    import_cupy,
    import_faiss,
    import_torch,
    preferred_cuvs_device,
    preferred_faiss_device,
    preferred_torch_device,
)


_CANONICAL_TIE_DECIMALS = 5
_CANONICAL_TIE_OVERFETCH = 64
_GPU_EXACT_LOAD_BATCH_SIZE = 10_000


def _canonicalize_neighbor_groups(
    grouped: Mapping[str, Sequence[Neighbor]],
    *,
    k: int,
) -> Dict[str, List[Neighbor]]:
    """Apply the cross-backend neighbor ordering contract."""
    return {
        str(query_id): sorted(
            neighbors,
            key=lambda neighbor: (
                round(float(neighbor.distance), _CANONICAL_TIE_DECIMALS),
                neighbor.protein_id,
            ),
        )[:k]
        for query_id, neighbors in grouped.items()
    }


class SearchService:
    """Encapsulates backend routing and neighbor-search implementations."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _per_query_exclusions(
        query_ids: Sequence[str],
        *,
        include_query: bool,
        excluded_protein_ids_by_query: Mapping[str, Set[str]] | None,
    ) -> Dict[str, Set[str]]:
        """Return local-search exclusions for every query."""
        if excluded_protein_ids_by_query is not None:
            return {
                str(query_id): set(excluded_protein_ids_by_query.get(str(query_id), set()))
                for query_id in query_ids
            }
        return {
            str(query_id): set() if include_query else {str(query_id)}
            for query_id in query_ids
        }

    def _find_nearest_neighbors_persistent(
        self,
        query_embeddings: Mapping[str, Any],
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        candidate_count: int | None,
        exclude_protein_ids: Sequence[str] | Mapping[str, Sequence[str]] | None,
        exclude_sequence_ids: Mapping[str, Set[int]] | None = None,
        source_revision: str | None = None,
    ) -> Dict[str, List[Neighbor]]:
        """Retrieve local IVF-PQ candidates and rerank them exactly in PostgreSQL."""
        manager = self._client._index_manager
        database_label = self._client._index_database_label
        if manager is None or database_label is None:
            raise BioDataError(
                "faiss_persistent requires a configured IndexManager. "
                "Pass index_manager and index_database_label to connect(), or call configure_persistent_index()."
            )
        query_items = list(query_embeddings.items())
        if not query_items:
            return {}
        effective_candidate_count = max(k, int(candidate_count)) if candidate_count is not None else max(k, 1_000)
        if effective_candidate_count < 1:
            raise BioDataError("ann_candidate_pool must be >= 1")
        key = IndexKey(
            database_label=database_label,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            dimension=embedding_dimension(query_items[0][1]),
        )
        revision = source_revision or self._client.embedding_index_revision(embedding_type_id, layer_index)
        cache_key = (key, revision)
        index = self._client._persistent_index_cache.get(cache_key)
        if index is None:
            index = manager.load(key, source_revision=revision)
            self._client._persistent_index_cache.clear()
            self._client._persistent_index_cache[cache_key] = index

        candidates = {
            str(query_id): [
                candidate.sequence_id
                for candidate in manager.search(
                    index,
                    query_embedding,
                    key=key,
                    candidate_count=effective_candidate_count,
                )
                if candidate.sequence_id
                not in (exclude_sequence_ids or {}).get(str(query_id), set())
            ]
            for query_id, query_embedding in query_items
        }
        return self._client.rerank_embedding_candidates_for_embeddings(
            query_embeddings,
            candidates,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            k=k,
            metric=metric,
            exclude_protein_ids=exclude_protein_ids,
        )

    def _canonicalize_exact_neighbor_groups(
        self,
        grouped: Mapping[str, Sequence[Neighbor]],
        query_embeddings: Mapping[str, Any],
        *,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        k: int,
    ) -> Dict[str, List[Neighbor]]:
        """Rank candidates with the shared float16 exact-store distance definition."""
        exact_store = self._load_persistent_exact_store(
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
        )
        manager = self._client._index_manager
        scorer = getattr(manager, "_exact_store_candidate_distances", None)
        if exact_store is None or not callable(scorer):
            return _canonicalize_neighbor_groups(grouped, k=k)
        canonical_groups: Dict[str, List[Neighbor]] = {}
        for query_id, neighbors in grouped.items():
            query_id_str = str(query_id)
            query_vector = query_embeddings.get(query_id_str)
            if query_vector is None:
                canonical_groups[query_id_str] = _canonicalize_neighbor_groups({query_id_str: neighbors}, k=k)[query_id_str]
                continue
            distances = cast(
                Mapping[str, float],
                scorer(
                    exact_store,
                    query_vector,
                    [neighbor.protein_id for neighbor in neighbors],
                    metric=metric,
                ),
            )
            canonical_groups[query_id_str] = sorted(
                (
                    Neighbor(
                        protein_id=neighbor.protein_id,
                        layer_index=neighbor.layer_index,
                        distance=distances[neighbor.protein_id],
                    )
                    for neighbor in neighbors
                    if neighbor.protein_id in distances
                ),
                key=lambda neighbor: (neighbor.distance, neighbor.protein_id),
            )[:k]
        return canonical_groups

    def _persistent_index_revision_if_current(
        self,
        query_embedding: Any,
        *,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
    ) -> str | None:
        """Return the source revision when the configured persistent index is current."""
        manager = self._client._index_manager
        database_label = self._client._index_database_label
        if manager is None or database_label is None:
            return None
        key = IndexKey(
            database_label=database_label,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            dimension=embedding_dimension(query_embedding),
        )
        revision = self._client.embedding_index_revision(embedding_type_id, layer_index)
        if manager.inspect(key, source_revision=revision).state != "current":
            return None
        return revision

    @staticmethod
    def _auto_persistent_resolution(resolved: ResolvedBackend) -> ResolvedBackend:
        """Mark a valid persistent index as the automatic backend choice."""
        return ResolvedBackend(
            "faiss_persistent",
            "cpu",
            True,
            True,
            False,
            "auto_persistent_index",
            resolved.batch_size,
            resolved.resident,
            "cpu",
        )

    def _search_workload_stats(self, *, embedding_type_id: int, layer_index: int) -> Dict[str, int]:
        cache_key = (int(embedding_type_id), int(layer_index))
        cached = self._client._search_workload_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

        sample = self._client.query_one(
            """
            SELECT embedding
            FROM sequence_embeddings
            WHERE embedding_type_id = %s
              AND layer_index = %s
            LIMIT 1;
            """,
            (embedding_type_id, layer_index),
        )
        if sample is None or sample.get("embedding") is None:
            raise NotFoundError(
                f"No embeddings found for embedding_type_id={embedding_type_id}, layer_index={layer_index}."
            )
        row_count = int(
            self._client.scalar(
                """
                SELECT count(*)
                FROM sequence_embeddings
                WHERE embedding_type_id = %s
                  AND layer_index = %s;
                """,
                (embedding_type_id, layer_index),
            )
        )
        stats = {
            "dim": int(embedding_dimension(sample["embedding"])),
            "row_count": row_count,
        }
        self._client._search_workload_cache[cache_key] = stats
        return dict(stats)

    def _probe_cuda_memory(self, *, device: str | None) -> tuple[int, int] | None:
        resolved_device = str(device or "").strip().lower()
        if not resolved_device.startswith("cuda"):
            return None
        device_index = cuda_device_index(resolved_device)

        cupy = import_cupy(allow_missing=True)
        if cupy is not None:
            try:
                with cupy.cuda.Device(device_index):
                    free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
                return int(free_bytes), int(total_bytes)
            except Exception:
                pass

        torch = import_torch()
        if bool(getattr(torch.cuda, "is_available", lambda: False)()):
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
                return int(free_bytes), int(total_bytes)
            except Exception:
                return None
        return None

    def _estimate_gpu_bytes(
        self,
        *,
        backend: ResolvedSearchBackend,
        embedding_type_id: int,
        layer_index: int,
        batch_size: int,
    ) -> int | None:
        if backend not in {"faiss_gpu", "cuvs_gpu", "torch_gpu"}:
            return None
        stats = self._search_workload_stats(embedding_type_id=embedding_type_id, layer_index=layer_index)
        dim = stats["dim"]
        row_count = stats["row_count"]

        index_bytes = row_count * dim * 4
        query_bytes = max(1, int(batch_size)) * dim * 4
        fixed_overhead = 128 * 1024 * 1024
        if backend == "torch_gpu":
            pairwise_bytes = max(1, int(batch_size)) * row_count * 4
            return int(index_bytes * 1.10 + query_bytes * 1.25 + pairwise_bytes * 1.10 + fixed_overhead)
        if backend == "cuvs_gpu":
            return int(index_bytes * 1.40 + query_bytes * 1.25 + fixed_overhead)
        return int(index_bytes * 1.50 + query_bytes * 1.30 + fixed_overhead)

    def _estimate_safe_gpu_batch_size(
        self,
        *,
        backend: ResolvedSearchBackend,
        embedding_type_id: int,
        layer_index: int,
        free_bytes: int,
        requested_batch_size: int,
    ) -> int | None:
        if backend not in {"faiss_gpu", "cuvs_gpu", "torch_gpu"}:
            return None
        stats = self._search_workload_stats(embedding_type_id=embedding_type_id, layer_index=layer_index)
        dim = stats["dim"]
        row_count = stats["row_count"]

        usable_bytes = max(0, int(free_bytes * 0.60) - 256 * 1024 * 1024)
        if usable_bytes < 1:
            return 0

        fixed_overhead = 128 * 1024 * 1024
        if backend == "torch_gpu":
            fixed_bytes = int(row_count * dim * 4 * 1.10 + fixed_overhead)
            per_query_bytes = int(dim * 4 * 1.25 + row_count * 4 * 1.10)
        elif backend == "cuvs_gpu":
            fixed_bytes = int(row_count * dim * 4 * 1.40 + fixed_overhead)
            per_query_bytes = int(dim * 4 * 1.25)
        else:
            fixed_bytes = int(row_count * dim * 4 * 1.50 + fixed_overhead)
            per_query_bytes = int(dim * 4 * 1.30)

        if usable_bytes <= fixed_bytes:
            return 0
        safe_batch = (usable_bytes - fixed_bytes) // max(1, per_query_bytes)
        return max(1, min(int(requested_batch_size), int(safe_batch)))

    def _apply_auto_gpu_heuristics(
        self,
        resolved: ResolvedBackend,
        *,
        requested_backend: SearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        batch_size: int,
        device: str | None,
    ) -> ResolvedBackend:
        del metric
        if str(requested_backend) != "auto":
            return resolved
        availability = self._client._detect_backend_availability(device=device)
        if resolved.backend not in {"faiss_gpu", "cuvs_gpu", "torch_gpu"}:
            return resolved
        if not str(resolved.device or "").startswith("cuda"):
            return resolved

        # Cold-start latency heuristic from current benchmark profile:
        # tiny batches should stay on pgvector, medium batches on faiss_cpu,
        # and only larger batches should pay the GPU warmup cost.
        if not resolved.resident:
            if batch_size <= 100:
                return ResolvedBackend(
                    "pgvector",
                    None,
                    resolved.ann_requested,
                    resolved.ann_requested,
                    False,
                    "auto_latency_pgvector",
                    batch_size,
                    False,
                    resolved.hardware_class,
                )
            if batch_size < 1000 and availability.faiss_cpu:
                return ResolvedBackend(
                    "faiss_cpu",
                    "cpu",
                    resolved.ann_requested,
                    resolved.ann_requested,
                    False,
                    "auto_latency_faiss_cpu",
                    batch_size,
                    False,
                    resolved.hardware_class,
                )

        memory = self._probe_cuda_memory(device=resolved.device)
        if memory is None:
            return resolved
        free_bytes, _total_bytes = memory
        estimated_bytes = self._estimate_gpu_bytes(
            backend=resolved.backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            batch_size=batch_size,
        )
        safe_chunk = self._estimate_safe_gpu_batch_size(
            backend=resolved.backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            free_bytes=free_bytes,
            requested_batch_size=batch_size,
        )
        effective_chunk = safe_chunk if safe_chunk and safe_chunk > 0 else None

        if effective_chunk is None:
            fallback_backend = "faiss_cpu" if self._client._detect_backend_availability(device=None).faiss_cpu else "pgvector"
            return ResolvedBackend(
                fallback_backend,
                "cpu" if fallback_backend == "faiss_cpu" else None,
                resolved.ann_requested,
                resolved.ann_requested if fallback_backend != "pgvector" else resolved.ann_used,
                True,
                "auto_gpu_memory_fallback_cpu",
                batch_size,
                False,
                resolved.hardware_class,
                chunk_size=None,
                estimated_bytes=estimated_bytes,
                free_bytes=free_bytes,
            )

        return ResolvedBackend(
            resolved.backend,
            resolved.device,
            resolved.ann_requested,
            resolved.ann_used,
            resolved.degraded,
            resolved.reason,
            resolved.batch_size,
            resolved.resident,
            resolved.hardware_class,
            chunk_size=effective_chunk,
            estimated_bytes=estimated_bytes,
            free_bytes=free_bytes,
        )

    def find_nearest_neighbors(
        self,
        query_embedding: Any,
        embedding_type_id: int,
        layer_index: int = 0,
        k: int | None = None,
        *,
        metric: DistanceMetric | None = None,
        exclude_protein_ids: Sequence[str] | None = None,
        use_ann: bool = False,
        ann_ef_search: int = 200,
        ann_candidate_pool: int | None = None,
        backend: SearchBackend | None = None,
        device: str | None = None,
    ) -> List[Neighbor]:
        """Find nearest neighbors for one query embedding."""
        effective_metric = metric or self._client.default_metric
        effective_k = self._client.default_k if k is None else int(k)
        if effective_k < 1:
            raise BioDataError("k must be >= 1")
        requested_backend = cast(SearchBackend, backend or self._client.default_backend)
        persistent_revision = (
            self._persistent_index_revision_if_current(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                metric=effective_metric,
            )
            if requested_backend == "auto"
            else None
        )
        backend_for_resolution = "faiss_persistent" if persistent_revision is not None else requested_backend
        resolved = self._client._resolve_search_backend(
            requested_backend=backend_for_resolution,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=1,
            ann_requested=use_ann,
            device=device,
        )
        if persistent_revision is not None:
            resolved = self._auto_persistent_resolution(resolved)
        resolved = self._apply_auto_gpu_heuristics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=1,
            device=device,
        )
        retrieval_k = effective_k + _CANONICAL_TIE_OVERFETCH if not resolved.ann_used else effective_k
        excluded_ids = [str(value) for value in (exclude_protein_ids or [])]

        if resolved.backend == "faiss_persistent":
            neighbors = self._find_nearest_neighbors_persistent(
                {"query": query_embedding},
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                candidate_count=ann_candidate_pool,
                exclude_protein_ids=sorted(excluded_ids),
                source_revision=persistent_revision,
            )["query"]
        elif resolved.backend == "pgvector":
            neighbors = self._client._find_nearest_neighbors_pgvector(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                use_ann=resolved.ann_used,
                ann_ef_search=ann_ef_search,
                ann_candidate_pool=ann_candidate_pool,
            )
        elif resolved.backend == "faiss_gpu":
            neighbors = self._client._find_nearest_neighbors_faiss(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                device=resolved.device,
                use_ann=resolved.ann_used,
            )
        elif resolved.backend == "faiss_cpu":
            neighbors = self._client._find_nearest_neighbors_faiss_cpu(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                use_ann=resolved.ann_used,
            )
        elif resolved.backend == "cuvs_gpu":
            neighbors = self._client._find_nearest_neighbors_cuvs(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                device=resolved.device,
                use_ann=resolved.ann_used,
            )
        else:
            neighbors = self._client._find_nearest_neighbors_torch(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                device=resolved.device,
            )

        neighbors = self._canonicalize_exact_neighbor_groups(
            {"query": neighbors},
            {"query": query_embedding},
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            k=effective_k,
        )["query"]
        self._client._record_search_diagnostics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            k=effective_k,
            query_count=1,
        )
        return neighbors

    def find_nearest_neighbors_for_embeddings(
        self,
        query_embeddings: Mapping[str, Any],
        embedding_type_id: int,
        layer_index: int = 0,
        k: int | None = None,
        *,
        metric: DistanceMetric | None = None,
        exclude_protein_ids: Sequence[str] | None = None,
        use_ann: bool = False,
        ann_ef_search: int = 200,
        ann_candidate_pool: int | None = None,
        backend: SearchBackend | None = None,
        device: str | None = None,
    ) -> Dict[str, List[Neighbor]]:
        """Find nearest neighbors for multiple external query embeddings."""
        query_items = [(str(query_id), embedding) for query_id, embedding in query_embeddings.items()]
        if not query_items:
            return {}

        effective_metric = metric or self._client.default_metric
        effective_k = self._client.default_k if k is None else int(k)
        if effective_k < 1:
            raise BioDataError("k must be >= 1")
        requested_backend = cast(SearchBackend, backend or self._client.default_backend)
        persistent_revision = (
            self._persistent_index_revision_if_current(
                query_items[0][1],
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                metric=effective_metric,
            )
            if requested_backend == "auto"
            else None
        )
        backend_for_resolution = "faiss_persistent" if persistent_revision is not None else requested_backend
        resolved = self._client._resolve_search_backend(
            requested_backend=backend_for_resolution,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=len(query_items),
            ann_requested=use_ann,
            device=device,
        )
        if persistent_revision is not None:
            resolved = self._auto_persistent_resolution(resolved)
        resolved = self._apply_auto_gpu_heuristics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=len(query_items),
            device=device,
        )
        retrieval_k = effective_k + _CANONICAL_TIE_OVERFETCH if not resolved.ann_used else effective_k

        query_ids = [query_id for query_id, _ in query_items]
        query_vectors = [embedding for _, embedding in query_items]
        excluded_ids = {str(protein_id) for protein_id in (exclude_protein_ids or [])}
        grouped: Dict[str, List[Neighbor]] = {}

        if resolved.backend == "faiss_persistent":
            grouped = self._find_nearest_neighbors_persistent(
                dict(query_items),
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                candidate_count=ann_candidate_pool,
                exclude_protein_ids=sorted(excluded_ids),
                source_revision=persistent_revision,
            )
        elif resolved.backend == "pgvector":
            grouped = self.find_nearest_neighbors_for_embeddings_pgvector(
                query_items,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                use_ann=resolved.ann_used,
                ann_ef_search=ann_ef_search,
                ann_candidate_pool=ann_candidate_pool,
            )
        else:
            state = self._client._get_or_load_gpu_search_state(
                backend=resolved.backend,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                metric=effective_metric,
                device=resolved.device,
                ann_requested=resolved.ann_used,
            )
            query_matrix = as_numpy_matrix(query_vectors)
            per_query_excluded = {query_id: set(excluded_ids) for query_id in query_ids}
            chunk_size = max(1, int(resolved.chunk_size or len(query_ids)))
            for chunk_start in range(0, len(query_ids), chunk_size):
                chunk_ids = query_ids[chunk_start:chunk_start + chunk_size]
                partial = search_state(
                    state,
                    query_ids=chunk_ids,
                    query_vectors=query_matrix[chunk_start:chunk_start + chunk_size],
                    k=retrieval_k,
                    per_query_excluded={query_id: per_query_excluded[query_id] for query_id in chunk_ids},
                )
                grouped.update(partial)

        grouped = self._canonicalize_exact_neighbor_groups(
            grouped,
            dict(query_items),
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            k=effective_k,
        )
        self._client._record_search_diagnostics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            k=effective_k,
            query_count=len(query_items),
        )
        return grouped

    def find_nearest_neighbors_for_proteins(
        self,
        protein_ids: Sequence[str],
        embedding_type_id: int,
        layer_index: int = 0,
        k: int | None = None,
        *,
        metric: DistanceMetric | None = None,
        include_query: bool = False,
        use_ann: bool = False,
        ann_ef_search: int = 200,
        ann_candidate_pool: int | None = None,
        backend: SearchBackend | None = None,
        device: str | None = None,
    ) -> Dict[str, List[Neighbor]]:
        """Find nearest neighbors for stored protein embeddings."""
        ids = [str(value) for value in protein_ids]
        if not ids:
            return {}

        effective_metric = metric or self._client.default_metric
        effective_k = self._client.default_k if k is None else int(k)
        if effective_k < 1:
            raise BioDataError("k must be >= 1")
        retrieval_k = effective_k + _CANONICAL_TIE_OVERFETCH
        requested_backend = cast(SearchBackend, backend or self._client.default_backend)
        persistent_query_map: Mapping[str, Any] | None = None
        persistent_revision: str | None = None
        if requested_backend == "auto" and self._client._index_manager is not None:
            persistent_query_map = self._client.get_protein_embeddings(
                ids,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            )
            if persistent_query_map:
                persistent_revision = self._persistent_index_revision_if_current(
                    next(iter(persistent_query_map.values())),
                    embedding_type_id=embedding_type_id,
                    layer_index=layer_index,
                    metric=effective_metric,
                )
        backend_for_resolution = "faiss_persistent" if persistent_revision is not None else requested_backend
        resolved = self._client._resolve_search_backend(
            requested_backend=backend_for_resolution,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=len(ids),
            ann_requested=use_ann,
            device=device,
        )
        if persistent_revision is not None:
            resolved = self._auto_persistent_resolution(resolved)
        resolved = self._apply_auto_gpu_heuristics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=len(ids),
            device=device,
        )

        excluded_protein_ids_by_query: Dict[str, Set[str]] = {}
        excluded_sequence_ids_by_query: Dict[str, Set[int]] = {}
        if not include_query and resolved.backend != "pgvector":
            excluded_protein_ids_by_query, excluded_sequence_ids_by_query = (
                self._client._stored_query_exclusions(ids)
            )

        if resolved.backend == "faiss_persistent":
            query_map = persistent_query_map or self._client.get_protein_embeddings(
                ids,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            )
            grouped = self._find_nearest_neighbors_persistent(
                query_map,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                candidate_count=ann_candidate_pool,
                exclude_protein_ids=(
                    None
                    if include_query
                    else {
                        query_id: sorted(protein_ids)
                        for query_id, protein_ids in excluded_protein_ids_by_query.items()
                    }
                ),
                exclude_sequence_ids=None if include_query else excluded_sequence_ids_by_query,
                source_revision=persistent_revision,
            )
        elif resolved.backend == "pgvector":
            grouped = self._client._find_nearest_neighbors_for_proteins_pgvector(
                ids,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=retrieval_k,
                metric=effective_metric,
                include_query=include_query,
                use_ann=resolved.ann_used,
                ann_ef_search=ann_ef_search,
                ann_candidate_pool=ann_candidate_pool,
            )
        else:
            query_map = persistent_query_map or self._client.get_protein_embeddings(
                ids,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            )
            if not query_map:
                return {}
            query_ids = [protein_id for protein_id in ids if protein_id in query_map]
            query_vectors = [query_map[protein_id] for protein_id in query_ids]
            chunk_size = max(1, int(resolved.chunk_size or len(query_ids) or 1))
            grouped: Dict[str, List[Neighbor]] = {}
            for chunk_start in range(0, len(query_ids), chunk_size):
                chunk_ids = query_ids[chunk_start:chunk_start + chunk_size]
                chunk_vectors = query_vectors[chunk_start:chunk_start + chunk_size]
                if resolved.backend == "faiss_gpu":
                    partial = self._client._find_nearest_neighbors_for_queries_faiss(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=retrieval_k,
                        metric=effective_metric,
                        include_query=include_query,
                        device=resolved.device,
                        use_ann=resolved.ann_used,
                        excluded_protein_ids_by_query=excluded_protein_ids_by_query,
                    )
                elif resolved.backend == "faiss_cpu":
                    partial = self._client._find_nearest_neighbors_for_queries_faiss_cpu(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=retrieval_k,
                        metric=effective_metric,
                        include_query=include_query,
                        use_ann=resolved.ann_used,
                        excluded_protein_ids_by_query=excluded_protein_ids_by_query,
                    )
                elif resolved.backend == "cuvs_gpu":
                    partial = self._client._find_nearest_neighbors_for_queries_cuvs(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=retrieval_k,
                        metric=effective_metric,
                        include_query=include_query,
                        device=resolved.device,
                        use_ann=resolved.ann_used,
                        excluded_protein_ids_by_query=excluded_protein_ids_by_query,
                    )
                else:
                    partial = self._client._find_nearest_neighbors_for_queries_torch(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=retrieval_k,
                        metric=effective_metric,
                        include_query=include_query,
                        device=resolved.device,
                        excluded_protein_ids_by_query=excluded_protein_ids_by_query,
                    )
                grouped.update(partial)

        grouped = self._canonicalize_exact_neighbor_groups(
            grouped,
            persistent_query_map or self._client.get_protein_embeddings(
                ids,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            ),
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            k=effective_k,
        )
        self._client._record_search_diagnostics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            k=effective_k,
            query_count=len(ids),
        )
        return grouped

    def find_nearest_neighbors_for_embeddings_pgvector(
        self,
        query_items: Sequence[tuple[str, Any]],
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        exclude_protein_ids: Set[str],
        use_ann: bool,
        ann_ef_search: int,
        ann_candidate_pool: int | None,
    ) -> Dict[str, List[Neighbor]]:
        """Find nearest neighbors for external embeddings through pgvector."""
        conn = self._client._require_connection()
        operator = metric_operator(metric)
        query_ids = [query_id for query_id, _ in query_items]
        grouped: Dict[str, List[Neighbor]] = {query_id: [] for query_id in query_ids}
        value_rows = ", ".join("(%s, %s::halfvec)" for _ in query_items)
        params: List[Any] = [value for query_id, embedding in query_items for value in (query_id, embedding)]

        exclusion_clause = ""
        if exclude_protein_ids:
            exclusion_clause = " AND p.id <> ALL(%s)"

        if use_ann:
            dim = embedding_dimension(query_items[0][1])
            candidate_limit = max(k, int(ann_candidate_pool)) if ann_candidate_pool is not None else max(k * 20, 200)
            sql = (
                "WITH query_embeddings(query_id, query_embedding) AS (VALUES "
                f"{value_rows}"
                "), ranked_neighbors AS ("
                "    SELECT q.query_id, "
                "           n.protein_id, "
                "           n.layer_index, "
                "           n.distance "
                "    FROM query_embeddings q "
                "    LEFT JOIN LATERAL ("
                "        WITH ann_candidates AS ("
                "            SELECT se.sequence_id, se.layer_index, se.embedding "
                "            FROM sequence_embeddings se "
                "            WHERE se.embedding_type_id = %s "
                "              AND se.layer_index = %s "
                f"            ORDER BY (se.embedding::halfvec({dim})) {operator} q.query_embedding "
                "            LIMIT %s"
                "        ) "
                "        SELECT p.id AS protein_id, "
                "               c.layer_index, "
                f"               c.embedding {operator} q.query_embedding AS distance "
                "        FROM ann_candidates c "
                "        JOIN protein p ON p.sequence_id = c.sequence_id "
                "        WHERE TRUE"
                f"{exclusion_clause} "
                "        ORDER BY distance "
                "        LIMIT %s"
                "    ) n ON TRUE"
                ") "
                "SELECT query_id, protein_id, layer_index, distance "
                "FROM ranked_neighbors "
                "ORDER BY query_id, distance;"
            )
            params.extend([embedding_type_id, layer_index, candidate_limit])
            if exclude_protein_ids:
                params.append(sorted(exclude_protein_ids))
            params.append(k)
        else:
            sql = (
                "WITH query_embeddings(query_id, query_embedding) AS (VALUES "
                f"{value_rows}"
                ") "
                "SELECT q.query_id, "
                "       n.protein_id, "
                "       n.layer_index, "
                "       n.distance "
                "FROM query_embeddings q "
                "LEFT JOIN LATERAL ("
                "    SELECT p.id AS protein_id, "
                "           se.layer_index, "
                f"           se.embedding {operator} q.query_embedding AS distance "
                "    FROM sequence_embeddings se "
                "    JOIN sequence s ON se.sequence_id = s.id "
                "    JOIN protein p ON p.sequence_id = s.id "
                "    WHERE se.embedding_type_id = %s "
                "      AND se.layer_index = %s"
                f"{exclusion_clause} "
                f"    ORDER BY se.embedding {operator} q.query_embedding "
                "    LIMIT %s"
                ") n ON TRUE "
                "ORDER BY q.query_id, n.distance;"
            )
            params.extend([embedding_type_id, layer_index])
            if exclude_protein_ids:
                params.append(sorted(exclude_protein_ids))
            params.append(k)

        with cursor(conn) as cur:
            if use_ann and ann_ef_search > 0:
                cur.execute(f"SET hnsw.ef_search = {int(ann_ef_search)};")
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        for query_id, protein_id, row_layer, distance in rows:
            if protein_id is None:
                continue
            grouped[str(query_id)].append(
                Neighbor(
                    protein_id=str(protein_id),
                    layer_index=int(row_layer),
                    distance=float(distance),
                )
            )
        return grouped

    def find_nearest_neighbors_pgvector(
        self,
        query_embedding: Any,
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        exclude_protein_ids: Sequence[str],
        use_ann: bool,
        ann_ef_search: int,
        ann_candidate_pool: int | None,
    ) -> List[Neighbor]:
        """Find nearest neighbors through pgvector."""
        conn = self._client._require_connection()
        operator = metric_operator(metric)
        excluded_ids = [str(value) for value in exclude_protein_ids]
        if use_ann:
            dim = embedding_dimension(query_embedding)
            candidate_limit = max(k, int(ann_candidate_pool)) if ann_candidate_pool is not None else max(k * 20, 200)
            extra_where = ""
            params = [
                embedding_type_id,
                layer_index,
                query_embedding,
                candidate_limit,
                query_embedding,
            ]
            if excluded_ids:
                extra_where = " AND p.id <> ALL(%s)"
                params.append(excluded_ids)

            sql = (
                "WITH ann_candidates AS ("
                "    SELECT se.sequence_id, "
                "           se.layer_index, "
                "           se.embedding "
                "    FROM sequence_embeddings se "
                "    WHERE se.embedding_type_id = %s "
                "      AND se.layer_index = %s "
                f"    ORDER BY (se.embedding::halfvec({dim})) {operator} %s::halfvec "
                "    LIMIT %s"
                "), protein_candidates AS ("
                "    SELECT p.id AS protein_id, "
                "           c.layer_index, "
                f"           c.embedding {operator} %s::halfvec AS distance "
                "    FROM ann_candidates c "
                "    JOIN protein p ON p.sequence_id = c.sequence_id "
                "    WHERE TRUE"
                f"{extra_where}"
                "), dedup AS ("
                "    SELECT protein_id, "
                "           MIN(layer_index) AS layer_index, "
                "           MIN(distance) AS distance "
                "    FROM protein_candidates "
                "    GROUP BY protein_id"
                ") "
                "SELECT protein_id, layer_index, distance "
                "FROM dedup "
                "ORDER BY distance "
                "LIMIT %s;"
            )
            params.append(k)
        else:
            extra_where = ""
            params = [query_embedding, embedding_type_id, layer_index]
            if excluded_ids:
                extra_where = " AND p.id <> ALL(%s)"
                params.append(excluded_ids)

            sql = (
                "SELECT p.id AS protein_id, "
                "       se.layer_index, "
                f"       se.embedding {operator} %s::halfvec AS distance "
                "FROM sequence_embeddings se "
                "JOIN sequence s ON se.sequence_id = s.id "
                "JOIN protein p ON p.sequence_id = s.id "
                "WHERE se.embedding_type_id = %s "
                "  AND se.layer_index = %s"
                f"{extra_where} "
                f"ORDER BY se.embedding {operator} %s::halfvec "
                "LIMIT %s;"
            )
            params.extend([query_embedding, k])

        with cursor(conn) as cur:
            if use_ann and ann_ef_search > 0:
                cur.execute(f"SET hnsw.ef_search = {int(ann_ef_search)};")
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

        return [
            Neighbor(protein_id=str(protein_id), layer_index=int(row_layer), distance=float(distance))
            for protein_id, row_layer, distance in rows
        ]

    def find_nearest_neighbors_for_proteins_pgvector(
        self,
        protein_ids: Sequence[str],
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        include_query: bool,
        use_ann: bool,
        ann_ef_search: int,
        ann_candidate_pool: int | None,
    ) -> Dict[str, List[Neighbor]]:
        """Find stored-protein nearest neighbors through pgvector."""
        conn = self._client._require_connection()
        operator = metric_operator(metric)
        dim_row = self._client.query_one(
            """
            SELECT embedding
            FROM sequence_embeddings
            WHERE embedding_type_id = %s
              AND layer_index = %s
            LIMIT 1;
            """,
            (embedding_type_id, layer_index),
        )
        if dim_row is None or dim_row.get("embedding") is None:
            return {}
        dim = embedding_dimension(dim_row["embedding"])
        if use_ann:
            self._client._warn_if_missing_ann_index(embedding_type_id, layer_index, metric)

        exclude_clause = ""
        if not include_query:
            exclude_clause = " AND se2.sequence_id <> q.query_sequence_id "

        candidate_limit = max(k, int(ann_candidate_pool)) if ann_candidate_pool is not None else max(k * 20, 200)
        sql = (
            "WITH query_embeddings AS ("
            "    SELECT p.id AS query_protein_id, "
            "           p.sequence_id AS query_sequence_id, "
            "           se.embedding AS query_embedding "
            "    FROM protein p "
            "    JOIN sequence s ON p.sequence_id = s.id "
            "    JOIN sequence_embeddings se ON se.sequence_id = s.id "
            "    WHERE p.id = ANY(%s) "
            "      AND se.embedding_type_id = %s "
            "      AND se.layer_index = %s"
            ") "
            "SELECT q.query_protein_id, "
            "       n.protein_id, "
            "       n.layer_index, "
            "       n.distance "
            "FROM query_embeddings q "
            "LEFT JOIN LATERAL ("
            "    SELECT p2.id AS protein_id, "
            "           c.layer_index, "
            "           c.distance "
            "    FROM ("
            "        SELECT se2.sequence_id, "
            "               se2.layer_index, "
            f"               (se2.embedding::halfvec({dim})) {operator} "
            f"               (q.query_embedding::halfvec({dim})) AS distance "
            "        FROM sequence_embeddings se2 "
            "        WHERE se2.embedding_type_id = %s "
            "          AND se2.layer_index = %s "
            f"         {exclude_clause}"
            f"        ORDER BY (se2.embedding::halfvec({dim})) {operator} (q.query_embedding::halfvec({dim})) "
            "        LIMIT %s"
            "    ) c "
            "    JOIN protein p2 ON p2.sequence_id = c.sequence_id "
            "    ORDER BY c.distance, p2.id "
            "    LIMIT %s"
            ") n ON TRUE "
            "ORDER BY q.query_protein_id, n.distance, n.protein_id;"
        )

        params = (
            [str(value) for value in protein_ids],
            embedding_type_id,
            layer_index,
            embedding_type_id,
            layer_index,
            candidate_limit if use_ann else k,
            k,
        )

        with cursor(conn) as cur:
            if use_ann and ann_ef_search > 0:
                cur.execute(f"SET hnsw.ef_search = {int(ann_ef_search)};")
            cur.execute(sql, params)
            rows = cur.fetchall()

        grouped: Dict[str, List[Neighbor]] = {}
        for query_protein_id, neighbor_id, row_layer, distance in rows:
            query_id = str(query_protein_id)
            grouped.setdefault(query_id, [])
            if neighbor_id is None:
                continue
            grouped[query_id].append(
                Neighbor(
                    protein_id=str(neighbor_id),
                    layer_index=int(row_layer),
                    distance=float(distance),
                )
            )
        return grouped

    def find_nearest_neighbors_faiss(
        self,
        query_embedding: Any,
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        exclude_protein_ids: Sequence[str],
        device: str | None,
        use_ann: bool,
    ) -> List[Neighbor]:
        """Find nearest neighbors through a FAISS GPU index."""
        state = self._client._get_or_load_gpu_search_state(
            backend="faiss_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix([query_embedding])
        grouped = self._client._search_faiss_state(
            state,
            query_ids=["__single__"],
            query_vectors=query_matrix,
            k=k,
            per_query_excluded={"__single__": set(str(value) for value in exclude_protein_ids)},
        )
        return grouped["__single__"]

    def find_nearest_neighbors_faiss_cpu(
        self,
        query_embedding: Any,
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        exclude_protein_ids: Sequence[str],
        use_ann: bool,
    ) -> List[Neighbor]:
        """Find nearest neighbors through a FAISS CPU index."""
        state = self._client._get_or_load_gpu_search_state(
            backend="faiss_cpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device="cpu",
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix([query_embedding])
        grouped = self._client._search_faiss_state(
            state,
            query_ids=["__single__"],
            query_vectors=query_matrix,
            k=k,
            per_query_excluded={"__single__": set(str(value) for value in exclude_protein_ids)},
        )
        return grouped["__single__"]

    def find_nearest_neighbors_torch(
        self,
        query_embedding: Any,
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        exclude_protein_ids: Sequence[str],
        device: str | None,
    ) -> List[Neighbor]:
        """Find nearest neighbors through a torch tensor backend."""
        state = self._client._get_or_load_gpu_search_state(
            backend="torch_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=False,
        )
        query_matrix = as_numpy_matrix([query_embedding])
        grouped = self._client._search_torch_state(
            state,
            query_ids=["__single__"],
            query_vectors=query_matrix,
            k=k,
            per_query_excluded={"__single__": set(str(value) for value in exclude_protein_ids)},
        )
        return grouped["__single__"]

    def find_nearest_neighbors_cuvs(
        self,
        query_embedding: Any,
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        exclude_protein_ids: Sequence[str],
        device: str | None,
        use_ann: bool,
    ) -> List[Neighbor]:
        """Find nearest neighbors through a cuVS GPU index."""
        state = self._client._get_or_load_gpu_search_state(
            backend="cuvs_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix([query_embedding])
        grouped = self._client._search_cuvs_state(
            state,
            query_ids=["__single__"],
            query_vectors=query_matrix,
            k=k,
            per_query_excluded={"__single__": set(str(value) for value in exclude_protein_ids)},
        )
        return grouped["__single__"]

    def find_nearest_neighbors_for_queries_faiss(
        self,
        query_ids: Sequence[str],
        query_vectors: Sequence[Any],
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        include_query: bool,
        device: str | None,
        use_ann: bool,
        excluded_protein_ids_by_query: Mapping[str, Set[str]] | None = None,
    ) -> Dict[str, List[Neighbor]]:
        """Find batch nearest neighbors through a FAISS GPU index."""
        state = self._client._get_or_load_gpu_search_state(
            backend="faiss_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded = self._per_query_exclusions(
            query_ids,
            include_query=include_query,
            excluded_protein_ids_by_query=excluded_protein_ids_by_query,
        )
        return self._client._search_faiss_state(
            state,
            query_ids=query_ids,
            query_vectors=query_matrix,
            k=k,
            per_query_excluded=per_query_excluded,
        )

    def find_nearest_neighbors_for_queries_faiss_cpu(
        self,
        query_ids: Sequence[str],
        query_vectors: Sequence[Any],
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        include_query: bool,
        use_ann: bool,
        excluded_protein_ids_by_query: Mapping[str, Set[str]] | None = None,
    ) -> Dict[str, List[Neighbor]]:
        """Find batch nearest neighbors through a FAISS CPU index."""
        state = self._client._get_or_load_gpu_search_state(
            backend="faiss_cpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device="cpu",
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded = self._per_query_exclusions(
            query_ids,
            include_query=include_query,
            excluded_protein_ids_by_query=excluded_protein_ids_by_query,
        )
        return self._client._search_faiss_state(
            state,
            query_ids=query_ids,
            query_vectors=query_matrix,
            k=k,
            per_query_excluded=per_query_excluded,
        )

    def find_nearest_neighbors_for_queries_cuvs(
        self,
        query_ids: Sequence[str],
        query_vectors: Sequence[Any],
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        include_query: bool,
        device: str | None,
        use_ann: bool,
        excluded_protein_ids_by_query: Mapping[str, Set[str]] | None = None,
    ) -> Dict[str, List[Neighbor]]:
        """Find batch nearest neighbors through a cuVS GPU index."""
        state = self._client._get_or_load_gpu_search_state(
            backend="cuvs_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded = self._per_query_exclusions(
            query_ids,
            include_query=include_query,
            excluded_protein_ids_by_query=excluded_protein_ids_by_query,
        )
        return self._client._search_cuvs_state(
            state,
            query_ids=query_ids,
            query_vectors=query_matrix,
            k=k,
            per_query_excluded=per_query_excluded,
        )

    def find_nearest_neighbors_for_queries_torch(
        self,
        query_ids: Sequence[str],
        query_vectors: Sequence[Any],
        *,
        embedding_type_id: int,
        layer_index: int,
        k: int,
        metric: DistanceMetric,
        include_query: bool,
        device: str | None,
        excluded_protein_ids_by_query: Mapping[str, Set[str]] | None = None,
    ) -> Dict[str, List[Neighbor]]:
        """Find batch nearest neighbors through a torch tensor backend."""
        state = self._client._get_or_load_gpu_search_state(
            backend="torch_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=False,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded = self._per_query_exclusions(
            query_ids,
            include_query=include_query,
            excluded_protein_ids_by_query=excluded_protein_ids_by_query,
        )
        return self._client._search_torch_state(
            state,
            query_ids=query_ids,
            query_vectors=query_matrix,
            k=k,
            per_query_excluded=per_query_excluded,
        )

    def search_faiss_state(
        self,
        state: GpuSearchState,
        *,
        query_ids: Sequence[str],
        query_vectors: Any,
        k: int,
        per_query_excluded: Mapping[str, Set[str]],
    ) -> Dict[str, List[Neighbor]]:
        """Search an initialized FAISS state for query vectors."""
        return search_faiss_state(
            state,
            query_ids=query_ids,
            query_vectors=query_vectors,
            k=k,
            per_query_excluded=per_query_excluded,
        )

    def search_cuvs_state(
        self,
        state: GpuSearchState,
        *,
        query_ids: Sequence[str],
        query_vectors: Any,
        k: int,
        per_query_excluded: Mapping[str, Set[str]],
    ) -> Dict[str, List[Neighbor]]:
        """Search an initialized cuVS state for query vectors."""
        return search_cuvs_state(
            state,
            query_ids=query_ids,
            query_vectors=query_vectors,
            k=k,
            per_query_excluded=per_query_excluded,
        )

    def search_torch_state(
        self,
        state: GpuSearchState,
        *,
        query_ids: Sequence[str],
        query_vectors: Any,
        k: int,
        per_query_excluded: Mapping[str, Set[str]],
    ) -> Dict[str, List[Neighbor]]:
        """Search an initialized torch state for query vectors."""
        return search_torch_state(
            state,
            query_ids=query_ids,
            query_vectors=query_vectors,
            k=k,
            per_query_excluded=per_query_excluded,
        )

    def neighbors_from_candidate_rows(
        self,
        state: GpuSearchState,
        *,
        candidate_indices: Sequence[Any],
        candidate_distances: Sequence[Any],
        k: int,
        excluded_protein_ids: Set[str],
        l2_squared: bool,
    ) -> List[Neighbor]:
        """Convert backend candidate rows into neighbor records."""
        return neighbors_from_candidate_rows(
            state,
            candidate_indices=candidate_indices,
            candidate_distances=candidate_distances,
            k=k,
            excluded_protein_ids=excluded_protein_ids,
            l2_squared=l2_squared,
        )

    def resolve_search_backend(
        self,
        *,
        requested_backend: SearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        batch_size: int,
        ann_requested: bool,
        device: str | None,
    ) -> ResolvedBackend:
        """Resolve the effective search backend for a workload."""
        requested = str(requested_backend).strip().lower()
        if requested not in {"auto", "gpu", "pgvector", "faiss_cpu", "faiss_gpu", "faiss_persistent", "cuvs_gpu", "torch_gpu"}:
            raise BioDataError(f"Unsupported search backend: {requested_backend!r}")

        if requested == "faiss_persistent":
            if self._client._index_manager is None or self._client._index_database_label is None:
                raise BioDataError(
                    "faiss_persistent requires a configured IndexManager. "
                    "Pass index_manager and index_database_label to connect(), or call configure_persistent_index()."
                )
            return ResolvedBackend(
                "faiss_persistent",
                "cpu",
                True,
                True,
                False,
                "explicit_faiss_persistent",
                batch_size,
                bool(self._client._persistent_index_cache),
                "cpu",
            )

        availability = self._client._detect_backend_availability(device=device)
        resident_faiss = self._client._gpu_state_matches(
            backend="faiss_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=availability.faiss_device,
            ann_enabled=ann_requested,
        )
        resident_faiss_cpu = self._client._gpu_state_matches(
            backend="faiss_cpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device="cpu",
            ann_enabled=ann_requested,
        )
        resident_cuvs = self._client._gpu_state_matches(
            backend="cuvs_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=availability.cuvs_device,
            ann_enabled=ann_requested,
        )
        resident_torch = self._client._gpu_state_matches(
            backend="torch_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=availability.torch_device,
            ann_enabled=False,
        )
        thresholds = self._client.backend_thresholds.get(availability.hardware_class, DEFAULT_BACKEND_THRESHOLDS["cpu"])
        faiss_min_batch = int(thresholds.get("faiss_gpu_min_batch", DEFAULT_BACKEND_THRESHOLDS["cpu"]["faiss_gpu_min_batch"]))
        cuvs_min_batch = int(thresholds.get("cuvs_gpu_min_batch", DEFAULT_BACKEND_THRESHOLDS["cpu"]["cuvs_gpu_min_batch"]))
        torch_min_batch = int(thresholds.get("torch_gpu_min_batch", DEFAULT_BACKEND_THRESHOLDS["cpu"]["torch_gpu_min_batch"]))
        resident_min_batch = int(thresholds.get("resident_gpu_min_batch", 1))

        if requested == "pgvector":
            return ResolvedBackend("pgvector", None, ann_requested, ann_requested, False, "explicit_pgvector", batch_size, False, availability.hardware_class)
        if requested == "faiss_cpu":
            if not availability.faiss_cpu:
                raise BioDataError("Requested backend 'faiss_cpu' is not available on this host.")
            return ResolvedBackend("faiss_cpu", "cpu", ann_requested, ann_requested, False, "explicit_faiss_cpu", batch_size, resident_faiss_cpu, availability.hardware_class)
        if requested == "faiss_gpu":
            if not availability.faiss_gpu or availability.faiss_device is None:
                raise BioDataError("Requested backend 'faiss_gpu' is not available on this host.")
            return ResolvedBackend("faiss_gpu", availability.faiss_device, ann_requested, ann_requested, False, "explicit_faiss_gpu", batch_size, resident_faiss, availability.hardware_class)
        if requested == "cuvs_gpu":
            if not availability.cuvs_gpu or availability.cuvs_device is None:
                raise BioDataError("Requested backend 'cuvs_gpu' is not available on this host.")
            return ResolvedBackend("cuvs_gpu", availability.cuvs_device, ann_requested, ann_requested, False, "explicit_cuvs_gpu", batch_size, resident_cuvs, availability.hardware_class)
        if requested == "torch_gpu":
            if not availability.torch_gpu or availability.torch_device is None:
                raise BioDataError("Requested backend 'torch_gpu' is not available on this host.")
            return ResolvedBackend("torch_gpu", availability.torch_device, ann_requested, False, bool(ann_requested), "explicit_torch_gpu", batch_size, resident_torch, availability.hardware_class)

        if requested == "gpu":
            if availability.faiss_gpu and availability.faiss_device is not None:
                return ResolvedBackend("faiss_gpu", availability.faiss_device, ann_requested, ann_requested, False, "gpu_preferred_faiss", batch_size, resident_faiss, availability.hardware_class)
            if availability.cuvs_gpu and availability.cuvs_device is not None:
                return ResolvedBackend("cuvs_gpu", availability.cuvs_device, ann_requested, ann_requested, False, "gpu_fallback_cuvs", batch_size, resident_cuvs, availability.hardware_class)
            if availability.torch_gpu and availability.torch_device is not None:
                reason = "gpu_degraded_torch_after_ann" if ann_requested else "gpu_fallback_torch"
                return ResolvedBackend("torch_gpu", availability.torch_device, ann_requested, False, bool(ann_requested), reason, batch_size, resident_torch, availability.hardware_class)
            return ResolvedBackend("pgvector", None, ann_requested, ann_requested, False, "gpu_requested_but_unavailable", batch_size, False, availability.hardware_class)

        if resident_faiss and availability.faiss_gpu and availability.faiss_device is not None and batch_size >= resident_min_batch:
            return ResolvedBackend("faiss_gpu", availability.faiss_device, ann_requested, ann_requested, False, "resident_faiss", batch_size, True, availability.hardware_class)
        if resident_cuvs and availability.cuvs_gpu and availability.cuvs_device is not None and batch_size >= resident_min_batch:
            return ResolvedBackend("cuvs_gpu", availability.cuvs_device, ann_requested, ann_requested, False, "resident_cuvs", batch_size, True, availability.hardware_class)
        if resident_torch and availability.torch_gpu and availability.torch_device is not None and batch_size >= resident_min_batch:
            return ResolvedBackend("torch_gpu", availability.torch_device, ann_requested, False, bool(ann_requested), "resident_torch", batch_size, True, availability.hardware_class)
        if ann_requested:
            if availability.faiss_gpu and availability.faiss_device is not None and batch_size >= faiss_min_batch:
                return ResolvedBackend("faiss_gpu", availability.faiss_device, True, True, False, "ann_auto_faiss", batch_size, False, availability.hardware_class)
            if availability.cuvs_gpu and availability.cuvs_device is not None and batch_size >= cuvs_min_batch:
                return ResolvedBackend("cuvs_gpu", availability.cuvs_device, True, True, False, "ann_auto_cuvs", batch_size, False, availability.hardware_class)
            if availability.torch_gpu and availability.torch_device is not None and batch_size >= torch_min_batch:
                return ResolvedBackend("torch_gpu", availability.torch_device, True, False, True, "ann_degraded_torch", batch_size, False, availability.hardware_class)
            return ResolvedBackend("pgvector", None, True, True, False, "ann_auto_pgvector", batch_size, False, availability.hardware_class)
        if availability.faiss_gpu and availability.faiss_device is not None and batch_size >= faiss_min_batch:
            return ResolvedBackend("faiss_gpu", availability.faiss_device, False, False, False, "auto_faiss_threshold", batch_size, False, availability.hardware_class)
        if availability.cuvs_gpu and availability.cuvs_device is not None and batch_size >= cuvs_min_batch:
            return ResolvedBackend("cuvs_gpu", availability.cuvs_device, False, False, False, "auto_cuvs_threshold", batch_size, False, availability.hardware_class)
        if availability.torch_gpu and availability.torch_device is not None and batch_size >= torch_min_batch:
            return ResolvedBackend("torch_gpu", availability.torch_device, False, False, False, "auto_torch_threshold", batch_size, False, availability.hardware_class)
        return ResolvedBackend("pgvector", None, ann_requested, ann_requested, False, "auto_pgvector_threshold", batch_size, False, availability.hardware_class)

    def detect_backend_availability(self, *, device: str | None) -> BackendAvailability:
        """Detect available local search backends and devices."""
        torch_device = preferred_torch_device(device)
        faiss_device = preferred_faiss_device(device)
        cuvs_device = preferred_cuvs_device(device)
        faiss_cpu_available = import_faiss(allow_missing=True) is not None
        faiss_available = False
        if faiss_device is not None:
            faiss = import_faiss(allow_missing=True)
            if faiss is not None:
                get_num_gpus = getattr(faiss, "get_num_gpus", None)
                if callable(get_num_gpus):
                    try:
                        faiss_available = int(cast(Any, get_num_gpus)()) > 0
                    except Exception:
                        faiss_available = False
                else:
                    faiss_available = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "index_cpu_to_gpu")
        cuvs_available = cuvs_device is not None
        torch_available = torch_device is not None
        preferred_device = faiss_device or cuvs_device or torch_device
        hardware_class = "cpu"
        if preferred_device is not None:
            hardware_class = "cuda" if preferred_device.startswith("cuda") else "mps"
        return BackendAvailability(
            faiss_gpu=faiss_available,
            torch_gpu=torch_available,
            preferred_device=preferred_device,
            torch_device=torch_device,
            faiss_device=faiss_device if faiss_available else None,
            hardware_class=hardware_class,
            faiss_cpu=faiss_cpu_available,
            cuvs_gpu=cuvs_available,
            cuvs_device=cuvs_device if cuvs_available else None,
        )

    def get_or_load_gpu_search_state(
        self,
        *,
        backend: ResolvedSearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        device: str | None,
        ann_requested: bool,
    ) -> GpuSearchState:
        """Return a cached GPU search state or load a new one."""
        resolved_device = str(device or "")
        if not resolved_device:
            raise BioDataError(f"{backend} selected without a usable accelerator device.")
        if self._client._gpu_state_matches(
            backend=backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=resolved_device,
            ann_enabled=ann_requested if backend in {"faiss_gpu", "cuvs_gpu"} else False,
        ):
            return self._client._search_state_cache[_search_state_cache_slot(backend, resolved_device)]
        state = self._client._load_gpu_search_state(
            backend=backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=resolved_device,
            ann_requested=ann_requested,
        )
        self._client._search_state_cache[_search_state_cache_slot(backend, resolved_device)] = state
        return state

    def load_gpu_search_state(
        self,
        *,
        backend: ResolvedSearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        device: str,
        ann_requested: bool,
    ) -> GpuSearchState:
        """Load vectors and initialize an accelerated search state."""
        exact_store = self._load_persistent_exact_store(
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
        )
        if exact_store is not None and not ann_requested and backend == "faiss_cpu":
            return build_faiss_streaming_search_state(
                batches=self._client._index_manager.iter_exact_store_batches(
                    exact_store,
                    batch_size=_GPU_EXACT_LOAD_BATCH_SIZE,
                ),
                vector_count=exact_store.manifest.vector_count,
                dimension=exact_store.manifest.dimension,
                metric=metric,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            )
        if exact_store is not None and not ann_requested and backend == "cuvs_gpu":
            return build_cuvs_streaming_search_state(
                batches=self._client._index_manager.iter_exact_store_batches(
                    exact_store,
                    batch_size=_GPU_EXACT_LOAD_BATCH_SIZE,
                ),
                vector_count=exact_store.manifest.vector_count,
                dimension=exact_store.manifest.dimension,
                metric=metric,
                device=device,
                ann_requested=False,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            )
        if backend == "cuvs_gpu" and not ann_requested:
            dimension = self._client.embedding_index_dimension(embedding_type_id, layer_index)
            vector_count = self._protein_search_vector_count(
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            )
            return build_cuvs_streaming_search_state(
                batches=self._iter_protein_search_vector_batches(
                    embedding_type_id=embedding_type_id,
                    layer_index=layer_index,
                    batch_size=_GPU_EXACT_LOAD_BATCH_SIZE,
                ),
                vector_count=vector_count,
                dimension=dimension,
                metric=metric,
                device=device,
                ann_requested=False,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
            )
        protein_ids, vectors = self._client._load_search_vectors(embedding_type_id=embedding_type_id, layer_index=layer_index)
        return build_search_state(
            backend=backend,
            item_ids=protein_ids,
            vectors=vectors,
            metric=metric,
            device=device,
            ann_requested=ann_requested,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
        )

    def _load_persistent_exact_store(
        self,
        *,
        embedding_type_id: int,
        layer_index: int,
    ) -> Any | None:
        """Return a current exact store when one is configured for this collection."""
        manager = self._client._index_manager
        database_label = self._client._index_database_label
        load_exact_store = getattr(manager, "load_exact_store", None)
        if manager is None or database_label is None or not callable(load_exact_store):
            return None
        source_revision = None
        if self._client.is_connected:
            source_revision = self._client.embedding_index_revision(embedding_type_id, layer_index)
        try:
            return load_exact_store(
                database_label=database_label,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                source_revision=source_revision,
            )
        except SearchIndexError:
            return None

    def _protein_search_vector_count(self, *, embedding_type_id: int, layer_index: int) -> int:
        """Return the number of protein rows represented in an exact local search."""
        value = self._client.scalar(
            """
            SELECT COUNT(*)
            FROM sequence_embeddings se
            JOIN protein p ON p.sequence_id = se.sequence_id
            WHERE se.embedding_type_id = %s
              AND se.layer_index = %s;
            """,
            (embedding_type_id, layer_index),
        )
        vector_count = int(value or 0)
        if vector_count < 1:
            raise NotFoundError(
                f"No embeddings found for embedding_type_id={embedding_type_id}, layer_index={layer_index}."
            )
        return vector_count

    def _iter_protein_search_vector_batches(
        self,
        *,
        embedding_type_id: int,
        layer_index: int,
        batch_size: int,
    ) -> Generator[tuple[list[str], Any], None, None]:
        """Yield binary-decoded protein vectors for an exact local search."""
        conn = self._client._require_connection()
        cursor_name = f"biodata_search_{uuid.uuid4().hex}"
        with conn.transaction():
            with conn.cursor(name=cursor_name, binary=True) as cur:
                cur.execute(
                    """
                    SELECT p.id, se.embedding
                    FROM sequence_embeddings se
                    JOIN protein p ON p.sequence_id = se.sequence_id
                    WHERE se.embedding_type_id = %s
                      AND se.layer_index = %s
                    ORDER BY p.id;
                    """,
                    (embedding_type_id, layer_index),
                )
                while rows := cur.fetchmany(batch_size):
                    yield (
                        [str(row[0]) for row in rows],
                        as_numpy_matrix([row[1] for row in rows]),
                    )

    def load_search_vectors(self, *, embedding_type_id: int, layer_index: int) -> tuple[List[str], Any]:
        """Load search vectors for one embedding type and layer."""
        rows = self._client._query_all_binary(
            """
            SELECT p.id AS protein_id, se.embedding
            FROM sequence_embeddings se
            JOIN sequence s ON se.sequence_id = s.id
            JOIN protein p ON p.sequence_id = s.id
            WHERE se.embedding_type_id = %s
              AND se.layer_index = %s
            ORDER BY p.id;
            """,
            (embedding_type_id, layer_index),
        )
        if not rows:
            raise NotFoundError(
                f"No embeddings found for embedding_type_id={embedding_type_id}, layer_index={layer_index}."
            )
        protein_ids = [str(row["protein_id"]) for row in rows]
        vectors = as_numpy_matrix([row["embedding"] for row in rows])
        return protein_ids, vectors

    def gpu_state_matches(
        self,
        *,
        backend: ResolvedSearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        device: str | None,
        ann_enabled: bool,
    ) -> bool:
        """Return whether the cached accelerated search state matches a request."""
        state = self._client._search_state_cache.get(_search_state_cache_slot(backend, device))
        return bool(
            state is not None
            and state.backend == backend
            and state.embedding_type_id == int(embedding_type_id)
            and state.layer_index == int(layer_index)
            and str(state.metric) == str(metric)
            and state.device == str(device or "")
            and state.ann_enabled == bool(ann_enabled)
        )

    def record_search_diagnostics(
        self,
        resolved: ResolvedBackend,
        *,
        requested_backend: SearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        k: int,
        query_count: int,
    ) -> None:
        """Record diagnostics for a completed search."""
        self._client._warn_if_search_backend_degraded(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
        )
        self._client._last_search_diagnostics = {
            "requested_backend": requested_backend,
            "resolved_backend": resolved.backend,
            "device": resolved.device,
            "ann_requested": resolved.ann_requested,
            "ann_used": resolved.ann_used,
            "degraded": resolved.degraded,
            "reason": resolved.reason,
            "batch_size": resolved.batch_size,
            "resident": resolved.resident,
            "hardware_class": resolved.hardware_class,
            "embedding_type_id": int(embedding_type_id),
            "layer_index": int(layer_index),
            "metric": str(metric),
            "k": int(k),
            "query_count": int(query_count),
            "chunk_size": None if resolved.chunk_size is None else int(resolved.chunk_size),
            "estimated_bytes": None if resolved.estimated_bytes is None else int(resolved.estimated_bytes),
            "free_bytes": None if resolved.free_bytes is None else int(resolved.free_bytes),
        }

    def warn_if_search_backend_degraded(
        self,
        resolved: ResolvedBackend,
        *,
        requested_backend: SearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
    ) -> None:
        """Warn once when backend routing degrades a search request."""
        warning_message: str | None = None
        requested = str(requested_backend)
        reason = str(resolved.reason)

        if requested == "gpu" and reason == "gpu_requested_but_unavailable":
            warning_message = (
                "GPU search was requested, but no accelerated backend is available on this host. "
                "Falling back to pgvector."
            )
        elif requested == "gpu" and reason == "gpu_fallback_cuvs":
            warning_message = "GPU search selected cuvs_gpu because faiss_gpu is not available on this host."
        elif requested == "gpu" and reason == "gpu_fallback_torch":
            warning_message = "GPU search selected torch_gpu because faiss_gpu and cuvs_gpu are not available on this host."
        elif requested == "gpu" and reason == "gpu_degraded_torch_after_ann":
            warning_message = (
                "GPU search requested ANN-capable search, but neither faiss_gpu nor cuvs_gpu is available. "
                "Degrading to exact torch_gpu search."
            )
        elif requested == "auto" and reason == "ann_degraded_torch":
            warning_message = (
                "Auto backend routing requested ANN-capable GPU search, but neither faiss_gpu nor cuvs_gpu is available. "
                "Degrading to exact torch_gpu search."
            )
        elif requested == "auto" and reason == "auto_cuvs_threshold":
            warning_message = (
                "Auto backend routing selected cuvs_gpu because faiss_gpu is not available for this host or device."
            )
        elif requested == "auto" and reason == "auto_torch_threshold":
            warning_message = (
                "Auto backend routing selected torch_gpu because faiss_gpu and cuvs_gpu are not available for this host or device."
            )
        elif requested == "auto" and reason == "auto_gpu_memory_fallback_cpu":
            warning_message = (
                "Auto backend routing fell back to a CPU backend because the estimated GPU memory requirement "
                "exceeded the safe free-VRAM budget."
            )
        elif requested == "auto" and reason == "auto_pgvector_threshold" and resolved.hardware_class != "cpu":
            warning_message = (
                "Auto backend routing fell back to pgvector because no accelerated backend met the current hardware "
                "or threshold requirements."
            )

        if warning_message is None:
            return

        cache_key = (
            requested,
            reason,
            str(resolved.backend),
            str(resolved.hardware_class),
            bool(resolved.ann_requested),
        )
        if cache_key in self._client._search_backend_warned:
            return

        warnings.warn(
            (
                f"{warning_message} "
                f"(embedding_type_id={int(embedding_type_id)}, layer_index={int(layer_index)}, metric={str(metric)})."
            ),
            RuntimeWarning,
            stacklevel=3,
        )
        self._client._search_backend_warned.add(cache_key)

    def warn_if_missing_ann_index(
        self,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
    ) -> None:
        """Warn once when pgvector ANN search lacks a matching HNSW index."""
        metric_name = str(metric).strip().lower()
        opclass = metric_opclass(cast(DistanceMetric, metric_name))
        cache_key = (int(embedding_type_id), int(layer_index), metric_name)
        cached = self._client._ann_index_presence_cache.get(cache_key)
        if cached is None:
            has_index = bool(
                self._client.scalar(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_indexes
                        WHERE schemaname = 'public'
                          AND tablename = 'sequence_embeddings'
                          AND indexdef ILIKE '%%USING hnsw%%'
                          AND indexdef ILIKE %s
                          AND indexdef ILIKE %s
                          AND indexdef ILIKE %s
                    );
                    """,
                    (
                        f"%embedding_type_id = {int(embedding_type_id)}%",
                        f"%layer_index = {int(layer_index)}%",
                        f"%{opclass}%",
                    ),
                )
            )
            self._client._ann_index_presence_cache[cache_key] = has_index
            cached = has_index
        if (not cached) and cache_key not in self._client._ann_index_warned:
            warnings.warn(
                "No matching HNSW index detected for sequence_embeddings "
                f"(embedding_type_id={embedding_type_id}, layer_index={layer_index}, metric={metric_name}). "
                "Nearest-neighbor search will run in slow mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._client._ann_index_warned.add(cache_key)


def _search_state_cache_slot(backend: ResolvedSearchBackend, device: str | None) -> str:
    """Return the bounded cache slot for a backend's physical compute resource."""
    if backend == "faiss_cpu":
        return "cpu"
    return str(device or "")
