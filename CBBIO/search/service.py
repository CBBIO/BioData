"""Search backend orchestration for BioData neighbor lookup."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, cast

from ..BioData import BioDataError, NotFoundError, cursor, embedding_dimension, metric_opclass, metric_operator
from ..types import DistanceMetric, Neighbor, SearchBackend
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
    import_cuvs,
    import_faiss,
    import_torch,
    normalize_distance,
    preferred_cuvs_device,
    preferred_faiss_device,
    preferred_torch_device,
    prepare_index_vectors,
    tensor_to_list,
    torch_normalize,
)


class SearchService:
    """Encapsulates backend routing and neighbor-search implementations."""

    def __init__(self, client: Any) -> None:
        self._client = client

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

    def _probe_cuda_memory(self, *, device: Optional[str]) -> Optional[tuple[int, int]]:
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
    ) -> Optional[int]:
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
    ) -> Optional[int]:
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
        device: Optional[str],
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
        k: Optional[int] = None,
        *,
        metric: Optional[DistanceMetric] = None,
        exclude_protein_ids: Optional[Sequence[str]] = None,
        use_ann: bool = False,
        ann_ef_search: int = 200,
        ann_candidate_pool: Optional[int] = None,
        backend: Optional[SearchBackend] = None,
        device: Optional[str] = None,
    ) -> List[Neighbor]:
        effective_metric = metric or self._client.default_metric
        effective_k = self._client.default_k if k is None else int(k)
        if effective_k < 1:
            raise BioDataError("k must be >= 1")

        requested_backend = cast(SearchBackend, backend or self._client.default_backend)
        resolved = self._client._resolve_search_backend(
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=1,
            ann_requested=use_ann,
            device=device,
        )
        resolved = self._apply_auto_gpu_heuristics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=1,
            device=device,
        )
        excluded_ids = [str(value) for value in (exclude_protein_ids or [])]

        if resolved.backend == "pgvector":
            neighbors = self._client._find_nearest_neighbors_pgvector(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=effective_k,
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
                k=effective_k,
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
                k=effective_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                use_ann=resolved.ann_used,
            )
        elif resolved.backend == "cuvs_gpu":
            neighbors = self._client._find_nearest_neighbors_cuvs(
                query_embedding,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=effective_k,
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
                k=effective_k,
                metric=effective_metric,
                exclude_protein_ids=excluded_ids,
                device=resolved.device,
            )

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

    def find_nearest_neighbors_for_proteins(
        self,
        protein_ids: Sequence[str],
        embedding_type_id: int,
        layer_index: int = 0,
        k: Optional[int] = None,
        *,
        metric: Optional[DistanceMetric] = None,
        include_query: bool = False,
        backend: Optional[SearchBackend] = None,
        device: Optional[str] = None,
    ) -> Dict[str, List[Neighbor]]:
        ids = [str(value) for value in protein_ids]
        if not ids:
            return {}

        effective_metric = metric or self._client.default_metric
        effective_k = self._client.default_k if k is None else int(k)
        if effective_k < 1:
            raise BioDataError("k must be >= 1")
        requested_backend = cast(SearchBackend, backend or self._client.default_backend)
        resolved = self._client._resolve_search_backend(
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=len(ids),
            ann_requested=False,
            device=device,
        )
        resolved = self._apply_auto_gpu_heuristics(
            resolved,
            requested_backend=requested_backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=effective_metric,
            batch_size=len(ids),
            device=device,
        )

        if resolved.backend == "pgvector":
            grouped = self._client._find_nearest_neighbors_for_proteins_pgvector(
                ids,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=effective_k,
                metric=effective_metric,
                include_query=include_query,
            )
        else:
            query_map = self._client.get_protein_embeddings(ids, embedding_type_id=embedding_type_id, layer_index=layer_index)
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
                        k=effective_k,
                        metric=effective_metric,
                        include_query=include_query,
                        device=resolved.device,
                    )
                elif resolved.backend == "faiss_cpu":
                    partial = self._client._find_nearest_neighbors_for_queries_faiss_cpu(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=effective_k,
                        metric=effective_metric,
                        include_query=include_query,
                        use_ann=False,
                    )
                elif resolved.backend == "cuvs_gpu":
                    partial = self._client._find_nearest_neighbors_for_queries_cuvs(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=effective_k,
                        metric=effective_metric,
                        include_query=include_query,
                        device=resolved.device,
                        use_ann=False,
                    )
                else:
                    partial = self._client._find_nearest_neighbors_for_queries_torch(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=effective_k,
                        metric=effective_metric,
                        include_query=include_query,
                        device=resolved.device,
                    )
                grouped.update(partial)

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
        ann_candidate_pool: Optional[int],
    ) -> List[Neighbor]:
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
    ) -> Dict[str, List[Neighbor]]:
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
        self._client._warn_if_missing_ann_index(embedding_type_id, layer_index, metric)

        exclude_clause = ""
        if not include_query:
            exclude_clause = " AND se2.sequence_id <> q.query_sequence_id "

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
            "    ORDER BY c.distance"
            ") n ON TRUE "
            "ORDER BY q.query_protein_id, n.distance;"
        )

        params = (
            [str(value) for value in protein_ids],
            embedding_type_id,
            layer_index,
            embedding_type_id,
            layer_index,
            k,
        )

        with cursor(conn) as cur:
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
        device: Optional[str],
        use_ann: bool,
    ) -> List[Neighbor]:
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
        device: Optional[str],
    ) -> List[Neighbor]:
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
        device: Optional[str],
        use_ann: bool,
    ) -> List[Neighbor]:
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
        device: Optional[str],
    ) -> Dict[str, List[Neighbor]]:
        state = self._client._get_or_load_gpu_search_state(
            backend="faiss_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=False,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded: Dict[str, Set[str]] = {
            str(query_id): set() if include_query else {str(query_id)} for query_id in query_ids
        }
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
    ) -> Dict[str, List[Neighbor]]:
        state = self._client._get_or_load_gpu_search_state(
            backend="faiss_cpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device="cpu",
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded: Dict[str, Set[str]] = {
            str(query_id): set() if include_query else {str(query_id)} for query_id in query_ids
        }
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
        device: Optional[str],
        use_ann: bool,
    ) -> Dict[str, List[Neighbor]]:
        state = self._client._get_or_load_gpu_search_state(
            backend="cuvs_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=use_ann,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded: Dict[str, Set[str]] = {
            str(query_id): set() if include_query else {str(query_id)} for query_id in query_ids
        }
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
        device: Optional[str],
    ) -> Dict[str, List[Neighbor]]:
        state = self._client._get_or_load_gpu_search_state(
            backend="torch_gpu",
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_requested=False,
        )
        query_matrix = as_numpy_matrix(query_vectors)
        per_query_excluded: Dict[str, Set[str]] = {
            str(query_id): set() if include_query else {str(query_id)} for query_id in query_ids
        }
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
        import numpy as np

        if state.faiss_index is None:
            raise BioDataError("FAISS backend selected without an initialized FAISS index.")

        requested = min(len(state.protein_ids), max(k, 1))
        if per_query_excluded:
            requested = min(
                len(state.protein_ids),
                max(
                    requested,
                    k + max(
                        sum(len(state.protein_rows.get(protein_id, [])) for protein_id in excluded)
                        for excluded in per_query_excluded.values()
                    ),
                ),
            )
        if requested < 1:
            requested = 1

        grouped: Dict[str, List[Neighbor]] = {str(query_id): [] for query_id in query_ids}
        while True:
            distances, indices = state.faiss_index.search(np.asarray(query_vectors, dtype=np.float32), requested)
            for row_index, query_id in enumerate(query_ids):
                grouped[str(query_id)] = self._client._neighbors_from_candidate_rows(
                    state,
                    candidate_indices=indices[row_index],
                    candidate_distances=distances[row_index],
                    k=k,
                    excluded_protein_ids=per_query_excluded.get(str(query_id), set()),
                    l2_squared=True,
                )
            if all(len(values) >= min(k, len(state.protein_ids)) for values in grouped.values()) or requested >= len(state.protein_ids):
                return grouped
            requested = min(len(state.protein_ids), max(requested * 2, requested + 8))

    def search_cuvs_state(
        self,
        state: GpuSearchState,
        *,
        query_ids: Sequence[str],
        query_vectors: Any,
        k: int,
        per_query_excluded: Mapping[str, Set[str]],
    ) -> Dict[str, List[Neighbor]]:
        cupy = import_cupy()
        import_cuvs()
        from cuvs.neighbors import brute_force as cuvs_brute_force  # type: ignore
        from cuvs.neighbors import cagra as cuvs_cagra  # type: ignore

        if state.cuvs_index is None:
            raise BioDataError("cuVS backend selected without an initialized cuVS index.")

        requested = min(len(state.protein_ids), max(k, 1))
        if per_query_excluded:
            requested = min(
                len(state.protein_ids),
                max(
                    requested,
                    k + max(
                        sum(len(state.protein_rows.get(protein_id, [])) for protein_id in excluded)
                        for excluded in per_query_excluded.values()
                    ),
                ),
            )
        if requested < 1:
            requested = 1

        query_matrix = cupy.asarray(query_vectors, dtype=cupy.float32)
        grouped: Dict[str, List[Neighbor]] = {str(query_id): [] for query_id in query_ids}
        while True:
            if state.ann_enabled:
                search_params = cast(Any, cuvs_cagra).SearchParams()
                distances, indices = cast(Any, cuvs_cagra).search(
                    search_params, state.cuvs_index, query_matrix, requested
                )
            else:
                distances, indices = cast(Any, cuvs_brute_force).search(
                    state.cuvs_index, query_matrix, requested
                )

            host_distances = cupy.asnumpy(distances)
            host_indices = cupy.asnumpy(indices)
            for row_index, query_id in enumerate(query_ids):
                grouped[str(query_id)] = self._client._neighbors_from_candidate_rows(
                    state,
                    candidate_indices=host_indices[row_index].tolist(),
                    candidate_distances=host_distances[row_index].tolist(),
                    k=k,
                    excluded_protein_ids=per_query_excluded.get(str(query_id), set()),
                    l2_squared=state.metric == "l2",
                )
            if all(len(values) >= min(k, len(state.protein_ids)) for values in grouped.values()) or requested >= len(state.protein_ids):
                return grouped
            requested = min(len(state.protein_ids), max(requested * 2, requested + 8))

    def search_torch_state(
        self,
        state: GpuSearchState,
        *,
        query_ids: Sequence[str],
        query_vectors: Any,
        k: int,
        per_query_excluded: Mapping[str, Set[str]],
    ) -> Dict[str, List[Neighbor]]:
        torch = import_torch()
        query_tensor = torch.as_tensor(query_vectors, dtype=torch.float32, device=state.device)
        if query_tensor.ndim == 1:
            query_tensor = query_tensor.reshape(1, -1)

        if state.metric == "cosine":
            query_tensor = torch_normalize(query_tensor, torch=torch)

        index_tensor = state.vectors
        if state.metric == "l2":
            raw_scores = torch.cdist(query_tensor, index_tensor, p=2.0)
            sort_desc = False
        else:
            raw_scores = torch.matmul(query_tensor, index_tensor.transpose(0, 1))
            sort_desc = True

        grouped: Dict[str, List[Neighbor]] = {}
        for row_index, query_id in enumerate(query_ids):
            query_id_str = str(query_id)
            excluded = set(per_query_excluded.get(query_id_str, set()))
            scores = raw_scores[row_index].clone()
            for excluded_id in excluded:
                for candidate_index in state.protein_rows.get(excluded_id, []):
                    scores[candidate_index] = float("-inf") if sort_desc else float("inf")
            top_k = min(k, scores.shape[0])
            if top_k < 1:
                grouped[query_id_str] = []
                continue
            values, indices = torch.topk(scores, k=top_k, largest=sort_desc)
            grouped[query_id_str] = self._client._neighbors_from_candidate_rows(
                state,
                candidate_indices=tensor_to_list(indices),
                candidate_distances=tensor_to_list(values),
                k=k,
                excluded_protein_ids=excluded,
                l2_squared=False,
            )
        return grouped

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
        neighbors: List[Neighbor] = []
        for raw_index, raw_distance in zip(candidate_indices, candidate_distances):
            candidate_index = int(raw_index)
            if candidate_index < 0 or candidate_index >= len(state.protein_ids):
                continue
            protein_id = state.protein_ids[candidate_index]
            if protein_id in excluded_protein_ids:
                continue
            neighbors.append(
                Neighbor(
                    protein_id=protein_id,
                    layer_index=state.layer_index,
                    distance=normalize_distance(metric=state.metric, value=raw_distance, l2_squared=l2_squared),
                )
            )
            if len(neighbors) >= k:
                break
        return neighbors

    def resolve_search_backend(
        self,
        *,
        requested_backend: SearchBackend,
        embedding_type_id: int,
        layer_index: int,
        metric: DistanceMetric,
        batch_size: int,
        ann_requested: bool,
        device: Optional[str],
    ) -> ResolvedBackend:
        requested = str(requested_backend).strip().lower()
        if requested not in {"auto", "gpu", "pgvector", "faiss_cpu", "faiss_gpu", "cuvs_gpu", "torch_gpu"}:
            raise BioDataError(f"Unsupported search backend: {requested_backend!r}")

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

    def detect_backend_availability(self, *, device: Optional[str]) -> BackendAvailability:
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
        device: Optional[str],
        ann_requested: bool,
    ) -> GpuSearchState:
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
            return cast(GpuSearchState, self._client._gpu_search_state)
        state = self._client._load_gpu_search_state(
            backend=backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=resolved_device,
            ann_requested=ann_requested,
        )
        self._client._gpu_search_state = state
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
        import numpy as np

        protein_ids, vectors = self._client._load_search_vectors(embedding_type_id=embedding_type_id, layer_index=layer_index)
        normalized_vectors = prepare_index_vectors(vectors, metric=metric)
        protein_rows: Dict[str, List[int]] = {}
        for index, protein_id in enumerate(protein_ids):
            protein_rows.setdefault(protein_id, []).append(index)

        if backend == "torch_gpu":
            torch = import_torch()
            tensor = torch.as_tensor(normalized_vectors, dtype=torch.float32, device=device)
            return GpuSearchState(backend, embedding_type_id, layer_index, metric, device, False, protein_ids, protein_rows, tensor)

        if backend == "cuvs_gpu":
            cupy = import_cupy()
            import_cuvs()
            from cuvs.neighbors import brute_force as cuvs_brute_force  # type: ignore
            from cuvs.neighbors import cagra as cuvs_cagra  # type: ignore
            dataset = cupy.asarray(normalized_vectors, dtype=cupy.float32)
            metric_name = "sqeuclidean" if metric == "l2" else str(metric)
            if ann_requested:
                index_params = cast(Any, cuvs_cagra).IndexParams(metric=metric_name)
                index = cast(Any, cuvs_cagra).build(index_params, dataset)
            else:
                index = cast(Any, cuvs_brute_force).build(dataset, metric=metric_name)
            return GpuSearchState(
                backend=backend,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                metric=metric,
                device=device,
                ann_enabled=ann_requested,
                protein_ids=protein_ids,
                protein_rows=protein_rows,
                vectors=dataset,
                cuvs_index=index,
            )

        faiss = import_faiss()
        dim = int(normalized_vectors.shape[1])
        metric_type = getattr(faiss, "METRIC_L2" if metric == "l2" else "METRIC_INNER_PRODUCT")
        if ann_requested:
            quantizer = faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
            nlist = max(1, min(int(max(4, round(np.sqrt(max(len(protein_ids), 1))))), len(protein_ids)))
            cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric_type)
            cpu_index.train(np.asarray(normalized_vectors, dtype=np.float32))
            cpu_index.add(np.asarray(normalized_vectors, dtype=np.float32))
            nprobe = max(1, min(nlist, 8))
            if hasattr(cpu_index, "nprobe"):
                cpu_index.nprobe = nprobe
        else:
            cpu_index = faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
            cpu_index.add(np.asarray(normalized_vectors, dtype=np.float32))

        if backend == "faiss_cpu":
            return GpuSearchState(
                backend=backend,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                metric=metric,
                device="cpu",
                ann_enabled=ann_requested,
                protein_ids=protein_ids,
                protein_rows=protein_rows,
                vectors=np.asarray(normalized_vectors, dtype=np.float32),
                faiss_index=cpu_index,
            )

        resources = getattr(faiss, "StandardGpuResources")()
        gpu_index = getattr(faiss, "index_cpu_to_gpu")(resources, cuda_device_index(device), cpu_index)
        return GpuSearchState(
            backend=backend,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            device=device,
            ann_enabled=ann_requested,
            protein_ids=protein_ids,
            protein_rows=protein_rows,
            vectors=np.asarray(normalized_vectors, dtype=np.float32),
            faiss_index=gpu_index,
            faiss_resources=resources,
        )

    def load_search_vectors(self, *, embedding_type_id: int, layer_index: int) -> tuple[List[str], Any]:
        rows = self._client.query_all(
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
        device: Optional[str],
        ann_enabled: bool,
    ) -> bool:
        state = self._client._gpu_search_state
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
        warning_message: Optional[str] = None
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
