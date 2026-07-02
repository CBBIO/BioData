"""Backend-neutral in-memory vector search engines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Set, cast

from ..BioData import BioDataError
from ..types import DistanceMetric, Neighbor
from .types import GpuSearchState, ResolvedSearchBackend
from .utils import (
    as_numpy_matrix,
    cuda_device_index,
    import_cupy,
    import_cuvs,
    import_faiss,
    import_torch,
    normalize_distance,
    prepare_index_vectors,
    tensor_to_list,
    torch_normalize,
)


def build_search_state(
    *,
    backend: ResolvedSearchBackend,
    item_ids: Sequence[str],
    vectors: Any,
    metric: DistanceMetric,
    device: str,
    ann_requested: bool,
    embedding_type_id: int = -1,
    layer_index: int = 0,
) -> GpuSearchState:
    """Build an in-memory search state from ids and vectors."""
    import numpy as np

    protein_ids = [str(item_id) for item_id in item_ids]
    matrix = as_numpy_matrix(vectors)
    if len(protein_ids) != int(matrix.shape[0]):
        raise BioDataError("Search ids and vectors must contain the same number of rows.")

    normalized_vectors = prepare_index_vectors(matrix, metric=metric)
    protein_rows: dict[str, list[int]] = {}
    for index, protein_id in enumerate(protein_ids):
        protein_rows.setdefault(protein_id, []).append(index)

    if backend == "torch_gpu":
        torch = import_torch()
        tensor = torch.as_tensor(normalized_vectors, dtype=torch.float32, device=device)
        return GpuSearchState(
            backend=backend,
            embedding_type_id=int(embedding_type_id),
            layer_index=int(layer_index),
            metric=metric,
            device=device,
            ann_enabled=False,
            protein_ids=protein_ids,
            protein_rows=protein_rows,
            vectors=tensor,
        )

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
            embedding_type_id=int(embedding_type_id),
            layer_index=int(layer_index),
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
            embedding_type_id=int(embedding_type_id),
            layer_index=int(layer_index),
            metric=metric,
            device="cpu",
            ann_enabled=ann_requested,
            protein_ids=protein_ids,
            protein_rows=protein_rows,
            vectors=np.asarray(normalized_vectors, dtype=np.float32),
            faiss_index=cpu_index,
        )

    resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(resources, cuda_device_index(device), cpu_index)
    return GpuSearchState(
        backend=backend,
        embedding_type_id=int(embedding_type_id),
        layer_index=int(layer_index),
        metric=metric,
        device=device,
        ann_enabled=ann_requested,
        protein_ids=protein_ids,
        protein_rows=protein_rows,
        vectors=np.asarray(normalized_vectors, dtype=np.float32),
        faiss_index=gpu_index,
        faiss_resources=resources,
    )


def search_state(
    state: GpuSearchState,
    *,
    query_ids: Sequence[str],
    query_vectors: Any,
    k: int,
    per_query_excluded: Mapping[str, Set[str]] | None = None,
) -> dict[str, list[Neighbor]]:
    """Search an initialized in-memory state for query vectors."""
    if state.backend in {"faiss_cpu", "faiss_gpu"}:
        return search_faiss_state(
            state,
            query_ids=query_ids,
            query_vectors=query_vectors,
            k=k,
            per_query_excluded=per_query_excluded or {},
        )
    if state.backend == "cuvs_gpu":
        return search_cuvs_state(
            state,
            query_ids=query_ids,
            query_vectors=query_vectors,
            k=k,
            per_query_excluded=per_query_excluded or {},
        )
    if state.backend == "torch_gpu":
        return search_torch_state(
            state,
            query_ids=query_ids,
            query_vectors=query_vectors,
            k=k,
            per_query_excluded=per_query_excluded or {},
        )
    raise BioDataError(f"Unsupported in-memory search backend: {state.backend!r}.")


def search_faiss_state(
    state: GpuSearchState,
    *,
    query_ids: Sequence[str],
    query_vectors: Any,
    k: int,
    per_query_excluded: Mapping[str, Set[str]],
) -> dict[str, list[Neighbor]]:
    """Search an initialized FAISS state for query vectors."""
    import numpy as np

    if state.faiss_index is None:
        raise BioDataError("FAISS backend selected without an initialized FAISS index.")

    requested = _initial_requested_count(state, k=k, per_query_excluded=per_query_excluded)
    grouped: dict[str, list[Neighbor]] = {str(query_id): [] for query_id in query_ids}
    while True:
        distances, indices = state.faiss_index.search(np.asarray(query_vectors, dtype=np.float32), requested)
        for row_index, query_id in enumerate(query_ids):
            grouped[str(query_id)] = neighbors_from_candidate_rows(
                state,
                candidate_indices=indices[row_index],
                candidate_distances=distances[row_index],
                k=k,
                excluded_protein_ids=per_query_excluded.get(str(query_id), set()),
                l2_squared=True,
            )
        if _has_enough_neighbors(grouped, state, k=k) or requested >= len(state.protein_ids):
            return grouped
        requested = min(len(state.protein_ids), max(requested * 2, requested + 8))


def search_cuvs_state(
    state: GpuSearchState,
    *,
    query_ids: Sequence[str],
    query_vectors: Any,
    k: int,
    per_query_excluded: Mapping[str, Set[str]],
) -> dict[str, list[Neighbor]]:
    """Search an initialized cuVS state for query vectors."""
    cupy = import_cupy()
    import_cuvs()
    from cuvs.neighbors import brute_force as cuvs_brute_force  # type: ignore
    from cuvs.neighbors import cagra as cuvs_cagra  # type: ignore

    if state.cuvs_index is None:
        raise BioDataError("cuVS backend selected without an initialized cuVS index.")

    requested = _initial_requested_count(state, k=k, per_query_excluded=per_query_excluded)
    query_matrix = cupy.asarray(query_vectors, dtype=cupy.float32)
    grouped: dict[str, list[Neighbor]] = {str(query_id): [] for query_id in query_ids}
    while True:
        if state.ann_enabled:
            search_params = cast(Any, cuvs_cagra).SearchParams()
            distances, indices = cast(Any, cuvs_cagra).search(
                search_params,
                state.cuvs_index,
                query_matrix,
                requested,
            )
        else:
            distances, indices = cast(Any, cuvs_brute_force).search(
                state.cuvs_index,
                query_matrix,
                requested,
            )

        host_distances = cupy.asnumpy(distances)
        host_indices = cupy.asnumpy(indices)
        for row_index, query_id in enumerate(query_ids):
            grouped[str(query_id)] = neighbors_from_candidate_rows(
                state,
                candidate_indices=host_indices[row_index].tolist(),
                candidate_distances=host_distances[row_index].tolist(),
                k=k,
                excluded_protein_ids=per_query_excluded.get(str(query_id), set()),
                l2_squared=state.metric == "l2",
            )
        if _has_enough_neighbors(grouped, state, k=k) or requested >= len(state.protein_ids):
            return grouped
        requested = min(len(state.protein_ids), max(requested * 2, requested + 8))


def search_torch_state(
    state: GpuSearchState,
    *,
    query_ids: Sequence[str],
    query_vectors: Any,
    k: int,
    per_query_excluded: Mapping[str, Set[str]],
) -> dict[str, list[Neighbor]]:
    """Search an initialized torch tensor state for query vectors."""
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

    grouped: dict[str, list[Neighbor]] = {}
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
        grouped[query_id_str] = neighbors_from_candidate_rows(
            state,
            candidate_indices=tensor_to_list(indices),
            candidate_distances=tensor_to_list(values),
            k=k,
            excluded_protein_ids=excluded,
            l2_squared=False,
        )
    return grouped


def neighbors_from_candidate_rows(
    state: GpuSearchState,
    *,
    candidate_indices: Sequence[Any],
    candidate_distances: Sequence[Any],
    k: int,
    excluded_protein_ids: set[str],
    l2_squared: bool,
) -> list[Neighbor]:
    """Convert backend candidate rows into neighbor records."""
    neighbors: list[Neighbor] = []
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
                distance=normalize_distance(
                    metric=state.metric,
                    value=raw_distance,
                    l2_squared=l2_squared,
                ),
            )
        )
        if len(neighbors) >= k:
            break
    return neighbors


def _initial_requested_count(
    state: GpuSearchState,
    *,
    k: int,
    per_query_excluded: Mapping[str, Set[str]],
) -> int:
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
    return max(1, requested)


def _has_enough_neighbors(
    grouped: Mapping[str, Sequence[Neighbor]],
    state: GpuSearchState,
    *,
    k: int,
) -> bool:
    return all(len(values) >= min(k, len(state.protein_ids)) for values in grouped.values())


__all__ = [
    "build_search_state",
    "neighbors_from_candidate_rows",
    "search_cuvs_state",
    "search_faiss_state",
    "search_state",
    "search_torch_state",
]
