#!/usr/bin/env python3
"""Measure nearest-neighbor search as the candidate corpus grows.

Example:
    poetry run python scripts/benchmark_neighbor_search_corpus_scaling.py \
        --corpus-sizes 10000 100000 1000000 --query-counts 100 1000 5000 \
        --backends pgvector faiss_cpu cuvs_gpu --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Set, cast
from urllib.parse import quote

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.BioData import BioDataClient
from CBBIO.search.engines import build_search_state, search_state
from CBBIO.search.utils import as_numpy_matrix
from CBBIO.types import DistanceMetric, Neighbor, SearchBackend


DEFAULT_CORPUS_SIZES = [10_000, 100_000, 1_000_000]
DEFAULT_QUERY_COUNTS = [100, 1_000, 5_000]
DEFAULT_BACKENDS: list[SearchBackend] = ["pgvector", "faiss_cpu", "torch_gpu", "faiss_gpu", "cuvs_gpu"]
_TEMP_TABLE_NAME = "benchmark_neighbor_candidates"
_TEMP_ANN_INDEX_NAME = "benchmark_neighbor_candidates_hnsw"
_SearchRunner = Callable[[Sequence[str], Any, int], None]

_METRIC_OPCLASS: dict[DistanceMetric, str] = {
    "l2": "halfvec_l2_ops",
    "cosine": "halfvec_cosine_ops",
    "inner_product": "halfvec_ip_ops",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark exact or ANN nearest-neighbor search over increasing candidate corpus sizes.",
    )
    parser.add_argument(
        "--corpus-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_CORPUS_SIZES,
        help="Candidate corpus sizes to benchmark. Default: 10000 100000 1000000",
    )
    query_count_group = parser.add_mutually_exclusive_group()
    query_count_group.add_argument(
        "--query-counts",
        type=int,
        nargs="+",
        default=None,
        help="Query batch sizes to benchmark. Default: 100 1000 5000",
    )
    query_count_group.add_argument(
        "--query-count",
        type=int,
        default=None,
        help="Deprecated single query batch size; use --query-counts.",
    )
    parser.add_argument(
        "--query-pool-size",
        type=int,
        default=None,
        help="Stored query embedding pool size before subsampling. Default: max(10000, largest query batch).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Warm-search repetitions per corpus size and backend. Default: 5.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["pgvector", "faiss_cpu", "torch_gpu", "faiss_gpu", "cuvs_gpu"],
        help="Search backends to benchmark.",
    )
    parser.add_argument(
        "--embedding-type-id",
        type=int,
        default=1,
        help="Stored embedding type to benchmark. Default: 1.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Stored embedding layer to benchmark. Default: 0.",
    )
    parser.add_argument(
        "--metric",
        choices=["l2", "cosine", "inner_product"],
        default="cosine",
        help="Distance metric. Default: cosine.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Neighbors returned for each query. Default: 10.",
    )
    parser.add_argument(
        "--use-ann",
        action="store_true",
        help="Use ANN indexes: HNSW for pgvector, IVF for FAISS, and CAGRA for cuVS.",
    )
    parser.add_argument(
        "--ann-ef-search",
        type=int,
        default=200,
        help="HNSW ef_search for pgvector ANN. Default: 200.",
    )
    parser.add_argument(
        "--ann-candidate-pool",
        type=int,
        default=None,
        help="Candidates reranked per query by pgvector ANN. Default: max(20*k, 200).",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="corpus-scaling-v1",
        help="Stable seed used to order candidate and query proteins.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit accelerator device for GPU backends.",
    )
    parser.add_argument(
        "--dsn",
        type=str,
        default=None,
        help=(
            "PostgreSQL connection DSN. Overrides POSTGRES_HOST, POSTGRES_PORT, "
            "POSTGRES_USER, POSTGRES_PASSWORD, and POSTGRES_DB."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "benchmark_neighbor_search_corpus_scaling.json",
        help="JSON output path. Default: benchmark_neighbor_search_corpus_scaling.json",
    )
    args = parser.parse_args()
    if args.query_counts is None:
        args.query_counts = [args.query_count] if args.query_count is not None else DEFAULT_QUERY_COUNTS
    args.query_pool_size = args.query_pool_size or max(10_000, max(args.query_counts))
    return args


def _resolve_dsn(dsn: str | None) -> str | None:
    """Return an explicit DSN or build one from standard PostgreSQL variables."""
    if dsn is not None:
        return dsn

    environment_names = (
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
    )
    values = {name: os.getenv(name) for name in environment_names}
    present_names = [name for name, value in values.items() if value is not None]
    if not present_names:
        return None

    missing_names = [name for name, value in values.items() if not value]
    if missing_names:
        missing = ", ".join(missing_names)
        raise SystemExit(
            "Set all POSTGRES_* connection variables or pass --dsn. "
            f"Missing: {missing}."
        )

    host = cast(str, values["POSTGRES_HOST"])
    port = cast(str, values["POSTGRES_PORT"])
    user = quote(cast(str, values["POSTGRES_USER"]), safe="")
    password = quote(cast(str, values["POSTGRES_PASSWORD"]), safe="")
    database = quote(cast(str, values["POSTGRES_DB"]), safe="")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def _summarize_timings(timings: Sequence[float]) -> Dict[str, float]:
    return {
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }


def _metric_operator(metric: DistanceMetric) -> str:
    operators = {
        "l2": "<->",
        "cosine": "<=>",
        "inner_product": "<#>",
    }
    return operators[metric]


def _validate_args(args: argparse.Namespace) -> None:
    if any(query_count < 1 for query_count in args.query_counts):
        raise SystemExit("--query-counts values must be >= 1")
    if list(args.query_counts) != sorted(set(args.query_counts)):
        raise SystemExit("--query-counts must be unique and sorted in ascending order")
    if args.query_pool_size < max(args.query_counts):
        raise SystemExit("--query-pool-size must be at least the largest --query-counts value")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if args.ann_ef_search < 1:
        raise SystemExit("--ann-ef-search must be >= 1")
    if args.ann_candidate_pool is not None and args.ann_candidate_pool < args.k:
        raise SystemExit("--ann-candidate-pool must be >= --k")
    if any(size < 1 for size in args.corpus_sizes):
        raise SystemExit("--corpus-sizes values must be >= 1")
    if list(args.corpus_sizes) != sorted(set(args.corpus_sizes)):
        raise SystemExit("--corpus-sizes must be unique and sorted in ascending order")


def _load_query_embedding_pool(
    client: BioDataClient,
    *,
    embedding_type_id: int,
    layer_index: int,
    query_pool_size: int,
    seed: str,
) -> tuple[list[str], Any]:
    rows = client.query_all(
        """
        SELECT p.id AS protein_id, se.embedding
        FROM sequence_embeddings se
        JOIN sequence s ON s.id = se.sequence_id
        JOIN protein p ON p.sequence_id = s.id
        WHERE se.embedding_type_id = %s
          AND se.layer_index = %s
        ORDER BY md5(p.id || %s)
        LIMIT %s;
        """,
        (embedding_type_id, layer_index, f"query:{seed}", query_pool_size),
    )
    if len(rows) != query_pool_size:
        raise SystemExit(
            f"Requested {query_pool_size} query embeddings but found only {len(rows)} for "
            f"embedding_type_id={embedding_type_id}, layer_index={layer_index}.",
        )
    return [str(row["protein_id"]) for row in rows], as_numpy_matrix([row["embedding"] for row in rows])


def _build_query_samples(
    query_ids: Sequence[str],
    query_vectors: Any,
    *,
    query_counts: Sequence[int],
    repeats: int,
    seed: str,
) -> Dict[int, List[tuple[list[str], Any]]]:
    rng = random.Random(seed)
    samples: Dict[int, List[tuple[list[str], Any]]] = {}
    for query_count in query_counts:
        if query_count > len(query_ids):
            raise SystemExit(
                f"Query batch size {query_count} exceeds query pool size {len(query_ids)}.",
            )
        runs: list[tuple[list[str], Any]] = []
        for _ in range(repeats):
            indices = rng.sample(range(len(query_ids)), query_count)
            runs.append(([query_ids[index] for index in indices], query_vectors[indices]))
        samples[query_count] = runs
    return samples


def _materialize_candidate_corpus(
    client: BioDataClient,
    *,
    embedding_type_id: int,
    layer_index: int,
    corpus_size: int,
    seed: str,
) -> float:
    started = time.perf_counter()
    client.execute(f"DROP TABLE IF EXISTS {_TEMP_TABLE_NAME};")
    client.execute(
        f"""
        CREATE TEMP TABLE {_TEMP_TABLE_NAME} AS
        SELECT p.id AS protein_id, se.layer_index, se.embedding
        FROM sequence_embeddings se
        JOIN sequence s ON s.id = se.sequence_id
        JOIN protein p ON p.sequence_id = s.id
        WHERE se.embedding_type_id = %s
          AND se.layer_index = %s
        ORDER BY md5(p.id || %s)
        LIMIT %s;
        """,
        (embedding_type_id, layer_index, f"candidate:{seed}", corpus_size),
    )
    count_row = client.query_one(f"SELECT count(*) AS count FROM {_TEMP_TABLE_NAME};")
    count = 0 if count_row is None else int(count_row["count"])
    if count != corpus_size:
        raise SystemExit(
            f"Requested corpus_size={corpus_size} but found only {count} matching stored embeddings.",
        )
    return time.perf_counter() - started


def _create_temporary_ann_index(
    client: BioDataClient,
    *,
    embedding_dimension: int,
    metric: DistanceMetric,
) -> float:
    started = time.perf_counter()
    client.execute(
        f"CREATE INDEX {_TEMP_ANN_INDEX_NAME} ON {_TEMP_TABLE_NAME} "
        f"USING hnsw ((embedding::halfvec({embedding_dimension})) {_METRIC_OPCLASS[metric]});",
    )
    return time.perf_counter() - started


def _load_candidate_vectors(client: BioDataClient) -> tuple[list[str], Any, float]:
    started = time.perf_counter()
    rows = client.query_all(
        f"SELECT protein_id, embedding FROM {_TEMP_TABLE_NAME} ORDER BY protein_id;",
    )
    return (
        [str(row["protein_id"]) for row in rows],
        as_numpy_matrix([row["embedding"] for row in rows]),
        time.perf_counter() - started,
    )


def _search_pgvector(
    client: BioDataClient,
    *,
    query_ids: Sequence[str],
    query_vectors: Any,
    metric: DistanceMetric,
    k: int,
    use_ann: bool = False,
    ann_ef_search: int = 200,
    ann_candidate_pool: int | None = None,
) -> Dict[str, List[Neighbor]]:
    value_rows = ", ".join("(%s, %s::halfvec)" for _ in query_ids)
    params: list[Any] = [
        value
        for query_id, query_vector in zip(query_ids, query_vectors.tolist())
        for value in (query_id, query_vector)
    ]
    operator = _metric_operator(metric)
    if use_ann:
        candidate_limit = ann_candidate_pool or max(k * 20, 200)
        params.extend([candidate_limit, k])
        sql = (
            "WITH query_embeddings(query_id, query_embedding) AS (VALUES "
            f"{value_rows}"
            ") "
            "SELECT q.query_id, n.protein_id, n.layer_index, n.distance "
            "FROM query_embeddings q "
            "LEFT JOIN LATERAL ("
            "    WITH ann_candidates AS ("
            "        SELECT c.protein_id, c.layer_index, c.embedding "
            f"        FROM {_TEMP_TABLE_NAME} c "
            f"        ORDER BY c.embedding {operator} q.query_embedding "
            "        LIMIT %s"
            "    ) "
            "    SELECT c.protein_id, c.layer_index, "
            f"           c.embedding {operator} q.query_embedding AS distance "
            "    FROM ann_candidates c "
            "    WHERE c.protein_id <> q.query_id "
            "    ORDER BY distance "
            "    LIMIT %s"
            ") n ON TRUE "
            "ORDER BY q.query_id, n.distance;"
        )
    else:
        params.append(k)
        sql = (
            "WITH query_embeddings(query_id, query_embedding) AS (VALUES "
            f"{value_rows}"
            ") "
            "SELECT q.query_id, n.protein_id, n.layer_index, n.distance "
            "FROM query_embeddings q "
            "LEFT JOIN LATERAL ("
            "    SELECT c.protein_id, c.layer_index, "
            f"           c.embedding {operator} q.query_embedding AS distance "
            f"    FROM {_TEMP_TABLE_NAME} c "
            "    WHERE c.protein_id <> q.query_id "
            f"    ORDER BY c.embedding {operator} q.query_embedding "
            "    LIMIT %s"
            ") n ON TRUE "
            "ORDER BY q.query_id, n.distance;"
        )
    if use_ann:
        client.execute(f"SET hnsw.ef_search = {ann_ef_search};")
    rows = client.query_all(sql, tuple(params))
    grouped: Dict[str, List[Neighbor]] = {str(query_id): [] for query_id in query_ids}
    for row in rows:
        if row["protein_id"] is None:
            continue
        grouped[str(row["query_id"])].append(
            Neighbor(
                protein_id=str(row["protein_id"]),
                layer_index=int(row["layer_index"]),
                distance=float(row["distance"]),
            )
        )
    return grouped


def _resolve_backend(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    embedding_type_id: int,
    layer_index: int,
    metric: DistanceMetric,
    query_count: int,
    device: str | None,
    use_ann: bool,
) -> tuple[str, str | None]:
    resolved = client._resolve_search_backend(
        requested_backend=backend,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
        metric=metric,
        batch_size=query_count,
        ann_requested=use_ann,
        device=device,
    )
    if resolved.backend != backend:
        raise RuntimeError(f"Requested backend {backend!r} resolved to {resolved.backend!r}.")
    return resolved.backend, resolved.device


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _benchmark_query_batches(
    *,
    corpus_size: int,
    backend: SearchBackend,
    samples: Dict[int, List[tuple[list[str], Any]]],
    k: int,
    run_search: _SearchRunner,
    cold_setup_before_search_seconds: float,
    progress_start: int,
    progress_total: int,
) -> Dict[str, Any]:
    first_query_count = min(samples)
    warmup_ids, warmup_vectors = samples[first_query_count][0]
    _print_progress(
        f"[{progress_start}/{progress_total}] corpus_size={corpus_size} backend={backend} "
        f"warmup (query_count={first_query_count}, k={min(k, 5)})",
    )
    warmup_started = time.perf_counter()
    run_search(warmup_ids, warmup_vectors, min(k, 5))
    search_warmup_seconds = time.perf_counter() - warmup_started
    _print_progress(
        f"[{progress_start}/{progress_total}] corpus_size={corpus_size} backend={backend} "
        f"warmup_complete search_elapsed={_format_seconds(search_warmup_seconds)} "
        f"cold_setup_elapsed={_format_seconds(cold_setup_before_search_seconds + search_warmup_seconds)}",
    )

    batch_results: Dict[str, Any] = {}
    completed_runs = progress_start
    for query_count, runs in samples.items():
        _print_progress(
            f"[{completed_runs}/{progress_total}] corpus_size={corpus_size} backend={backend} "
            f"starting query_count={query_count} ({len(runs)} run{'s' if len(runs) != 1 else ''})",
        )
        timings: list[float] = []
        run_details: list[Dict[str, Any]] = []
        for run_index, (query_ids, query_vectors) in enumerate(runs, start=1):
            started = time.perf_counter()
            run_search(query_ids, query_vectors, k)
            elapsed = time.perf_counter() - started
            timings.append(elapsed)
            completed_runs += 1
            throughput_qps = query_count / elapsed
            _print_progress(
                f"[{completed_runs}/{progress_total}] corpus_size={corpus_size} backend={backend} "
                f"query_count={query_count} run={run_index}/{len(runs)} "
                f"elapsed={_format_seconds(elapsed)} throughput={throughput_qps:.2f} q/s",
            )
            run_details.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": elapsed,
                    "query_count": query_count,
                    "throughput_qps": throughput_qps,
                },
            )

        summary = _summarize_timings(timings)
        summary["runs"] = len(timings)
        summary["mean_seconds_per_query"] = summary["mean_seconds"] / query_count
        summary["throughput_qps"] = query_count / summary["mean_seconds"]
        summary["total_seconds_for_batch"] = sum(timings)
        batch_results[str(query_count)] = {"summary": summary, "runs": run_details}

    return {
        "search_warmup_seconds": search_warmup_seconds,
        "cold_setup_seconds": cold_setup_before_search_seconds + search_warmup_seconds,
        "batches": batch_results,
    }


def _benchmark_in_memory_backend(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    candidate_ids: Sequence[str],
    candidate_vectors: Any,
    samples: Dict[int, List[tuple[list[str], Any]]],
    embedding_type_id: int,
    layer_index: int,
    metric: DistanceMetric,
    k: int,
    device: str | None,
    use_ann: bool,
    cold_setup_before_index_seconds: float,
    corpus_size: int,
    progress_start: int,
    progress_total: int,
) -> Dict[str, Any]:
    if use_ann and backend == "torch_gpu":
        raise RuntimeError("torch_gpu does not provide an ANN index; choose faiss_cpu, faiss_gpu, or cuvs_gpu.")
    resolved_backend, resolved_device = _resolve_backend(
        client,
        backend=backend,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
        metric=metric,
        query_count=max(samples),
        device=device,
        use_ann=use_ann,
    )
    if resolved_backend == "pgvector":
        raise RuntimeError("pgvector is not an in-memory backend.")

    _print_progress(
        f"[{progress_start}/{progress_total}] corpus_size={corpus_size} backend={backend} "
        f"building {'ANN' if use_ann else 'exact'} index",
    )
    build_started = time.perf_counter()
    state = build_search_state(
        backend=cast(Any, resolved_backend),
        item_ids=candidate_ids,
        vectors=candidate_vectors,
        metric=metric,
        device=str(resolved_device or "cpu"),
        ann_requested=use_ann,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
    )
    build_seconds = time.perf_counter() - build_started
    _print_progress(
        f"[{progress_start}/{progress_total}] corpus_size={corpus_size} backend={backend} "
        f"index_complete elapsed={_format_seconds(build_seconds)}",
    )

    def run_search(query_ids: Sequence[str], query_vectors: Any, neighbor_count: int) -> None:
        exclusions: Dict[str, Set[str]] = {str(query_id): {str(query_id)} for query_id in query_ids}
        search_state(
            state,
            query_ids=query_ids,
            query_vectors=query_vectors,
            k=neighbor_count,
            per_query_excluded=exclusions,
        )

    batch_benchmark = _benchmark_query_batches(
        corpus_size=corpus_size,
        backend=backend,
        samples=samples,
        k=k,
        run_search=run_search,
        cold_setup_before_search_seconds=cold_setup_before_index_seconds + build_seconds,
        progress_start=progress_start,
        progress_total=progress_total,
    )
    return {
        "backend": backend,
        "resolved_backend": resolved_backend,
        "device": resolved_device,
        "index_build_seconds": build_seconds,
        **batch_benchmark,
    }


def _benchmark_pgvector_backend(
    client: BioDataClient,
    *,
    samples: Dict[int, List[tuple[list[str], Any]]],
    metric: DistanceMetric,
    k: int,
    corpus_size: int,
    progress_start: int,
    progress_total: int,
    use_ann: bool,
    ann_ef_search: int,
    ann_candidate_pool: int | None,
    cold_setup_before_search_seconds: float,
) -> Dict[str, Any]:
    def run_search(query_ids: Sequence[str], query_vectors: Any, neighbor_count: int) -> None:
        _search_pgvector(
            client,
            query_ids=query_ids,
            query_vectors=query_vectors,
            metric=metric,
            k=neighbor_count,
            use_ann=use_ann,
            ann_ef_search=ann_ef_search,
            ann_candidate_pool=ann_candidate_pool,
        )

    batch_benchmark = _benchmark_query_batches(
        corpus_size=corpus_size,
        backend="pgvector",
        samples=samples,
        k=k,
        run_search=run_search,
        cold_setup_before_search_seconds=cold_setup_before_search_seconds,
        progress_start=progress_start,
        progress_total=progress_total,
    )
    return {
        "backend": "pgvector",
        "resolved_backend": "pgvector",
        "device": None,
        "index_build_seconds": 0.0,
        **batch_benchmark,
    }


def _print_report(results: Sequence[Dict[str, Any]]) -> None:
    print(
        "corpus_size\tbackend\tquery_count\truns\tmean\tstdev\tper_query\tthroughput\t"
        "warmup\tbatch_total\tend_to_end\tstatus",
    )
    for result in results:
        if result["status"] != "ok":
            print(
                f"{result['corpus_size']}\t{result['backend']}\t-\t-\t-\t-\t-\t-\t-\t-\t"
                f"{result['status']}: {result['error']}",
            )
            continue
        benchmark = result["benchmark"]
        warmup_text = _format_seconds(float(benchmark["cold_setup_seconds"]))
        for query_count, batch_result in benchmark["batches"].items():
            summary = batch_result["summary"]
            print(
                "\t".join(
                    [
                        str(result["corpus_size"]),
                        str(result["backend"]),
                        str(query_count),
                        str(summary["runs"]),
                        _format_seconds(float(summary["mean_seconds"])),
                        _format_seconds(float(summary["stdev_seconds"])),
                        _format_seconds(float(summary["mean_seconds_per_query"])),
                        f"{float(summary['throughput_qps']):.2f} q/s",
                        warmup_text,
                        _format_seconds(float(summary["total_seconds_for_batch"])),
                        _format_seconds(float(summary["end_to_end_seconds"])),
                        "ok",
                    ],
                ),
            )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    metric = cast(DistanceMetric, args.metric)
    results: list[Dict[str, Any]] = []
    dsn = _resolve_dsn(args.dsn)

    with BioDataClient(dsn=dsn) as client:
        query_ids, query_vectors = _load_query_embedding_pool(
            client,
            embedding_type_id=args.embedding_type_id,
            layer_index=args.layer_index,
            query_pool_size=args.query_pool_size,
            seed=args.seed,
        )
        samples = _build_query_samples(
            query_ids,
            query_vectors,
            query_counts=args.query_counts,
            repeats=args.repeats,
            seed=args.seed,
        )
        runs_per_backend = sum(len(runs) for runs in samples.values())
        total_runs = runs_per_backend * len(args.backends) * len(args.corpus_sizes)
        _print_progress(
            f"Benchmarking {len(args.corpus_sizes)} corpus size(s), {len(args.backends)} backend(s), and "
            f"{len(args.query_counts)} query batch size(s) over {total_runs} timed run"
            f"{'s' if total_runs != 1 else ''} "
            f"(embedding_type_id={args.embedding_type_id}, layer_index={args.layer_index}, "
            f"metric={args.metric}, k={args.k})",
        )
        completed_runs = 0
        for corpus_size in args.corpus_sizes:
            _print_progress(
                f"[{completed_runs}/{total_runs}] corpus_size={corpus_size} preparing candidate corpus",
            )
            materialize_seconds = _materialize_candidate_corpus(
                client,
                embedding_type_id=args.embedding_type_id,
                layer_index=args.layer_index,
                corpus_size=corpus_size,
                seed=args.seed,
            )
            _print_progress(
                f"[{completed_runs}/{total_runs}] corpus_size={corpus_size} candidate_corpus_ready "
                f"elapsed={_format_seconds(materialize_seconds)}",
            )
            candidate_ids: list[str] | None = None
            candidate_vectors: Any = None
            candidate_load_seconds: float | None = None
            temporary_ann_index_seconds = 0.0

            if args.use_ann and "pgvector" in args.backends:
                _print_progress(
                    f"[{completed_runs}/{total_runs}] corpus_size={corpus_size} backend=pgvector "
                    "building temporary HNSW index",
                )
                temporary_ann_index_seconds = _create_temporary_ann_index(
                    client,
                    embedding_dimension=int(query_vectors.shape[1]),
                    metric=metric,
                )
                _print_progress(
                    f"[{completed_runs}/{total_runs}] corpus_size={corpus_size} backend=pgvector "
                    f"temporary_hnsw_ready elapsed={_format_seconds(temporary_ann_index_seconds)}",
                )

            if any(backend != "pgvector" for backend in args.backends):
                _print_progress(
                    f"[{completed_runs}/{total_runs}] corpus_size={corpus_size} loading candidate vectors for in-memory backends",
                )
                candidate_ids, candidate_vectors, candidate_load_seconds = _load_candidate_vectors(client)
                _print_progress(
                    f"[{completed_runs}/{total_runs}] corpus_size={corpus_size} candidate_vectors_ready "
                    f"elapsed={_format_seconds(candidate_load_seconds)}",
                )

            for backend_value in args.backends:
                backend = cast(SearchBackend, backend_value)
                try:
                    if backend == "pgvector":
                        benchmark = _benchmark_pgvector_backend(
                            client,
                            samples=samples,
                            metric=metric,
                            k=args.k,
                            corpus_size=corpus_size,
                            progress_start=completed_runs,
                            progress_total=total_runs,
                            use_ann=args.use_ann,
                            ann_ef_search=args.ann_ef_search,
                            ann_candidate_pool=args.ann_candidate_pool,
                            cold_setup_before_search_seconds=materialize_seconds + temporary_ann_index_seconds,
                        )
                    else:
                        if candidate_ids is None or candidate_vectors is None or candidate_load_seconds is None:
                            raise RuntimeError("Candidate vectors were not loaded for the in-memory backend.")
                        benchmark = _benchmark_in_memory_backend(
                            client,
                            backend=backend,
                            candidate_ids=candidate_ids,
                            candidate_vectors=candidate_vectors,
                            samples=samples,
                            embedding_type_id=args.embedding_type_id,
                            layer_index=args.layer_index,
                            metric=metric,
                            k=args.k,
                            device=args.device,
                            use_ann=args.use_ann,
                            cold_setup_before_index_seconds=materialize_seconds + float(candidate_load_seconds),
                            corpus_size=corpus_size,
                            progress_start=completed_runs,
                            progress_total=total_runs,
                        )
                    cold_setup_seconds = float(benchmark["cold_setup_seconds"])
                    for batch_result in benchmark["batches"].values():
                        summary = batch_result["summary"]
                        summary["end_to_end_seconds"] = cold_setup_seconds + summary["total_seconds_for_batch"]
                    results.append(
                        {
                            "corpus_size": corpus_size,
                            "backend": backend,
                            "status": "ok",
                            "candidate_materialize_seconds": materialize_seconds,
                            "candidate_load_seconds": candidate_load_seconds if backend != "pgvector" else None,
                            "temporary_ann_index_seconds": temporary_ann_index_seconds if backend == "pgvector" else 0.0,
                            "benchmark": benchmark,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "corpus_size": corpus_size,
                            "backend": backend,
                            "status": "unavailable",
                            "error": str(exc),
                            "candidate_materialize_seconds": materialize_seconds,
                            "candidate_load_seconds": candidate_load_seconds if backend != "pgvector" else None,
                            "temporary_ann_index_seconds": temporary_ann_index_seconds if backend == "pgvector" else 0.0,
                        }
                    )
                completed_runs += runs_per_backend

    payload = {
        "corpus_sizes": list(args.corpus_sizes),
        "query_counts": list(args.query_counts),
        "query_pool_size": args.query_pool_size,
        "repeats": args.repeats,
        "backends": list(args.backends),
        "embedding_type_id": args.embedding_type_id,
        "layer_index": args.layer_index,
        "metric": args.metric,
        "k": args.k,
        "seed": args.seed,
        "device": args.device,
        "use_ann": bool(args.use_ann),
        "ann_ef_search": int(args.ann_ef_search),
        "ann_candidate_pool": args.ann_candidate_pool,
        "results": results,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_report(results)
    print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
