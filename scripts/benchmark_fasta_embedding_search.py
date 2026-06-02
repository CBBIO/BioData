#!/usr/bin/env python3
"""Benchmark an end-to-end FASTA -> embeddings -> neighbor-search pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
import warnings
from numbers import Real
from pathlib import Path
from typing import Any, Dict, List, Sequence, cast

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.BioData import BioDataClient
from CBBIO.embeddings import GenerationInput, Generator, load_fasta_inputs
from CBBIO.search.types import _ResolvedBackend
from CBBIO.types import DistanceMetric, SearchBackend


DEFAULT_BATCH_SIZES = [1, 10, 100, 1_000]
DEFAULT_BACKENDS: list[SearchBackend] = ["auto", "pgvector", "faiss_cpu", "torch_gpu", "cuvs_gpu"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FASTA loading, embedding generation, pooling, and neighbor search.",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Input FASTA file containing the query proteins to benchmark.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark. Default: 1 10 100 1000",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeated sampled runs per batch size. Default: 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Sampling seed for repeated FASTA subsamples. Default: 7.",
    )
    parser.add_argument(
        "--generator-class",
        type=str,
        required=True,
        help="Embedding generator family (for example: protT5, esm2, esmc).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional explicit model identifier. Uses the generator default when omitted.",
    )
    parser.add_argument(
        "--generator-device",
        type=str,
        default="cpu",
        help="Device used for embedding generation. Default: cpu.",
    )
    parser.add_argument(
        "--embedding-type-id",
        type=int,
        required=True,
        help="Target DB embedding type used for neighbor search.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Layer index for both embedding generation and neighbor search. Default: 0.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "first"],
        default="mean",
        help="How to reduce per-residue embeddings to one query vector. Default: mean.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=1,
        help="Optional micro-batch size for embedding generation. Default: 1.",
    )
    parser.add_argument(
        "--embedding-sort-by-length",
        action="store_true",
        help="Sort each sampled embedding batch by sequence length before chunking.",
    )
    parser.add_argument(
        "--embedding-max-tokens-per-batch",
        type=int,
        default=None,
        help="Optional padded-token budget per embedding microbatch. Overrides --embedding-batch-size when set.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["auto", "gpu", "pgvector", "faiss_cpu", "torch_gpu", "faiss_gpu", "cuvs_gpu"],
        help="Neighbor-search backends to benchmark.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cosine",
        choices=["l2", "cosine", "inner_product"],
        help="Distance metric. Default: cosine.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of neighbors per query. Default: 10.",
    )
    parser.add_argument(
        "--use-ann",
        action="store_true",
        help="Enable ANN when supported by the backend/path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit search device for GPU backends.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "benchmark_fasta_embedding_search.json",
        help="JSON output path. Default: benchmark_fasta_embedding_search.json",
    )
    return parser.parse_args()


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _summarize_timings(timings: Sequence[float]) -> Dict[str, float]:
    summary = {
        "runs": len(timings),
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "total_seconds": sum(timings),
    }
    if len(timings) > 1:
        summary["stdev_seconds"] = statistics.stdev(timings)
    else:
        summary["stdev_seconds"] = 0.0
    return summary


def _is_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _pool_embedding(embedding: Sequence[Any], *, pooling: str) -> List[float]:
    values = list(embedding)
    if not values:
        raise ValueError("Cannot pool an empty embedding.")

    if _is_scalar(values[0]):
        return [float(item) for item in values]

    rows = [list(row) for row in values]
    if not rows or not rows[0]:
        raise ValueError("Cannot pool an empty per-residue embedding.")
    dims = len(rows[0])
    if any(len(row) != dims for row in rows):
        raise ValueError("Embedding rows do not share the same dimension.")

    if pooling == "first":
        return [float(item) for item in rows[0]]
    if pooling != "mean":
        raise ValueError(f"Unsupported pooling mode: {pooling!r}")

    pooled = [0.0] * dims
    for row in rows:
        for index, value in enumerate(row):
            pooled[index] += float(value)
    count = float(len(rows))
    return [value / count for value in pooled]


def _build_samples(
    inputs: Sequence[GenerationInput],
    batch_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[int, List[List[GenerationInput]]]:
    rng = random.Random(seed)
    pool = list(inputs)
    samples: Dict[int, List[List[GenerationInput]]] = {}
    for batch_size in batch_sizes:
        if batch_size < 1:
            raise SystemExit("Batch sizes must be >= 1")
        if batch_size > len(pool):
            raise SystemExit(
                f"Batch size {batch_size} exceeds FASTA pool size {len(pool)}",
            )
        run_count = 1 if batch_size >= len(pool) else repeats
        samples[batch_size] = [rng.sample(pool, batch_size) for _ in range(run_count)]
    return samples


def _chunk_records(records: Sequence[GenerationInput], chunk_size: int | None) -> List[List[GenerationInput]]:
    if chunk_size is None:
        return [list(records)]
    effective = int(chunk_size)
    if effective < 1:
        raise SystemExit("--embedding-batch-size must be >= 1")
    return [list(records[start : start + effective]) for start in range(0, len(records), effective)]


def _sort_records_by_length(records: Sequence[GenerationInput]) -> List[GenerationInput]:
    return sorted(records, key=lambda record: (len(str(record.sequence)), str(record.id)))


def _chunk_records_by_token_budget(
    records: Sequence[GenerationInput],
    max_tokens_per_batch: int,
) -> List[List[GenerationInput]]:
    effective = int(max_tokens_per_batch)
    if effective < 1:
        raise SystemExit("--embedding-max-tokens-per-batch must be >= 1")

    chunks: List[List[GenerationInput]] = []
    current: List[GenerationInput] = []
    current_max_len = 0
    for record in records:
        seq_len = len(str(record.sequence))
        next_max_len = max(current_max_len, seq_len)
        projected_tokens = (len(current) + 1) * next_max_len
        if current and projected_tokens > effective:
            chunks.append(list(current))
            current = [record]
            current_max_len = seq_len
            continue
        current.append(record)
        current_max_len = next_max_len
    if current:
        chunks.append(list(current))
    return chunks


def _schedule_embedding_chunks(
    records: Sequence[GenerationInput],
    *,
    chunk_size: int | None,
    max_tokens_per_batch: int | None,
    sort_by_length: bool,
) -> List[List[GenerationInput]]:
    scheduled = list(records)
    if sort_by_length:
        scheduled = _sort_records_by_length(scheduled)
    if max_tokens_per_batch is not None:
        return _chunk_records_by_token_budget(scheduled, max_tokens_per_batch)
    return _chunk_records(scheduled, chunk_size)


def _maybe_release_cuda_cache(device: str | None) -> None:
    if device is None or not str(device).startswith("cuda"):
        return


def _resolve_hf_layer_index(layer_index: int, *, total_layers: int) -> int:
    resolved = total_layers + layer_index if layer_index < 0 else layer_index
    if resolved < 0 or resolved >= total_layers:
        raise RuntimeError(f"Requested layer index {layer_index} out of range for total layers={total_layers}.")
    return resolved


def _generate_embeddings_for_batch_prott5(
    generator: Any,
    inputs: Sequence[GenerationInput],
    *,
    layer_index: int,
    pooling: str,
    embedding_batch_size: int | None,
    embedding_max_tokens_per_batch: int | None,
    embedding_sort_by_length: bool,
    generator_device: str | None,
) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for ProtT5 batch benchmarking.") from exc

    tokenizer = getattr(getattr(generator, "tokenizer", None), "tokenizer", None)
    model = getattr(getattr(generator, "model", None), "model", None)
    preprocessor = getattr(getattr(generator, "preprocessor", None), "preprocess", None)
    if tokenizer is None or model is None or not callable(preprocessor):
        raise RuntimeError("ProtT5 benchmark batching could not access tokenizer/model/preprocessor.")

    chunks = _schedule_embedding_chunks(
        inputs,
        chunk_size=embedding_batch_size,
        max_tokens_per_batch=embedding_max_tokens_per_batch,
        sort_by_length=embedding_sort_by_length,
    )
    generate_started = time.perf_counter()
    pooled_ids: List[str] = []
    pooled_vectors: List[List[float]] = []

    model.eval()
    with torch.inference_mode():
        for chunk in chunks:
            prepared = [preprocessor(record.sequence) for record in chunk]
            encoded = tokenizer(
                prepared,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(generator_device or "cpu")
            attention_mask = encoded["attention_mask"].to(generator_device or "cpu")
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = getattr(model_output, "hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("ProtT5 model output did not include hidden_states.")
            hf_layer_index = _resolve_hf_layer_index(layer_index, total_layers=len(hidden_states))
            selected = hidden_states[hf_layer_index]
            for row_index, record in enumerate(chunk):
                residue_len = max(int(attention_mask[row_index].sum().item()) - 1, 1)
                matrix = selected[row_index, :residue_len].detach().cpu().tolist()
                pooled_ids.append(str(record.id))
                pooled_vectors.append(_pool_embedding(matrix, pooling=pooling))
            del model_output
            del input_ids
            del attention_mask
            _maybe_release_cuda_cache(generator_device)

    generate_seconds = time.perf_counter() - generate_started
    if not pooled_vectors:
        raise RuntimeError("Embedding generation produced no pooled query vectors.")
    return {
        "query_ids": pooled_ids,
        "query_vectors": pooled_vectors,
        "record_count": len(pooled_vectors),
        "embedding_dim": len(pooled_vectors[0]),
        "generate_seconds": generate_seconds,
        "pooling_seconds": 0.0,
        "total_seconds": generate_seconds,
        "embedding_microbatch_count": len(chunks),
        "embedding_microbatch_size": max((len(chunk) for chunk in chunks), default=0),
        "errors": [],
        "skipped": [],
    }
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return
    if not torch.cuda.is_available():
        return
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        return


def _resolve_backend(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    embedding_type_id: int,
    layer_index: int,
    metric: DistanceMetric,
    batch_size: int,
    use_ann: bool,
    device: str | None,
) -> _ResolvedBackend:
    resolved = client._resolve_search_backend(
        requested_backend=backend,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
        metric=metric,
        batch_size=batch_size,
        ann_requested=use_ann,
        device=device,
    )
    return client._search._apply_auto_gpu_heuristics(
        resolved,
        requested_backend=backend,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
        metric=metric,
        batch_size=batch_size,
        device=device,
    )


def _search_embeddings_once(
    client: BioDataClient,
    *,
    resolved: _ResolvedBackend,
    query_ids: Sequence[str],
    query_vectors: Sequence[Sequence[float]],
    embedding_type_id: int,
    layer_index: int,
    metric: DistanceMetric,
    k: int,
    use_ann: bool,
) -> Dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        if resolved.backend == "pgvector":
            grouped = {
                query_id: client.find_nearest_neighbors(
                    vector,
                    embedding_type_id=embedding_type_id,
                    layer_index=layer_index,
                    k=k,
                    metric=metric,
                    use_ann=use_ann,
                    backend="pgvector",
                )
                for query_id, vector in zip(query_ids, query_vectors)
            }
        else:
            chunk_size = max(1, int(resolved.chunk_size or len(query_ids) or 1))
            grouped: Dict[str, Any] = {}
            for chunk_start in range(0, len(query_ids), chunk_size):
                chunk_ids = list(query_ids[chunk_start : chunk_start + chunk_size])
                chunk_vectors = list(query_vectors[chunk_start : chunk_start + chunk_size])
                if resolved.backend == "faiss_gpu":
                    partial = client._find_nearest_neighbors_for_queries_faiss(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=k,
                        metric=metric,
                        include_query=False,
                        device=resolved.device,
                    )
                elif resolved.backend == "faiss_cpu":
                    partial = client._find_nearest_neighbors_for_queries_faiss_cpu(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=k,
                        metric=metric,
                        include_query=False,
                        use_ann=resolved.ann_used,
                    )
                elif resolved.backend == "cuvs_gpu":
                    partial = client._find_nearest_neighbors_for_queries_cuvs(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=k,
                        metric=metric,
                        include_query=False,
                        device=resolved.device,
                        use_ann=resolved.ann_used,
                    )
                else:
                    partial = client._find_nearest_neighbors_for_queries_torch(
                        chunk_ids,
                        chunk_vectors,
                        embedding_type_id=embedding_type_id,
                        layer_index=layer_index,
                        k=k,
                        metric=metric,
                        include_query=False,
                        device=resolved.device,
                    )
                grouped.update(partial)
        elapsed = time.perf_counter() - started

    return {
        "elapsed_seconds": elapsed,
        "query_count": len(grouped),
        "neighbor_rows": sum(len(neighbors) for neighbors in grouped.values()),
        "warnings": [str(item.message) for item in caught],
        "resolved_backend": resolved.backend,
        "reason": resolved.reason,
        "chunk_size": resolved.chunk_size,
        "estimated_bytes": resolved.estimated_bytes,
        "free_bytes": resolved.free_bytes,
    }


def _generate_embeddings_for_batch(
    generator: Any,
    inputs: Sequence[GenerationInput],
    *,
    generator_class: str,
    layer_index: int,
    pooling: str,
    embedding_batch_size: int | None,
    embedding_max_tokens_per_batch: int | None,
    embedding_sort_by_length: bool,
    generator_device: str | None,
) -> Dict[str, Any]:
    if str(generator_class).strip().lower() == "prott5":
        return _generate_embeddings_for_batch_prott5(
            generator,
            inputs,
            layer_index=layer_index,
            pooling=pooling,
            embedding_batch_size=embedding_batch_size,
            embedding_max_tokens_per_batch=embedding_max_tokens_per_batch,
            embedding_sort_by_length=embedding_sort_by_length,
            generator_device=generator_device,
        )

    generate_started = time.perf_counter()
    chunks = _schedule_embedding_chunks(
        inputs,
        chunk_size=embedding_batch_size,
        max_tokens_per_batch=embedding_max_tokens_per_batch,
        sort_by_length=embedding_sort_by_length,
    )
    generated_errors: List[Any] = []
    generated_skipped: List[Any] = []
    pooled_ids: List[str] = []
    pooled_vectors: List[List[float]] = []
    for chunk in chunks:
        generated = generator.generate(chunk, layer_index=layer_index, fail_fast=True)
        generated_errors.extend(generated.errors)
        generated_skipped.extend(generated.skipped)
        for record in generated.records:
            pooled_ids.append(str(record.id))
            pooled_vectors.append(_pool_embedding(record.embedding, pooling=pooling))
        del generated
        _maybe_release_cuda_cache(generator_device)
    generate_seconds = time.perf_counter() - generate_started

    pooling_seconds = 0.0

    if not pooled_vectors:
        raise RuntimeError("Embedding generation produced no pooled query vectors.")

    return {
        "query_ids": pooled_ids,
        "query_vectors": pooled_vectors,
        "record_count": len(pooled_vectors),
        "embedding_dim": len(pooled_vectors[0]),
        "generate_seconds": generate_seconds,
        "pooling_seconds": pooling_seconds,
        "total_seconds": generate_seconds + pooling_seconds,
        "embedding_microbatch_count": len(chunks),
        "embedding_microbatch_size": (
            max(len(chunk) for chunk in chunks) if chunks else 0
        ),
        "errors": generated_errors,
        "skipped": generated_skipped,
    }


def _benchmark_batch_backend(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    batch_size: int,
    embedding_runs: Sequence[Dict[str, Any]],
    embedding_type_id: int,
    layer_index: int,
    metric: DistanceMetric,
    k: int,
    use_ann: bool,
    device: str | None,
) -> Dict[str, Any]:
    resolved = _resolve_backend(
        client,
        backend=backend,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
        metric=metric,
        batch_size=batch_size,
        use_ann=use_ann,
        device=device,
    )
    warmup_result: Dict[str, Any] | None = None
    try:
        warmup_result = _search_embeddings_once(
            client,
            resolved=resolved,
            query_ids=embedding_runs[0]["query_ids"],
            query_vectors=embedding_runs[0]["query_vectors"],
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            k=min(k, 5),
            use_ann=use_ann,
        )
    except Exception as exc:
        return {
            "backend": backend,
            "resolved_backend": resolved.backend,
            "reason": resolved.reason,
            "status": "unavailable",
            "error": str(exc),
            "warmup_seconds": None,
        }

    timings: List[float] = []
    run_details: List[Dict[str, Any]] = []
    for run_index, embedding_run in enumerate(embedding_runs, start=1):
        run_result = _search_embeddings_once(
            client,
            resolved=resolved,
            query_ids=embedding_run["query_ids"],
            query_vectors=embedding_run["query_vectors"],
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            k=k,
            use_ann=use_ann,
        )
        timings.append(float(run_result["elapsed_seconds"]))
        run_details.append(
            {
                "run_index": run_index,
                "elapsed_seconds": run_result["elapsed_seconds"],
                "query_count": run_result["query_count"],
                "neighbor_rows": run_result["neighbor_rows"],
                "warnings": run_result["warnings"],
            }
        )

    summary = _summarize_timings(timings)
    summary["mean_seconds_per_query"] = summary["mean_seconds"] / batch_size
    summary["throughput_qps"] = batch_size / summary["mean_seconds"]
    summary["warmup_seconds"] = float(warmup_result["elapsed_seconds"])
    summary["cold_total_seconds"] = summary["warmup_seconds"] + summary["mean_seconds"]
    return {
        "backend": backend,
        "resolved_backend": resolved.backend,
        "reason": resolved.reason,
        "chunk_size": resolved.chunk_size,
        "estimated_bytes": resolved.estimated_bytes,
        "free_bytes": resolved.free_bytes,
        "status": "ok",
        "warmup_seconds": summary["warmup_seconds"],
        "summary": summary,
        "runs": run_details,
    }


def _print_report(payload: Dict[str, Any]) -> None:
    print(
        "batch_size\tbackend\tresolved_backend\tembed\tsearch_mean\tsearch_warmup\twarm_pipeline\tcold_pipeline\tembed_share_warm\tembed_share_cold\tstatus",
    )
    for batch_size, batch_payload in payload["batches"].items():
        embedding_summary = batch_payload["embedding"]["summary"]
        embedding_total = float(embedding_summary["mean_total_seconds"])
        for backend_payload in batch_payload["search_backends"]:
            status = str(backend_payload["status"])
            if status != "ok":
                print(
                    "\t".join(
                        [
                            str(batch_size),
                            str(backend_payload["backend"]),
                            str(backend_payload.get("resolved_backend", "-")),
                            _format_seconds(embedding_total),
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                            f"{status}: {backend_payload['error']}",
                        ]
                    )
                )
                continue
            search_summary = backend_payload["summary"]
            warm_pipeline = embedding_total + float(search_summary["mean_seconds"])
            cold_pipeline = embedding_total + float(search_summary["cold_total_seconds"])
            embed_share_warm = 100.0 * embedding_total / warm_pipeline if warm_pipeline else 0.0
            embed_share_cold = 100.0 * embedding_total / cold_pipeline if cold_pipeline else 0.0
            print(
                "\t".join(
                    [
                        str(batch_size),
                        str(backend_payload["backend"]),
                        str(backend_payload["resolved_backend"]),
                        _format_seconds(embedding_total),
                        _format_seconds(float(search_summary["mean_seconds"])),
                        _format_seconds(float(search_summary["warmup_seconds"])),
                        _format_seconds(warm_pipeline),
                        _format_seconds(cold_pipeline),
                        f"{embed_share_warm:.1f}%",
                        f"{embed_share_cold:.1f}%",
                        status,
                    ]
                )
            )


def main() -> None:
    args = _parse_args()
    load_started = time.perf_counter()
    fasta_inputs = load_fasta_inputs(args.fasta)
    fasta_load_seconds = time.perf_counter() - load_started
    if not fasta_inputs:
        raise SystemExit(f"No FASTA records found in {args.fasta}")

    samples = _build_samples(fasta_inputs, args.batch_sizes, args.repeats, args.seed)

    generator_kwargs: Dict[str, Any] = {
        "model_class": args.generator_class,
        "device": args.generator_device,
    }
    if args.model_name:
        generator_kwargs["name"] = args.model_name

    _print_progress(
        f"Loaded {len(fasta_inputs)} FASTA records from {args.fasta} in {_format_seconds(fasta_load_seconds)}"
    )
    generator_started = time.perf_counter()
    generator = Generator(**generator_kwargs)
    generator_init_seconds = time.perf_counter() - generator_started
    _print_progress(
        f"Initialized generator class={args.generator_class} on {args.generator_device} in {_format_seconds(generator_init_seconds)}"
    )

    payload: Dict[str, Any] = {
        "fasta": str(args.fasta),
        "fasta_record_count": len(fasta_inputs),
        "fasta_load_seconds": fasta_load_seconds,
        "generator_class": args.generator_class,
        "model_name": args.model_name or getattr(generator, "model_reference", None),
        "generator_device": args.generator_device,
        "generator_init_seconds": generator_init_seconds,
        "batch_sizes": list(args.batch_sizes),
        "repeats": int(args.repeats),
        "seed": int(args.seed),
        "embedding_type_id": int(args.embedding_type_id),
        "layer_index": int(args.layer_index),
        "pooling": args.pooling,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_sort_by_length": bool(args.embedding_sort_by_length),
        "embedding_max_tokens_per_batch": args.embedding_max_tokens_per_batch,
        "backends": list(args.backends),
        "metric": args.metric,
        "k": int(args.k),
        "use_ann": bool(args.use_ann),
        "device": args.device,
        "batches": {},
    }

    with BioDataClient() as client:
        for batch_size in args.batch_sizes:
            batch_runs = samples[batch_size]
            _print_progress(
                f"Embedding batch_size={batch_size} over {len(batch_runs)} sampled run{'s' if len(batch_runs) != 1 else ''}"
            )
            embedding_runs: List[Dict[str, Any]] = []
            for run_index, batch_inputs in enumerate(batch_runs, start=1):
                run_result = _generate_embeddings_for_batch(
                    generator,
                    batch_inputs,
                    generator_class=args.generator_class,
                    layer_index=args.layer_index,
                    pooling=args.pooling,
                    embedding_batch_size=args.embedding_batch_size,
                    embedding_max_tokens_per_batch=args.embedding_max_tokens_per_batch,
                    embedding_sort_by_length=bool(args.embedding_sort_by_length),
                    generator_device=args.generator_device,
                )
                embedding_runs.append(run_result)
                _print_progress(
                    f"batch_size={batch_size} embed_run={run_index}/{len(batch_runs)} "
                    f"embed={_format_seconds(run_result['generate_seconds'])} "
                    f"pool={_format_seconds(run_result['pooling_seconds'])} "
                    f"microbatches={run_result['embedding_microbatch_count']}"
                )

            generate_timings = [float(item["generate_seconds"]) for item in embedding_runs]
            pool_timings = [float(item["pooling_seconds"]) for item in embedding_runs]
            total_timings = [float(item["total_seconds"]) for item in embedding_runs]
            embedding_summary = {
                "runs": len(embedding_runs),
                "mean_generate_seconds": statistics.fmean(generate_timings),
                "mean_pooling_seconds": statistics.fmean(pool_timings),
                "mean_total_seconds": statistics.fmean(total_timings),
                "stdev_total_seconds": statistics.stdev(total_timings) if len(total_timings) > 1 else 0.0,
                "embedding_dim": int(embedding_runs[0]["embedding_dim"]),
                "embedding_microbatch_size": int(embedding_runs[0]["embedding_microbatch_size"]),
            }

            search_results: List[Dict[str, Any]] = []
            for backend in args.backends:
                _print_progress(f"Searching batch_size={batch_size} backend={backend}")
                backend_result = _benchmark_batch_backend(
                    client,
                    backend=cast(SearchBackend, backend),
                    batch_size=batch_size,
                    embedding_runs=embedding_runs,
                    embedding_type_id=args.embedding_type_id,
                    layer_index=args.layer_index,
                    metric=cast(DistanceMetric, args.metric),
                    k=args.k,
                    use_ann=bool(args.use_ann),
                    device=args.device,
                )
                search_results.append(backend_result)

            payload["batches"][str(batch_size)] = {
                "embedding": {
                    "summary": embedding_summary,
                    "runs": [
                        {
                            "run_index": index + 1,
                            "record_count": item["record_count"],
                            "embedding_dim": item["embedding_dim"],
                            "generate_seconds": item["generate_seconds"],
                            "pooling_seconds": item["pooling_seconds"],
                            "total_seconds": item["total_seconds"],
                            "embedding_microbatch_count": item["embedding_microbatch_count"],
                            "embedding_microbatch_size": item["embedding_microbatch_size"],
                            "errors": item["errors"],
                            "skipped": item["skipped"],
                        }
                        for index, item in enumerate(embedding_runs)
                    ],
                },
                "search_backends": search_results,
            }

    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_report(payload)
    print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
