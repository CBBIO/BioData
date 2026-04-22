#!/usr/bin/env python3
"""Benchmark neighbor search backends over repeated subsamples of a protein pool."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.BioData import BioDataClient
from CBBIO.types import SearchBackend


DEFAULT_BATCH_SIZES = [1, 10, 100, 1_000, 10_000]
DEFAULT_BACKENDS: list[SearchBackend] = ["pgvector", "faiss_cpu", "torch_gpu", "faiss_gpu", "cuvs_gpu"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pgvector, faiss_cpu, torch_gpu, faiss_gpu, and cuvs_gpu neighbor lookup.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "random_protein_ids_10000.txt",
        help="Protein ID pool file. Default: random_protein_ids_10000.txt",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark. Default: 1 10 100 1000 10000",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Repeats for batch sizes smaller than the full pool. Default: 5.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        choices=["pgvector", "faiss_cpu", "torch_gpu", "faiss_gpu", "cuvs_gpu", "auto", "gpu"],
        help="Backends to benchmark.",
    )
    parser.add_argument(
        "--embedding-type-id",
        type=int,
        default=1,
        help="Embedding type ID to benchmark. Default: 1.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Layer index to benchmark. Default: 0.",
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
        "--seed",
        type=int,
        default=7,
        help="Sampling seed for repeated subsamples. Default: 7.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional explicit device for GPU backends.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "benchmark_neighbor_search.json",
        help="JSON output path. Default: benchmark_neighbor_search.json",
    )
    return parser.parse_args()


def _load_protein_ids(path: Path) -> list[str]:
    protein_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not protein_ids:
        raise SystemExit(f"No protein IDs found in {path}")
    return protein_ids


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
    }
    if len(timings) > 1:
        summary["stdev_seconds"] = statistics.stdev(timings)
    else:
        summary["stdev_seconds"] = 0.0
    return summary


def _build_samples(
    protein_ids: Sequence[str],
    batch_sizes: Sequence[int],
    repeats: int,
    seed: int,
) -> Dict[int, List[list[str]]]:
    rng = random.Random(seed)
    pool = list(protein_ids)
    samples: Dict[int, List[list[str]]] = {}
    for batch_size in batch_sizes:
        if batch_size < 1:
            raise SystemExit("Batch sizes must be >= 1")
        if batch_size > len(pool):
            raise SystemExit(
                f"Batch size {batch_size} exceeds ID pool size {len(pool)} from the input file",
            )
        run_count = 1 if batch_size >= len(pool) else repeats
        samples[batch_size] = [rng.sample(pool, batch_size) for _ in range(run_count)]
    return samples


def _run_backend_once(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    protein_ids: Sequence[str],
    embedding_type_id: int,
    layer_index: int,
    metric: str,
    k: int,
    device: str | None,
) -> Dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        result = client.find_nearest_neighbors_for_proteins(
            protein_ids,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            k=k,
            metric=metric,  # type: ignore[arg-type]
            backend=backend,
            device=device,
        )
        elapsed = time.perf_counter() - started

    diagnostics = client.last_search_diagnostics
    return {
        "elapsed_seconds": elapsed,
        "query_count": len(result),
        "neighbor_rows": sum(len(neighbors) for neighbors in result.values()),
        "diagnostics": diagnostics,
        "warnings": [str(item.message) for item in caught],
    }


def _benchmark_backend(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    samples: Dict[int, List[list[str]]],
    embedding_type_id: int,
    layer_index: int,
    metric: str,
    k: int,
    device: str | None,
    progress_start: int,
    progress_total: int,
) -> Dict[str, Any]:
    backend_started = time.perf_counter()
    first_batch_size = min(samples)
    warmup_ids = samples[first_batch_size][0]
    _print_progress(
        f"[{progress_start}/{progress_total}] backend={backend} warmup "
        f"(batch_size={first_batch_size}, k={min(k, 5)})"
    )
    try:
        warmup_result = _run_backend_once(
            client,
            backend=backend,
            protein_ids=warmup_ids,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            k=min(k, 5),
            device=device,
        )
    except Exception as exc:
        return {
            "backend": backend,
            "status": "unavailable",
            "error": str(exc),
            "warmup_seconds": None,
            "total_seconds_including_setup": time.perf_counter() - backend_started,
            "batches": {},
        }
    warmup_seconds = float(warmup_result["elapsed_seconds"])
    _print_progress(
        f"[{progress_start}/{progress_total}] backend={backend} warmup_complete "
        f"elapsed={_format_seconds(warmup_seconds)}"
    )

    batch_results: Dict[str, Any] = {}
    completed_runs = progress_start
    for batch_size, runs in samples.items():
        _print_progress(
            f"[{completed_runs}/{progress_total}] backend={backend} "
            f"starting batch_size={batch_size} ({len(runs)} run{'s' if len(runs) != 1 else ''})"
        )
        timings: list[float] = []
        run_details: list[Dict[str, Any]] = []
        for run_index, protein_ids in enumerate(runs, start=1):
            try:
                run_result = _run_backend_once(
                    client,
                    backend=backend,
                    protein_ids=protein_ids,
                    embedding_type_id=embedding_type_id,
                    layer_index=layer_index,
                    metric=metric,
                    k=k,
                    device=device,
                )
            except Exception as exc:
                return {
                    "backend": backend,
                    "status": "failed",
                    "error": f"batch_size={batch_size}, run={run_index}: {exc}",
                    "warmup_seconds": warmup_seconds,
                    "total_seconds_including_setup": time.perf_counter() - backend_started,
                    "batches": batch_results,
                }
            timings.append(float(run_result["elapsed_seconds"]))
            completed_runs += 1
            throughput_qps = batch_size / float(run_result["elapsed_seconds"])
            warning_suffix = ""
            if run_result["warnings"]:
                warning_suffix = f", warnings={len(run_result['warnings'])}"
            _print_progress(
                f"[{completed_runs}/{progress_total}] backend={backend} "
                f"batch_size={batch_size} run={run_index}/{len(runs)} "
                f"elapsed={_format_seconds(float(run_result['elapsed_seconds']))} "
                f"throughput={throughput_qps:.2f} q/s{warning_suffix}"
            )
            run_details.append(
                {
                    "run_index": run_index,
                    "elapsed_seconds": run_result["elapsed_seconds"],
                    "query_count": run_result["query_count"],
                    "neighbor_rows": run_result["neighbor_rows"],
                    "warnings": run_result["warnings"],
                    "diagnostics": run_result["diagnostics"],
                }
            )

        summary = _summarize_timings(timings)
        summary["mean_seconds_per_query"] = summary["mean_seconds"] / batch_size
        summary["throughput_qps"] = batch_size / summary["mean_seconds"]
        summary["total_seconds_for_batch"] = sum(timings)
        summary["end_to_end_seconds"] = warmup_seconds + summary["total_seconds_for_batch"]
        batch_results[str(batch_size)] = {
            "summary": summary,
            "runs": run_details,
        }

    return {
        "backend": backend,
        "status": "ok",
        "warmup_seconds": warmup_seconds,
        "timed_query_seconds": sum(
            batch_result["summary"]["mean_seconds"] * batch_result["summary"]["runs"]
            for batch_result in batch_results.values()
        ),
        "total_seconds_including_setup": time.perf_counter() - backend_started,
        "batches": batch_results,
    }


def _print_report(results: Sequence[Dict[str, Any]]) -> None:
    print(
        "backend\tbatch_size\truns\tmean\tstdev\tper_query\tthroughput\twarmup\tbatch_total\tend_to_end\tstatus",
    )
    for result in results:
        backend = str(result["backend"])
        status = str(result["status"])
        warmup_text = (
            _format_seconds(float(result["warmup_seconds"]))
            if result.get("warmup_seconds") is not None
            else "-"
        )
        total_text = (
            _format_seconds(float(result["total_seconds_including_setup"]))
            if result.get("total_seconds_including_setup") is not None
            else "-"
        )
        if status != "ok":
            print(f"{backend}\t-\t-\t-\t-\t-\t-\t{warmup_text}\t-\t{total_text}\t{status}: {result['error']}")
            continue
        for batch_size, batch_result in result["batches"].items():
            summary = batch_result["summary"]
            batch_total_text = _format_seconds(summary["total_seconds_for_batch"])
            end_to_end_text = _format_seconds(summary["end_to_end_seconds"])
            print(
                "\t".join(
                    [
                        backend,
                        str(batch_size),
                        str(summary["runs"]),
                        _format_seconds(summary["mean_seconds"]),
                        _format_seconds(summary["stdev_seconds"]),
                        _format_seconds(summary["mean_seconds_per_query"]),
                        f"{summary['throughput_qps']:.2f} q/s",
                        warmup_text,
                        batch_total_text,
                        end_to_end_text,
                        status,
                    ]
                )
            )


def main() -> None:
    args = _parse_args()
    protein_ids = _load_protein_ids(args.ids_file)
    samples = _build_samples(protein_ids, args.batch_sizes, args.repeats, args.seed)
    runs_per_backend = sum(len(runs) for runs in samples.values())
    total_runs = runs_per_backend * len(args.backends)
    _print_progress(
        f"Benchmarking {len(args.backends)} backend(s) over {total_runs} timed run"
        f"{'s' if total_runs != 1 else ''} "
        f"(embedding_type_id={args.embedding_type_id}, layer_index={args.layer_index}, metric={args.metric}, k={args.k})"
    )

    results: list[Dict[str, Any]] = []
    completed_runs = 0
    with BioDataClient() as client:
        for backend in args.backends:
            results.append(
                _benchmark_backend(
                    client,
                    backend=backend,
                    samples=samples,
                    embedding_type_id=args.embedding_type_id,
                    layer_index=args.layer_index,
                    metric=args.metric,
                    k=args.k,
                    device=args.device,
                    progress_start=completed_runs,
                    progress_total=total_runs,
                )
            )
            completed_runs += runs_per_backend

    payload = {
        "ids_file": str(args.ids_file),
        "batch_sizes": list(args.batch_sizes),
        "repeats": int(args.repeats),
        "backends": list(args.backends),
        "embedding_type_id": int(args.embedding_type_id),
        "layer_index": int(args.layer_index),
        "metric": args.metric,
        "k": int(args.k),
        "seed": int(args.seed),
        "device": args.device,
        "results": results,
    }
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_report(results)
    print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
