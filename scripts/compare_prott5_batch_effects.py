#!/usr/bin/env python3
"""Compare true ProtT5 batched inference outputs across embedding batch sizes."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.BioData import BioDataClient
from CBBIO.embeddings import GenerationInput, load_fasta_inputs
from CBBIO.types import DistanceMetric, Neighbor, SearchBackend


DEFAULT_MODEL_NAME = "Rostlab/prot_t5_xl_uniref50"
DEFAULT_BATCH_SIZES = [1, 10, 50]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run true ProtT5 batched inference for the same proteins with different batch sizes, "
            "compare pooled embeddings numerically, and optionally compare top-k neighbor results."
        ),
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file.")
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of proteins sampled from FASTA for the comparison. Default: 100.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_BATCH_SIZES,
        help="True inference batch sizes to compare. Default: 1 10 50.",
    )
    parser.add_argument(
        "--token-budgets",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional token-budgeted batching settings. Each value is an approximate "
            "max padded-token budget per batch after length sorting."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Sampling seed for FASTA records. Default: 7.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Hugging Face ProtT5 checkpoint. Default: {DEFAULT_MODEL_NAME}",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for inference. Default: cuda:0.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="BioData layer index, where 0 means the last hidden layer. Default: 0.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "first"],
        default="mean",
        help="How to reduce per-residue embeddings. Default: mean.",
    )
    parser.add_argument(
        "--sort-by-length",
        action="store_true",
        help="Sort sampled proteins by sequence length before forming inference batches.",
    )
    parser.add_argument(
        "--embedding-type-id",
        type=int,
        default=None,
        help="Optional DB embedding type for neighbor-comparison runs.",
    )
    parser.add_argument(
        "--search-backend",
        choices=["pgvector", "faiss_cpu", "faiss_gpu", "cuvs_gpu", "torch_gpu"],
        default="faiss_cpu",
        help="Backend used for neighbor-comparison runs. Default: faiss_cpu.",
    )
    parser.add_argument(
        "--metric",
        choices=["l2", "cosine", "inner_product"],
        default="cosine",
        help="Neighbor-search metric. Default: cosine.",
    )
    parser.add_argument("--k", type=int, default=10, help="Neighbors per query. Default: 10.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "compare_prott5_batch_effects.json",
        help="JSON output path.",
    )
    return parser.parse_args()


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _sample_inputs(path: Path, *, count: int, seed: int) -> List[GenerationInput]:
    inputs = load_fasta_inputs(path)
    if count < 1:
        raise SystemExit("--count must be >= 1")
    if count > len(inputs):
        raise SystemExit(f"--count {count} exceeds FASTA pool size {len(inputs)}")
    rng = random.Random(seed)
    return rng.sample(list(inputs), count)


def _sort_inputs_by_length(records: Sequence[GenerationInput]) -> List[GenerationInput]:
    return sorted(records, key=lambda record: (len(str(record.sequence)), str(record.id)))


def _preprocess_sequence(sequence: str) -> str:
    replaced = sequence.strip().upper().replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
    return " ".join(list(replaced))


def _chunked(items: Sequence[GenerationInput], chunk_size: int) -> Iterable[Sequence[GenerationInput]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _chunked_by_token_budget(
    items: Sequence[GenerationInput],
    max_tokens_per_batch: int,
) -> Iterable[Sequence[GenerationInput]]:
    if max_tokens_per_batch < 1:
        raise SystemExit("--token-budgets values must be >= 1")

    current: List[GenerationInput] = []
    current_max_len = 0
    for item in items:
        seq_len = len(str(item.sequence))
        next_max_len = max(current_max_len, seq_len)
        projected_cost = (len(current) + 1) * next_max_len
        if current and projected_cost > max_tokens_per_batch:
            yield list(current)
            current = [item]
            current_max_len = seq_len
            continue
        current.append(item)
        current_max_len = next_max_len

    if current:
        yield list(current)


def _schedule_batches(
    items: Sequence[GenerationInput],
    *,
    batch_size: int | None,
    max_tokens_per_batch: int | None,
) -> List[List[GenerationInput]]:
    if (batch_size is None) == (max_tokens_per_batch is None):
        raise RuntimeError("Specify exactly one of batch_size or max_tokens_per_batch.")
    if batch_size is not None:
        return [list(chunk) for chunk in _chunked(items, batch_size)]
    return [list(chunk) for chunk in _chunked_by_token_budget(items, int(max_tokens_per_batch))]


def _to_hf_layer_index(user_layer_index: int, *, total_layers: int) -> int:
    return (total_layers - 1) - int(user_layer_index)


def _pool_rows(rows: Sequence[Sequence[float]], *, pooling: str) -> List[float]:
    if not rows:
        raise RuntimeError("Cannot pool an empty residue embedding.")
    if pooling == "first":
        return [float(value) for value in rows[0]]
    if pooling != "mean":
        raise RuntimeError(f"Unsupported pooling mode: {pooling!r}")
    dims = len(rows[0])
    accum = [0.0] * dims
    for row in rows:
        if len(row) != dims:
            raise RuntimeError("Inconsistent residue embedding width.")
        for index, value in enumerate(row):
            accum[index] += float(value)
    count = float(len(rows))
    return [value / count for value in accum]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        a_f = float(a)
        b_f = float(b)
        dot += a_f * b_f
        norm_a += a_f * a_f
        norm_b += b_f * b_f
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def _vector_diff_metrics(vec_a: Sequence[float], vec_b: Sequence[float]) -> Dict[str, float]:
    if len(vec_a) != len(vec_b):
        raise RuntimeError("Embedding dimensions do not match.")
    abs_diffs = [abs(float(a) - float(b)) for a, b in zip(vec_a, vec_b)]
    sq_diffs = [diff * diff for diff in abs_diffs]
    return {
        "cosine_similarity": _cosine_similarity(vec_a, vec_b),
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
        "mean_abs_diff": statistics.fmean(abs_diffs) if abs_diffs else 0.0,
        "l2_diff": math.sqrt(sum(sq_diffs)),
    }


def _summarize_scalar_metrics(rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not rows:
        return {}
    metric_names = sorted(rows[0].keys())
    summary: Dict[str, Dict[str, float]] = {}
    for metric_name in metric_names:
        values = [float(row[metric_name]) for row in rows]
        summary[metric_name] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def _search_embeddings(
    client: BioDataClient,
    *,
    backend: SearchBackend,
    query_ids: Sequence[str],
    query_vectors: Sequence[Sequence[float]],
    embedding_type_id: int,
    layer_index: int,
    metric: DistanceMetric,
    k: int,
    device: str | None,
) -> Dict[str, List[Neighbor]]:
    if backend == "pgvector":
        return {
            query_id: client.find_nearest_neighbors(
                vector,
                embedding_type_id=embedding_type_id,
                layer_index=layer_index,
                k=k,
                metric=metric,
                use_ann=False,
                backend="pgvector",
            )
            for query_id, vector in zip(query_ids, query_vectors)
        }
    if backend == "faiss_cpu":
        return client._find_nearest_neighbors_for_queries_faiss_cpu(
            query_ids,
            query_vectors,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            k=k,
            metric=metric,
            include_query=False,
            use_ann=False,
        )
    if backend == "faiss_gpu":
        return client._find_nearest_neighbors_for_queries_faiss(
            query_ids,
            query_vectors,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            k=k,
            metric=metric,
            include_query=False,
            device=device,
        )
    if backend == "cuvs_gpu":
        return client._find_nearest_neighbors_for_queries_cuvs(
            query_ids,
            query_vectors,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            k=k,
            metric=metric,
            include_query=False,
            device=device,
            use_ann=False,
        )
    return client._find_nearest_neighbors_for_queries_torch(
        query_ids,
        query_vectors,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
        k=k,
        metric=metric,
        include_query=False,
        device=device,
    )


def _neighbor_overlap_metrics(
    baseline: Mapping[str, Sequence[Neighbor]],
    candidate: Mapping[str, Sequence[Neighbor]],
    *,
    k: int,
) -> Dict[str, float]:
    query_ids = sorted(set(baseline.keys()) & set(candidate.keys()))
    if not query_ids:
        return {}

    exact_topk = 0
    exact_top1 = 0
    overlap_values: List[float] = []
    jaccard_values: List[float] = []

    for query_id in query_ids:
        base_ids = [neighbor.protein_id for neighbor in baseline[query_id][:k]]
        cand_ids = [neighbor.protein_id for neighbor in candidate[query_id][:k]]
        if base_ids == cand_ids:
            exact_topk += 1
        if base_ids and cand_ids and base_ids[0] == cand_ids[0]:
            exact_top1 += 1

        base_set = set(base_ids)
        cand_set = set(cand_ids)
        intersection = len(base_set & cand_set)
        union = len(base_set | cand_set)
        overlap_values.append(intersection / float(max(1, k)))
        jaccard_values.append(intersection / float(max(1, union)))

    query_count = len(query_ids)
    return {
        "query_count": float(query_count),
        "exact_top1_fraction": exact_top1 / float(query_count),
        "exact_topk_fraction": exact_topk / float(query_count),
        "mean_overlap_at_k": statistics.fmean(overlap_values),
        "mean_jaccard_at_k": statistics.fmean(jaccard_values),
    }


def _run_true_batched_embeddings(
    records: Sequence[GenerationInput],
    *,
    model_name: str,
    device: str,
    batch_size: int | None,
    max_tokens_per_batch: int | None,
    layer_index: int,
    pooling: str,
) -> Dict[str, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoTokenizer, T5EncoderModel  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers and torch are required. Install with: pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()

    started = time.perf_counter()
    pooled_vectors: Dict[str, List[float]] = {}
    total_layers: int | None = None
    batches = _schedule_batches(
        records,
        batch_size=batch_size,
        max_tokens_per_batch=max_tokens_per_batch,
    )

    with torch.inference_mode():
        for chunk in batches:
            prepared = [_preprocess_sequence(record.sequence) for record in chunk]
            encoded = tokenizer(
                prepared,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = model_output.hidden_states
            if hidden_states is None:
                raise RuntimeError("ProtT5 output did not include hidden_states.")
            total_layers = len(hidden_states)
            hf_layer_index = _to_hf_layer_index(layer_index, total_layers=total_layers)
            selected = hidden_states[hf_layer_index]

            for row_index, record in enumerate(chunk):
                residue_len = max(int(attention_mask[row_index].sum().item()) - 1, 1)
                matrix = selected[row_index, :residue_len].detach().cpu().tolist()
                pooled_vectors[str(record.id)] = _pool_rows(matrix, pooling=pooling)

            del model_output
            del input_ids
            del attention_mask
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    ordered_ids = [str(record.id) for record in records]
    ordered_vectors = [pooled_vectors[record_id] for record_id in ordered_ids]
    return {
        "query_ids": ordered_ids,
        "query_vectors": ordered_vectors,
        "batch_size": batch_size,
        "max_tokens_per_batch": max_tokens_per_batch,
        "elapsed_seconds": elapsed,
        "embedding_dim": len(ordered_vectors[0]) if ordered_vectors else 0,
        "layer_count": total_layers,
        "microbatch_count": len(batches),
        "max_realized_batch_size": max((len(chunk) for chunk in batches), default=0),
        "mean_realized_batch_size": (
            statistics.fmean(len(chunk) for chunk in batches) if batches else 0.0
        ),
    }


def _setting_key(*, batch_size: int | None, max_tokens_per_batch: int | None) -> str:
    if batch_size is not None:
        return str(int(batch_size))
    return f"tokens:{int(max_tokens_per_batch)}"


def main() -> None:
    args = _parse_args()
    records = _sample_inputs(args.fasta, count=args.count, seed=args.seed)
    if args.sort_by_length:
        records = _sort_inputs_by_length(records)
    _print_progress(f"Loaded {len(records)} sampled proteins from {args.fasta}")

    run_specs: List[Dict[str, int | None]] = [
        {"batch_size": int(batch_size), "max_tokens_per_batch": None}
        for batch_size in args.batch_sizes
    ]
    if args.token_budgets:
        run_specs.extend(
            {
                "batch_size": None,
                "max_tokens_per_batch": int(token_budget),
            }
            for token_budget in args.token_budgets
        )

    results_by_setting: Dict[str, Dict[str, Any]] = {}
    for spec in run_specs:
        key = _setting_key(
            batch_size=spec["batch_size"],
            max_tokens_per_batch=spec["max_tokens_per_batch"],
        )
        if spec["batch_size"] is not None:
            _print_progress(f"Embedding with true inference batch_size={spec['batch_size']}")
        else:
            _print_progress(f"Embedding with token_budget={spec['max_tokens_per_batch']}")
        results_by_setting[key] = _run_true_batched_embeddings(
            records,
            model_name=args.model_name,
            device=args.device,
            batch_size=spec["batch_size"],
            max_tokens_per_batch=spec["max_tokens_per_batch"],
            layer_index=int(args.layer_index),
            pooling=str(args.pooling),
        )

    baseline_batch = int(args.batch_sizes[0])
    baseline_key = _setting_key(batch_size=baseline_batch, max_tokens_per_batch=None)
    baseline = results_by_setting[baseline_key]
    comparisons: Dict[str, Any] = {}

    baseline_vectors = {
        query_id: vector for query_id, vector in zip(baseline["query_ids"], baseline["query_vectors"])
    }
    baseline_neighbors: Dict[str, List[Neighbor]] | None = None

    if args.embedding_type_id is not None:
        _print_progress(
            f"Running baseline neighbor search on backend={args.search_backend} with embedding_type_id={args.embedding_type_id}"
        )
        with BioDataClient() as client:
            baseline_neighbors = _search_embeddings(
                client,
                backend=args.search_backend,
                query_ids=baseline["query_ids"],
                query_vectors=baseline["query_vectors"],
                embedding_type_id=int(args.embedding_type_id),
                layer_index=int(args.layer_index),
                metric=args.metric,
                k=int(args.k),
                device=args.device,
            )

            for spec in run_specs[1:]:
                candidate_key = _setting_key(
                    batch_size=spec["batch_size"],
                    max_tokens_per_batch=spec["max_tokens_per_batch"],
                )
                candidate = results_by_setting[candidate_key]
                per_query_metrics: List[Dict[str, float]] = []
                for query_id, vector in zip(candidate["query_ids"], candidate["query_vectors"]):
                    per_query_metrics.append(_vector_diff_metrics(baseline_vectors[query_id], vector))

                _print_progress(f"Running neighbor comparison for setting={candidate_key}")
                candidate_neighbors = _search_embeddings(
                    client,
                    backend=args.search_backend,
                    query_ids=candidate["query_ids"],
                    query_vectors=candidate["query_vectors"],
                    embedding_type_id=int(args.embedding_type_id),
                    layer_index=int(args.layer_index),
                    metric=args.metric,
                    k=int(args.k),
                    device=args.device,
                )
                comparisons[candidate_key] = {
                    "vs_batch_size": baseline_batch,
                    "embedding_metrics": _summarize_scalar_metrics(per_query_metrics),
                    "search_metrics": _neighbor_overlap_metrics(
                        baseline_neighbors,
                        candidate_neighbors,
                        k=int(args.k),
                    ),
                }
    else:
        for spec in run_specs[1:]:
            candidate_key = _setting_key(
                batch_size=spec["batch_size"],
                max_tokens_per_batch=spec["max_tokens_per_batch"],
            )
            candidate = results_by_setting[candidate_key]
            per_query_metrics = []
            for query_id, vector in zip(candidate["query_ids"], candidate["query_vectors"]):
                per_query_metrics.append(_vector_diff_metrics(baseline_vectors[query_id], vector))
            comparisons[candidate_key] = {
                "vs_batch_size": baseline_batch,
                "embedding_metrics": _summarize_scalar_metrics(per_query_metrics),
            }

    payload = {
        "fasta": str(args.fasta),
        "sample_count": len(records),
        "seed": args.seed,
        "model_name": args.model_name,
        "device": args.device,
        "layer_index": args.layer_index,
        "pooling": args.pooling,
        "sort_by_length": bool(args.sort_by_length),
        "batch_sizes": [int(value) for value in args.batch_sizes],
        "token_budgets": [int(value) for value in (args.token_budgets or [])],
        "baseline_batch_size": baseline_batch,
        "embedding_type_id": args.embedding_type_id,
        "search_backend": args.search_backend if args.embedding_type_id is not None else None,
        "metric": args.metric,
        "k": args.k,
        "embedding_runs": {
            key: {
                "elapsed_seconds": run["elapsed_seconds"],
                "embedding_dim": run["embedding_dim"],
                "layer_count": run["layer_count"],
                "batch_size": run["batch_size"],
                "max_tokens_per_batch": run["max_tokens_per_batch"],
                "microbatch_count": run["microbatch_count"],
                "max_realized_batch_size": run["max_realized_batch_size"],
                "mean_realized_batch_size": run["mean_realized_batch_size"],
            }
            for key, run in results_by_setting.items()
        },
        "comparisons": comparisons,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    print("setting\tseconds\tvs_baseline_cosine_mean\tvs_baseline_max_abs_mean\tsearch_exact_topk", flush=True)
    print(f"{baseline_key}\t{baseline['elapsed_seconds']:.6f}\t1.000000\t0.000000\t1.000000", flush=True)
    for spec in run_specs[1:]:
        candidate_key = _setting_key(
            batch_size=spec["batch_size"],
            max_tokens_per_batch=spec["max_tokens_per_batch"],
        )
        comparison = comparisons[candidate_key]
        embedding_metrics = comparison["embedding_metrics"]
        cosine_mean = embedding_metrics["cosine_similarity"]["mean"]
        max_abs_mean = embedding_metrics["max_abs_diff"]["mean"]
        exact_topk = comparison.get("search_metrics", {}).get("exact_topk_fraction", float("nan"))
        print(
            f"{candidate_key}\t{results_by_setting[candidate_key]['elapsed_seconds']:.6f}\t"
            f"{cosine_mean:.6f}\t{max_abs_mean:.6e}\t{exact_topk:.6f}",
            flush=True,
        )
    _print_progress(f"JSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
