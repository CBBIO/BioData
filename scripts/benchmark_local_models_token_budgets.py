#!/usr/bin/env python3
"""Benchmark local embedding generators under sorted token-budget batching."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.embeddings import GenerationInput, Generator, load_fasta_inputs


DEFAULT_MODELS = ["protT5", "prostT5", "ankh3", "esm2", "esm1b", "esmc"]
DEFAULT_TOKEN_BUDGETS = [4096, 8192, 12288]
_TRUE_BATCH_TOKEN_BUDGET_MODELS = {"prott5", "prostt5", "ankh3", "esm2", "esm1b"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local embedding models with a single-sequence baseline and "
            "sorted token-budget batching where true batched forward passes are supported."
        ),
    )
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file.")
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of proteins sampled from FASTA for each model. Default: 1000.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Local generator classes to benchmark.",
    )
    parser.add_argument(
        "--token-budgets",
        type=int,
        nargs="+",
        default=DEFAULT_TOKEN_BUDGETS,
        help="Sorted token-budget settings to compare against the single-sequence baseline.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed. Default: 7.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Generator device for local inference. Default: cuda:0.",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="Layer index used for embedding generation. Default: 0.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "first"],
        default="mean",
        help="How to reduce per-residue embeddings. Default: mean.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "benchmark_local_models_token_budgets.json",
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
    sort_by_length: bool,
) -> List[List[GenerationInput]]:
    if (batch_size is None) == (max_tokens_per_batch is None):
        raise RuntimeError("Specify exactly one of batch_size or max_tokens_per_batch.")

    scheduled = list(items)
    if sort_by_length:
        scheduled = _sort_inputs_by_length(scheduled)
    if batch_size is not None:
        if batch_size < 1:
            raise SystemExit("--batch-size must be >= 1")
        return [scheduled[index : index + batch_size] for index in range(0, len(scheduled), batch_size)]
    if max_tokens_per_batch is None:
        raise RuntimeError("max_tokens_per_batch must be provided when batch_size is not set.")
    return [list(chunk) for chunk in _chunked_by_token_budget(scheduled, max_tokens_per_batch)]


def _setting_key(*, batch_size: int | None, max_tokens_per_batch: int | None) -> str:
    if (batch_size is None) == (max_tokens_per_batch is None):
        raise RuntimeError("Specify exactly one of batch_size or max_tokens_per_batch.")
    if batch_size is not None:
        return str(batch_size)
    if max_tokens_per_batch is None:
        raise RuntimeError("max_tokens_per_batch must be provided when batch_size is not set.")
    return f"tokens:{max_tokens_per_batch}"


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


def _maybe_import_torch() -> Any | None:
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return None
    return torch


def _maybe_release_cuda_cache(device: str | None) -> None:
    if device is None or not str(device).startswith("cuda"):
        return
    torch = _maybe_import_torch()
    if torch is None or not torch.cuda.is_available():
        return
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        return


def _reset_peak_cuda_memory(device: str | None) -> None:
    if device is None or not str(device).startswith("cuda"):
        return
    torch = _maybe_import_torch()
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def _peak_cuda_memory_bytes(device: str | None) -> int | None:
    if device is None or not str(device).startswith("cuda"):
        return None
    torch = _maybe_import_torch()
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return None


def _extract_pooled_vectors_from_generation(
    generated: Any,
    *,
    pooling: str,
) -> tuple[List[str], List[List[float]]]:
    pooled_ids: List[str] = []
    pooled_vectors: List[List[float]] = []
    for record in generated.records:
        pooled_ids.append(str(record.id))
        pooled_vectors.append(_pool_rows(record.embedding, pooling=pooling))
    if not pooled_vectors:
        raise RuntimeError("Embedding generation produced no pooled query vectors.")
    return pooled_ids, pooled_vectors


def _run_baseline(
    generator: Any,
    inputs: Sequence[GenerationInput],
    *,
    layer_index: int,
    pooling: str,
    generator_device: str | None,
) -> Dict[str, Any]:
    _reset_peak_cuda_memory(generator_device)
    started = time.perf_counter()
    generated = generator.generate(inputs, layer_index=layer_index, fail_fast=True)
    elapsed = time.perf_counter() - started
    query_ids, query_vectors = _extract_pooled_vectors_from_generation(generated, pooling=pooling)
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "record_count": len(query_vectors),
        "embedding_dim": len(query_vectors[0]),
        "query_ids": query_ids,
        "query_vectors": query_vectors,
        "batch_size": 1,
        "max_tokens_per_batch": None,
        "microbatch_count": len(query_vectors),
        "max_realized_batch_size": 1,
        "mean_realized_batch_size": 1.0,
        "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(generator_device),
    }


def _batched_t5_forward(
    generator: Any,
    inputs: Sequence[GenerationInput],
    *,
    layer_index: int,
    pooling: str,
    max_tokens_per_batch: int,
    generator_device: str | None,
    start_token_count: int,
    reverse_layer_indexing: bool,
) -> Dict[str, Any]:
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for token-budget batching.")

    tokenizer = getattr(getattr(generator, "tokenizer", None), "tokenizer", None)
    model = getattr(getattr(generator, "model", None), "model", None)
    preprocessor = getattr(getattr(generator, "preprocessor", None), "preprocess", None)
    if tokenizer is None or model is None or not callable(preprocessor):
        raise RuntimeError("Could not access tokenizer/model/preprocessor for batched T5 inference.")

    chunks = _schedule_batches(
        inputs,
        batch_size=None,
        max_tokens_per_batch=max_tokens_per_batch,
        sort_by_length=True,
    )
    pooled_ids: List[str] = []
    pooled_vectors: List[List[float]] = []
    _reset_peak_cuda_memory(generator_device)
    started = time.perf_counter()

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
                raise RuntimeError("Model output did not include hidden_states.")
            hf_layer_index = int(layer_index)
            if reverse_layer_indexing:
                hf_layer_index = (len(hidden_states) - 1) - int(layer_index)
            selected = hidden_states[hf_layer_index]
            for row_index, record in enumerate(chunk):
                valid_len = int(attention_mask[row_index].sum().item())
                if valid_len <= start_token_count:
                    raise RuntimeError("Expected at least one residue plus special tokens in batched input.")
                matrix = selected[row_index, start_token_count : valid_len - 1].detach().cpu().tolist()
                pooled_ids.append(str(record.id))
                pooled_vectors.append(_pool_rows(matrix, pooling=pooling))
            del model_output
            del input_ids
            del attention_mask
            _maybe_release_cuda_cache(generator_device)

    elapsed = time.perf_counter() - started
    if not pooled_vectors:
        raise RuntimeError("Embedding generation produced no pooled query vectors.")
    realized_sizes = [len(chunk) for chunk in chunks]
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "record_count": len(pooled_vectors),
        "embedding_dim": len(pooled_vectors[0]),
        "query_ids": pooled_ids,
        "query_vectors": pooled_vectors,
        "batch_size": None,
        "max_tokens_per_batch": max_tokens_per_batch,
        "microbatch_count": len(chunks),
        "max_realized_batch_size": max(realized_sizes),
        "mean_realized_batch_size": statistics.fmean(realized_sizes),
        "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(generator_device),
    }


def _batched_esm_forward(
    generator: Any,
    inputs: Sequence[GenerationInput],
    *,
    layer_index: int,
    pooling: str,
    max_tokens_per_batch: int,
    generator_device: str | None,
) -> Dict[str, Any]:
    torch = _maybe_import_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for token-budget batching.")

    batch_converter = getattr(getattr(generator, "tokenizer", None), "batch_converter", None)
    padding_idx = getattr(getattr(generator, "tokenizer", None), "padding_idx", None)
    hf_tokenizer = getattr(getattr(generator, "tokenizer", None), "tokenizer", None)
    model = getattr(getattr(generator, "model", None), "model", None)
    preprocessor = getattr(getattr(generator, "preprocessor", None), "preprocess", None)
    if model is None or not callable(preprocessor):
        raise RuntimeError("Could not access model/preprocessor for batched ESM inference.")
    if (batch_converter is None or padding_idx is None) and hf_tokenizer is None:
        raise RuntimeError("Could not access ESM batch_converter or HF tokenizer for batched ESM inference.")

    chunks = _schedule_batches(
        inputs,
        batch_size=None,
        max_tokens_per_batch=max_tokens_per_batch,
        sort_by_length=True,
    )
    pooled_ids: List[str] = []
    pooled_vectors: List[List[float]] = []
    _reset_peak_cuda_memory(generator_device)
    started = time.perf_counter()

    model.eval()
    with torch.inference_mode():
        for chunk in chunks:
            prepared = [preprocessor(record.sequence) for record in chunk]
            if hf_tokenizer is not None:
                encoded = hf_tokenizer(prepared, add_special_tokens=True, padding=True, return_tensors="pt")
                input_ids = encoded["input_ids"].to(generator_device or "cpu")
                attention_mask = encoded["attention_mask"].to(generator_device or "cpu")
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden_states = getattr(output, "hidden_states", None)
                if hidden_states is None:
                    raise RuntimeError("HF ESM output missing hidden_states.")
                layer_tensor = hidden_states[int(layer_index)]
                for row_index, record in enumerate(chunk):
                    tokens_len = int(attention_mask[row_index].sum().item())
                    if tokens_len < 3:
                        raise RuntimeError("Expected at least one residue plus BOS/EOS in ESM input.")
                    matrix = layer_tensor[row_index, 1 : tokens_len - 1].detach().cpu().tolist()
                    pooled_ids.append(str(record.id))
                    pooled_vectors.append(_pool_rows(matrix, pooling=pooling))
                del input_ids
                del attention_mask
                del output
            else:
                if batch_converter is None or padding_idx is None:
                    raise RuntimeError("ESM batch_converter and padding_idx are required for non-HF batched inference.")
                labeled = [(str(record.id), sequence) for record, sequence in zip(chunk, prepared)]
                _, _, tokens = batch_converter(labeled)
                tokens = tokens.to(generator_device or "cpu")
                lengths = (tokens != padding_idx).sum(1)
                output = model(tokens, repr_layers=[int(layer_index)], return_contacts=False)
                representations = output.get("representations")
                if not isinstance(representations, dict):
                    raise RuntimeError("ESM output missing representations.")
                layer_tensor = representations.get(int(layer_index))
                if layer_tensor is None:
                    raise RuntimeError(f"ESM output missing requested layer {layer_index}.")
                for row_index, record in enumerate(chunk):
                    tokens_len = int(lengths[row_index].item())
                    if tokens_len < 3:
                        raise RuntimeError("Expected at least one residue plus BOS/EOS in ESM input.")
                    matrix = layer_tensor[row_index, 1 : tokens_len - 1].detach().cpu().tolist()
                    pooled_ids.append(str(record.id))
                    pooled_vectors.append(_pool_rows(matrix, pooling=pooling))
                del output
                del tokens
            _maybe_release_cuda_cache(generator_device)

    elapsed = time.perf_counter() - started
    if not pooled_vectors:
        raise RuntimeError("Embedding generation produced no pooled query vectors.")
    realized_sizes = [len(chunk) for chunk in chunks]
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "record_count": len(pooled_vectors),
        "embedding_dim": len(pooled_vectors[0]),
        "query_ids": pooled_ids,
        "query_vectors": pooled_vectors,
        "batch_size": None,
        "max_tokens_per_batch": max_tokens_per_batch,
        "microbatch_count": len(chunks),
        "max_realized_batch_size": max(realized_sizes),
        "mean_realized_batch_size": statistics.fmean(realized_sizes),
        "peak_cuda_memory_bytes": _peak_cuda_memory_bytes(generator_device),
    }


def _run_token_budget(
    generator: Any,
    generator_class: str,
    inputs: Sequence[GenerationInput],
    *,
    max_tokens_per_batch: int,
    layer_index: int,
    pooling: str,
    generator_device: str | None,
) -> Dict[str, Any]:
    normalized = str(generator_class).strip().lower()
    if normalized == "prott5":
        return _batched_t5_forward(
            generator,
            inputs,
            layer_index=layer_index,
            pooling=pooling,
            max_tokens_per_batch=max_tokens_per_batch,
            generator_device=generator_device,
            start_token_count=1,
            reverse_layer_indexing=True,
        )
    if normalized == "prostt5":
        return _batched_t5_forward(
            generator,
            inputs,
            layer_index=layer_index,
            pooling=pooling,
            max_tokens_per_batch=max_tokens_per_batch,
            generator_device=generator_device,
            start_token_count=1,
            reverse_layer_indexing=True,
        )
    if normalized == "ankh3":
        return _batched_t5_forward(
            generator,
            inputs,
            layer_index=layer_index,
            pooling=pooling,
            max_tokens_per_batch=max_tokens_per_batch,
            generator_device=generator_device,
            start_token_count=1,
            reverse_layer_indexing=False,
        )
    if normalized in {"esm2", "esm1b"}:
        return _batched_esm_forward(
            generator,
            inputs,
            layer_index=layer_index,
            pooling=pooling,
            max_tokens_per_batch=max_tokens_per_batch,
            generator_device=generator_device,
        )
    raise RuntimeError(f"True token-budget batching is not implemented for {generator_class}.")


def _compare_to_baseline(
    baseline_vectors: Mapping[str, Sequence[float]],
    candidate_vectors: Mapping[str, Sequence[float]],
) -> Dict[str, Dict[str, float]]:
    shared_ids = sorted(set(baseline_vectors.keys()) & set(candidate_vectors.keys()))
    rows = [_vector_diff_metrics(baseline_vectors[item_id], candidate_vectors[item_id]) for item_id in shared_ids]
    return _summarize_scalar_metrics(rows)


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def _print_report(payload: Dict[str, Any]) -> None:
    print("model\tsetting\tseconds\tproteins_per_s\tspeedup_vs_baseline\tcosine_mean\tmax_abs_mean\tstatus")
    for model_name, model_payload in payload["models"].items():
        if model_payload.get("status") != "ok":
            print(
                "\t".join(
                    [
                        model_name,
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                        f"unavailable: {model_payload.get('error', '-')}",
                    ]
                )
            )
            continue
        settings = model_payload.get("settings", {})
        baseline_seconds = None
        baseline_payload = settings.get("1")
        if isinstance(baseline_payload, dict) and baseline_payload.get("status") == "ok":
            baseline_seconds = float(baseline_payload["elapsed_seconds"])
        for setting_name, setting_payload in settings.items():
            status = str(setting_payload["status"])
            if status != "ok":
                print(
                    "\t".join(
                        [
                            model_name,
                            setting_name,
                            "-",
                            "-",
                            "-",
                            "-",
                            "-",
                            f"{status}: {setting_payload.get('error', '-')}",
                        ]
                    )
                )
                continue
            seconds = float(setting_payload["elapsed_seconds"])
            proteins_per_s = float(setting_payload["proteins_per_second"])
            speedup = baseline_seconds / seconds if baseline_seconds and seconds > 0 else 1.0
            comparison = model_payload.get("comparisons", {}).get(setting_name, {})
            cosine_summary = comparison.get("cosine_similarity", {})
            max_abs_summary = comparison.get("max_abs_diff", {})
            print(
                "\t".join(
                    [
                        model_name,
                        setting_name,
                        _format_seconds(seconds),
                        f"{proteins_per_s:.2f}",
                        f"{speedup:.3f}x",
                        f"{float(cosine_summary.get('mean', 1.0)):.12f}",
                        f"{float(max_abs_summary.get('mean', 0.0)):.6e}",
                        status,
                    ]
                )
            )


def _strip_large_setting_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    trimmed = json.loads(json.dumps(payload))
    for model_payload in trimmed.get("models", {}).values():
        for setting_payload in model_payload.get("settings", {}).values():
            setting_payload.pop("query_ids", None)
            setting_payload.pop("query_vectors", None)
    return trimmed


def main() -> None:
    args = _parse_args()
    sampled_inputs = _sample_inputs(args.fasta, count=args.count, seed=args.seed)
    _print_progress(f"Loaded {len(sampled_inputs)} sampled proteins from {args.fasta}")

    payload: Dict[str, Any] = {
        "fasta": str(args.fasta),
        "sample_count": len(sampled_inputs),
        "seed": int(args.seed),
        "device": args.device,
        "layer_index": int(args.layer_index),
        "pooling": args.pooling,
        "sort_by_length": True,
        "token_budgets": [int(value) for value in args.token_budgets],
        "models": {},
    }

    for model_class in args.models:
        _print_progress(f"Initializing generator class={model_class} on {args.device}")
        init_started = time.perf_counter()
        try:
            generator = Generator(model_class=model_class, device=args.device)
        except Exception as exc:
            payload["models"][model_class] = {
                "status": "unavailable",
                "error": str(exc),
            }
            continue
        init_seconds = time.perf_counter() - init_started
        model_payload: Dict[str, Any] = {
            "status": "ok",
            "model_reference": getattr(generator, "model_reference", None),
            "generator_init_seconds": init_seconds,
            "supports_true_token_budget_batching": model_class.lower() in _TRUE_BATCH_TOKEN_BUDGET_MODELS,
            "settings": {},
            "comparisons": {},
        }

        _print_progress(f"Benchmarking {model_class} baseline 1 seq/pass")
        baseline = _run_baseline(
            generator,
            sampled_inputs,
            layer_index=args.layer_index,
            pooling=args.pooling,
            generator_device=args.device,
        )
        baseline["proteins_per_second"] = baseline["record_count"] / float(baseline["elapsed_seconds"])
        baseline_key = _setting_key(batch_size=1, max_tokens_per_batch=None)
        model_payload["settings"][baseline_key] = baseline
        baseline_vectors = {
            item_id: vector for item_id, vector in zip(baseline["query_ids"], baseline["query_vectors"])
        }
        model_payload["comparisons"][baseline_key] = _summarize_scalar_metrics(
            [_vector_diff_metrics(vector, vector) for vector in baseline_vectors.values()]
        )

        for token_budget in args.token_budgets:
            setting_key = _setting_key(batch_size=None, max_tokens_per_batch=token_budget)
            _print_progress(f"Benchmarking {model_class} {setting_key}")
            if model_class.lower() not in _TRUE_BATCH_TOKEN_BUDGET_MODELS:
                model_payload["settings"][setting_key] = {
                    "status": "unsupported",
                    "error": f"True token-budget batching is not implemented for {model_class}.",
                    "max_tokens_per_batch": int(token_budget),
                }
                continue
            try:
                setting = _run_token_budget(
                    generator,
                    model_class,
                    sampled_inputs,
                    max_tokens_per_batch=int(token_budget),
                    layer_index=args.layer_index,
                    pooling=args.pooling,
                    generator_device=args.device,
                )
                setting["proteins_per_second"] = setting["record_count"] / float(setting["elapsed_seconds"])
                model_payload["settings"][setting_key] = setting
                candidate_vectors = {
                    item_id: vector for item_id, vector in zip(setting["query_ids"], setting["query_vectors"])
                }
                model_payload["comparisons"][setting_key] = _compare_to_baseline(
                    baseline_vectors,
                    candidate_vectors,
                )
            except Exception as exc:
                model_payload["settings"][setting_key] = {
                    "status": "error",
                    "error": str(exc),
                    "max_tokens_per_batch": int(token_budget),
                }

        payload["models"][model_class] = model_payload
        _maybe_release_cuda_cache(args.device)

    json_payload = _strip_large_setting_fields(payload)
    args.json_out.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    _print_report(payload)
    print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
