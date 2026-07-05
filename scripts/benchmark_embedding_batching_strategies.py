#!/usr/bin/env python3
"""Benchmark embedding generators across batching strategies and length limits."""

from __future__ import annotations

import argparse
import gc
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

from CBBIO.embeddings import GenerationInput, load_fasta_inputs
from CBBIO import EmbeddingWriter, Generator, IterableBatcher, pooler_factory, run_embedding_generation


DEFAULT_MODELS = ["protT5", "prostT5", "ankh3", "esm2", "esm1b", "esmc"]
DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
DEFAULT_TOKEN_BUDGETS = [4096, 8192, 16384]
DEFAULT_LENGTH_LIMITS = [None, 512, 1024]
DEFAULT_FASTA_PATH = _SCRIPT_REPO_ROOT / "scripts" / "data" / "random_protein_ids_10000_lenle1000.fasta"


def _default_ints_as_strings(values: Sequence[int]) -> List[str]:
    return [str(value) for value in values]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local embedding models with fixed-size batches, token-budget batches, "
            "and optional protein length limits."
        ),
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=DEFAULT_FASTA_PATH,
        help=f"Input FASTA file. Default: {DEFAULT_FASTA_PATH.relative_to(_SCRIPT_REPO_ROOT)}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Generator classes to benchmark. Default: {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of FASTA records sampled before length filtering. Default: 100.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Sampling seed. Default: 7.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Generator device for all models. Default: cuda:0.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Optional torch dtype passed to generators that support it, e.g. float16.",
    )
    parser.add_argument("--layer-index", type=int, default=0, help="Layer index. Default: 0.")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        nargs="+",
        default=_default_ints_as_strings(DEFAULT_BATCH_SIZES),
        help="Fixed batch sizes to benchmark, or 'none' to disable. Default: 1 2 4 8.",
    )
    parser.add_argument(
        "--token-budgets",
        type=str,
        nargs="+",
        default=_default_ints_as_strings(DEFAULT_TOKEN_BUDGETS),
        help="Padded token-budget settings, or 'none' to disable. Default: 4096 8192 16384.",
    )
    parser.add_argument(
        "--length-limits",
        type=str,
        nargs="+",
        default=["none", "512", "1024"],
        help="Max protein lengths to benchmark. Use 'none' for no limit. Default: none 512 1024.",
    )
    parser.add_argument(
        "--length-sort-window",
        type=int,
        default=None,
        help=(
            "Window size used for sorted batching settings. "
            "Default: sampled count, i.e. full sampled set sorted by length."
        ),
    )
    parser.add_argument(
        "--sort-modes",
        choices=["off", "on", "both"],
        default="both",
        help="Benchmark batching without length sorting, with length sorting, or both. Default: both.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeated runs per model/strategy/length-limit. Default: 1.",
    )
    parser.add_argument(
        "--repeat-sampling",
        choices=["same", "resample"],
        default="same",
        help=(
            "Use the same sampled proteins for every repeat, or draw a deterministic "
            "new sample per repeat and reuse it across all models/settings. Default: same."
        ),
    )
    parser.add_argument(
        "--pooler",
        choices=["none", "mean"],
        default="mean",
        help="Run-level pooler. Default: mean.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise on the first generation failure instead of collecting recoverable errors.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "notebooks" / "data" / "benchmark_embedding_batching_strategies.json",
        help="JSON output path.",
    )
    return parser.parse_args()


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    return f"{seconds:.3f} s"


def _format_mean_sd(mean: float, stdev: float, *, decimals: int = 2) -> str:
    if stdev <= 0:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f}+-{stdev:.{decimals}f}"


def _format_seconds_mean_sd(mean: float, stdev: float) -> str:
    if stdev <= 0:
        return _format_seconds(mean)
    return f"{_format_seconds(mean)}+-{_format_seconds(stdev)}"


def _format_skipped(run: Dict[str, Any]) -> str:
    skipped = int(run.get("skipped_count", 0))
    total = int(run.get("input_count", 0))
    percent = (100.0 * skipped / float(total)) if total > 0 else 0.0
    return f"skipped={skipped} ({percent:.1f}%)"


def _cuda_oom_error_count(errors: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for error in errors if str(error.get("reason", "")).strip().lower() == "cuda_oom")


def _parse_length_limits(values: Sequence[str]) -> List[int | None]:
    limits: List[int | None] = []
    for value in values:
        text = str(value).strip().lower()
        if text in {"none", "null", "no", "unlimited"}:
            limits.append(None)
            continue
        resolved = int(text)
        if resolved < 1:
            raise SystemExit("--length-limits values must be >= 1 or 'none'")
        limits.append(resolved)
    return limits


def _parse_positive_int_values(values: Sequence[str], *, flag_name: str) -> List[int]:
    if not values:
        return []

    normalized = [str(value).strip().lower() for value in values]
    if any(value in {"none", "null", "no", "off", "disable", "disabled"} for value in normalized):
        if len(values) > 1:
            raise SystemExit(f"{flag_name} accepts either positive integers or a single 'none'")
        return []

    resolved: List[int] = []
    for value in values:
        parsed = int(str(value).strip())
        if parsed < 1:
            raise SystemExit(f"{flag_name} values must be >= 1")
        resolved.append(parsed)
    return resolved


def _load_repeat_samples(
    path: Path,
    *,
    count: int,
    seed: int,
    repeats: int,
    mode: str,
) -> List[List[GenerationInput]]:
    inputs = load_fasta_inputs(path)
    if count < 1:
        raise SystemExit("--count must be >= 1")
    if repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if count > len(inputs):
        raise SystemExit(f"--count {count} exceeds FASTA pool size {len(inputs)}")
    rng = random.Random(seed)
    if mode == "same":
        sample = rng.sample(list(inputs), count)
        return [list(sample) for _ in range(repeats)]
    if mode == "resample":
        return [rng.sample(list(inputs), count) for _ in range(repeats)]
    raise SystemExit("--repeat-sampling must be one of: same, resample")


def _filter_by_length(records: Sequence[GenerationInput], limit: int | None) -> List[GenerationInput]:
    if limit is None:
        return list(records)
    return [record for record in records if len(record.sequence) <= limit]


def _length_sort_windows(records: Sequence[GenerationInput], window_size: int | None) -> List[GenerationInput]:
    if window_size is None:
        return list(records)
    resolved = int(window_size)
    if resolved < 1:
        raise SystemExit("--length-sort-window must be >= 1 when provided")
    ordered: List[GenerationInput] = []
    for start in range(0, len(records), resolved):
        window = list(records[start : start + resolved])
        window.sort(key=lambda record: len(record.sequence), reverse=True)
        ordered.extend(window)
    return ordered


def _setting_key(setting: Dict[str, Any]) -> str:
    suffix = "sorted" if setting.get("sort_by_length") else "unsorted"
    if setting["kind"] == "fixed":
        return f"batch:{setting['batch_size']}:{suffix}"
    return f"tokens:{setting['max_batch_tokens']}:{suffix}"


def _build_settings(
    batch_sizes: Sequence[int],
    token_budgets: Sequence[int],
    *,
    sort_modes: str,
) -> List[Dict[str, Any]]:
    settings: List[Dict[str, Any]] = []
    sort_values = [False, True]
    if sort_modes == "off":
        sort_values = [False]
    elif sort_modes == "on":
        sort_values = [True]
    for value in batch_sizes:
        resolved = int(value)
        fixed_sort_values = [False] if resolved == 1 else sort_values
        for sort_by_length in fixed_sort_values:
            settings.append(
                {
                    "kind": "fixed",
                    "batch_size": resolved,
                    "max_batch_tokens": None,
                    "sort_by_length": sort_by_length,
                }
            )
    for value in token_budgets:
        resolved = int(value)
        for sort_by_length in sort_values:
            settings.append(
                {
                    "kind": "token_budget",
                    "batch_size": None,
                    "max_batch_tokens": resolved,
                    "sort_by_length": sort_by_length,
                }
            )
    return settings


def _generator_max_sequence_length(generator: Any) -> int | None:
    value = getattr(generator, "max_sequence_length", None)
    if value is None:
        value = getattr(generator, "MAX_SEQUENCE_LENGTH", None)
    if value is None:
        metadata = getattr(generator, "model_metadata", None)
        parameters = getattr(metadata, "parameters", None)
        if isinstance(parameters, dict):
            value = parameters.get("max_sequence_length")
    if value is None:
        return None
    resolved = int(value)
    return resolved if resolved > 0 else None


def _resolve_effective_length_limit(generator: Any, requested_limit: int | None) -> int | None:
    model_limit = _generator_max_sequence_length(generator)
    if model_limit is None:
        return requested_limit
    if requested_limit is None:
        return model_limit
    return min(int(requested_limit), model_limit)


def _length_limit_key(requested_limit: int | None, effective_limit: int | None) -> str:
    requested_key = "none" if requested_limit is None else str(requested_limit)
    if effective_limit == requested_limit:
        return requested_key
    effective_key = "none" if effective_limit is None else str(effective_limit)
    if requested_limit is None:
        return f"auto:{effective_key}"
    return f"{requested_key}->{effective_key}"


def _summarize_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    timings = [float(run["elapsed_seconds"]) for run in runs if run.get("status") == "ok"]
    if not timings:
        return {"ok_runs": 0, "runs": len(runs)}
    records_per_second = [float(run["records_per_second"]) for run in runs if run.get("status") == "ok"]
    aa_per_second = [float(run["amino_acids_per_second"]) for run in runs if run.get("status") == "ok"]
    skipped_counts = [int(run.get("skipped_count", 0)) for run in runs if run.get("status") == "ok"]
    length_skipped_counts = [int(run.get("length_skipped_count", 0)) for run in runs if run.get("status") == "ok"]
    oom_skipped_counts = [int(run.get("oom_skipped_count", 0)) for run in runs if run.get("status") == "ok"]
    accepted_counts = [int(run.get("accepted_count", 0)) for run in runs if run.get("status") == "ok"]
    error_counts = [int(run.get("error_count", 0)) for run in runs if run.get("status") == "ok"]
    mean_batch_sizes = [float(run.get("mean_realized_batch_size", 0.0)) for run in runs if run.get("status") == "ok"]
    max_batch_sizes = [int(run.get("max_realized_batch_size", 0)) for run in runs if run.get("status") == "ok"]
    min_batch_sizes = [int(run.get("min_realized_batch_size", 0)) for run in runs if run.get("status") == "ok"]
    return {
        "runs": len(runs),
        "ok_runs": len(timings),
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "mean_records_per_second": statistics.fmean(records_per_second),
        "stdev_records_per_second": statistics.stdev(records_per_second) if len(records_per_second) > 1 else 0.0,
        "mean_amino_acids_per_second": statistics.fmean(aa_per_second),
        "stdev_amino_acids_per_second": statistics.stdev(aa_per_second) if len(aa_per_second) > 1 else 0.0,
        "mean_accepted_count": statistics.fmean(accepted_counts) if accepted_counts else 0.0,
        "min_accepted_count": min(accepted_counts) if accepted_counts else 0,
        "max_accepted_count": max(accepted_counts) if accepted_counts else 0,
        "mean_skipped_count": statistics.fmean(skipped_counts) if skipped_counts else 0.0,
        "min_skipped_count": min(skipped_counts) if skipped_counts else 0,
        "max_skipped_count": max(skipped_counts) if skipped_counts else 0,
        "mean_length_skipped_count": statistics.fmean(length_skipped_counts) if length_skipped_counts else 0.0,
        "min_length_skipped_count": min(length_skipped_counts) if length_skipped_counts else 0,
        "max_length_skipped_count": max(length_skipped_counts) if length_skipped_counts else 0,
        "mean_oom_skipped_count": statistics.fmean(oom_skipped_counts) if oom_skipped_counts else 0.0,
        "min_oom_skipped_count": min(oom_skipped_counts) if oom_skipped_counts else 0,
        "max_oom_skipped_count": max(oom_skipped_counts) if oom_skipped_counts else 0,
        "max_error_count": max(error_counts) if error_counts else 0,
        "mean_realized_batch_size": statistics.fmean(mean_batch_sizes) if mean_batch_sizes else 0.0,
        "min_realized_batch_size": min(min_batch_sizes) if min_batch_sizes else 0,
        "max_realized_batch_size": max(max_batch_sizes) if max_batch_sizes else 0,
    }


def _maybe_release_cuda_cache(device: str | None) -> None:
    if device is None or not str(device).startswith("cuda"):
        return
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


def _run_once(
    generator: Any,
    records: Sequence[GenerationInput],
    *,
    setting: Dict[str, Any],
    length_limit: int | None,
    length_sort_window: int | None,
    layer_index: int,
    pooler: str,
    fail_fast: bool,
    device: str | None,
) -> Dict[str, Any]:
    writer = EmbeddingWriter(format="memory")
    events: List[Dict[str, Any]] = []
    resolved_sort_window = length_sort_window if length_sort_window is not None else len(records)
    scheduled_records = (
        _length_sort_windows(records, resolved_sort_window)
        if bool(setting.get("sort_by_length"))
        else list(records)
    )
    batcher = IterableBatcher(
        scheduled_records,
        batch_size=setting["batch_size"],
        max_batch_tokens=setting["max_batch_tokens"],
        max_sequence_length=length_limit,
    )
    realized_batch_sizes = [len(batch) for batch in batcher]
    batcher = IterableBatcher(
        scheduled_records,
        batch_size=setting["batch_size"],
        max_batch_tokens=setting["max_batch_tokens"],
        max_sequence_length=length_limit,
    )
    effective_pooler = None if pooler == "none" else pooler_factory(pooler)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        result = run_embedding_generation(
            generator,
            batcher,
            writer,
            layer_index=layer_index,
            pooler=effective_pooler,
            fail_fast=fail_fast,
            progress_callback=events.append,
        )
        elapsed = time.perf_counter() - started

    accepted_records = _filter_by_length(scheduled_records, length_limit)
    amino_acids = sum(len(record.sequence) for record in accepted_records)
    batch_completed = [event for event in events if event.get("event") == "batch_completed"]
    oom_skipped_count = _cuda_oom_error_count(result.errors)
    skipped_count = int(result.skipped_count) + oom_skipped_count
    _maybe_release_cuda_cache(device)
    return {
        "status": "ok",
        "elapsed_seconds": elapsed,
        "job_elapsed_seconds": result.elapsed_seconds,
        "input_count": len(records),
        "accepted_count": len(accepted_records),
        "record_count": result.record_count,
        "error_count": result.error_count,
        "length_skipped_count": result.skipped_count,
        "oom_skipped_count": oom_skipped_count,
        "skipped_count": skipped_count,
        "amino_acids": amino_acids,
        "records_per_second": result.record_count / elapsed if elapsed > 0 else 0.0,
        "amino_acids_per_second": amino_acids / elapsed if elapsed > 0 else 0.0,
        "batch_count": len(batch_completed),
        "mean_realized_batch_size": statistics.fmean(realized_batch_sizes) if realized_batch_sizes else 0.0,
        "min_realized_batch_size": min(realized_batch_sizes) if realized_batch_sizes else 0,
        "max_realized_batch_size": max(realized_batch_sizes) if realized_batch_sizes else 0,
        "paths": [str(path) for path in result.paths],
        "warnings": [str(item.message) for item in caught],
        "errors": result.errors,
        "skipped": result.skipped,
    }


def _benchmark_model(
    model_class: str,
    repeat_records: Sequence[Sequence[GenerationInput]],
    *,
    settings: Sequence[Dict[str, Any]],
    length_limits: Sequence[int | None],
    repeats: int,
    device: str,
    dtype: str | None,
    layer_index: int,
    pooler: str,
    fail_fast: bool,
    length_sort_window: int | None,
    progress_index: int,
    progress_total: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    _print_progress(f"[{progress_index}/{progress_total}] initializing model={model_class} device={device}")
    generator_kwargs: Dict[str, Any] = {"model_class": model_class, "device": device}
    if dtype is not None and str(model_class).strip().lower() not in {"esmc", "esm-c"}:
        generator_kwargs["dtype"] = dtype
    try:
        generator = Generator(**generator_kwargs)
    except Exception as exc:
        return {
            "status": "unavailable",
            "model_class": model_class,
            "error": str(exc),
            "total_seconds": time.perf_counter() - started,
            "settings": {},
        }

    payload: Dict[str, Any] = {
        "status": "ok",
        "model_class": model_class,
        "model_reference": getattr(generator, "model_reference", None),
        "model_metadata": _jsonable(getattr(generator, "model_metadata", None)),
        "settings": {},
    }

    for length_limit in length_limits:
        effective_length_limit = _resolve_effective_length_limit(generator, length_limit)
        length_key = _length_limit_key(length_limit, effective_length_limit)
        filtered_counts = [len(_filter_by_length(records, effective_length_limit)) for records in repeat_records]
        if not filtered_counts or max(filtered_counts) == 0:
            payload["settings"][length_key] = {
                "status": "skipped",
                "reason": "no records remain after length filtering",
                "requested_length_limit": length_limit,
                "effective_length_limit": effective_length_limit,
                "runs": {},
            }
            continue

        length_payload: Dict[str, Any] = {
            "status": "ok",
            "requested_length_limit": length_limit,
            "effective_length_limit": effective_length_limit,
            "mean_accepted_count": statistics.fmean(filtered_counts),
            "min_accepted_count": min(filtered_counts),
            "max_accepted_count": max(filtered_counts),
            "runs": {},
        }
        for setting in settings:
            setting_name = _setting_key(setting)
            _print_progress(
                f"[{progress_index}/{progress_total}] model={model_class} "
                f"length_limit={length_key} setting={setting_name}"
            )
            runs: List[Dict[str, Any]] = []
            for repeat_index, records in enumerate(repeat_records, start=1):
                try:
                    run = _run_once(
                        generator,
                        records,
                        setting=setting,
                        length_limit=effective_length_limit,
                        length_sort_window=length_sort_window,
                        layer_index=layer_index,
                        pooler=pooler,
                        fail_fast=fail_fast,
                        device=device,
                    )
                    run["repeat"] = repeat_index
                    _print_progress(
                        f"  repeat={repeat_index} elapsed={_format_seconds(float(run['elapsed_seconds']))} "
                        f"records/s={float(run['records_per_second']):.2f} "
                        f"{_format_skipped(run)}"
                    )
                except Exception as exc:
                    run = {"status": "error", "repeat": repeat_index, "error": str(exc)}
                    _print_progress(f"  repeat={repeat_index} error={exc}")
                    if fail_fast:
                        raise
                runs.append(run)
            length_payload["runs"][setting_name] = {
                "setting": setting,
                "summary": _summarize_runs(runs),
                "runs": runs,
            }
        payload["settings"][length_key] = length_payload

    payload["total_seconds"] = time.perf_counter() - started
    return payload


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "__dict__"):
        return {key: _jsonable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _print_report(payload: Dict[str, Any]) -> None:
    print("\nmodel\tlength_limit\tsetting\taccepted\tlength_skipped\toom_skipped\tbatch_mean\tbatch_max\trecords/s\tok_runs\tstatus")
    rows: List[tuple[float, str]] = []
    for model_name, model_payload in payload["models"].items():
        if model_payload.get("status") != "ok":
            rows.append(
                (
                    -1.0,
                        f"{model_name}\t-\t-\t-\t-\t-\t-\t-\t-\t0\t{model_payload.get('status')}: {model_payload.get('error')}",
                )
            )
            continue
        for length_key, length_payload in model_payload.get("settings", {}).items():
            if length_payload.get("status") != "ok":
                rows.append(
                    (
                        -1.0,
                        f"{model_name}\t{length_key}\t-\t0\t-\t-\t-\t-\t-\t0\t{length_payload.get('reason')}",
                    )
                )
                continue
            for setting_name, setting_payload in length_payload.get("runs", {}).items():
                summary = setting_payload.get("summary", {})
                ok_runs = int(summary.get("ok_runs", 0))
                if ok_runs == 0:
                    rows.append(
                        (
                            -1.0,
                            f"{model_name}\t{length_key}\t{setting_name}\t-\t-\t-\t-\t-\t-\t0\terror",
                        )
                    )
                    continue
                records_per_second = float(summary["mean_records_per_second"])
                rows.append(
                    (
                        records_per_second,
                        "\t".join(
                            [
                                model_name,
                                length_key,
                                setting_name,
                                f"{float(summary.get('mean_accepted_count', 0.0)):.1f}",
                                f"{float(summary.get('mean_length_skipped_count', 0.0)):.1f}",
                                f"{float(summary.get('mean_oom_skipped_count', 0.0)):.1f}",
                                f"{float(summary.get('mean_realized_batch_size', 0.0)):.2f}",
                                str(int(summary.get("max_realized_batch_size", 0))),
                                _format_mean_sd(
                                    records_per_second,
                                    float(summary.get("stdev_records_per_second", 0.0)),
                                ),
                                str(ok_runs),
                                "ok",
                            ]
                        ),
                    )
                )
    for _records_per_second, row in sorted(rows, key=lambda item: item[0], reverse=True):
        print(row)


def main() -> None:
    args = _parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    length_limits = _parse_length_limits(args.length_limits)
    batch_sizes = _parse_positive_int_values(args.batch_sizes, flag_name="--batch-sizes")
    token_budgets = _parse_positive_int_values(args.token_budgets, flag_name="--token-budgets")
    if not batch_sizes and not token_budgets:
        raise SystemExit("At least one batching dimension must be enabled")
    settings = _build_settings(batch_sizes, token_budgets, sort_modes=args.sort_modes)

    _print_progress(
        f"Loading sample count={args.count} repeats={args.repeats} "
        f"repeat_sampling={args.repeat_sampling} from {args.fasta}"
    )
    repeat_records = _load_repeat_samples(
        args.fasta,
        count=args.count,
        seed=args.seed,
        repeats=args.repeats,
        mode=args.repeat_sampling,
    )
    payload: Dict[str, Any] = {
        "fasta": str(args.fasta),
        "sample_count": args.count,
        "seed": int(args.seed),
        "repeat_sampling": args.repeat_sampling,
        "device": args.device,
        "dtype": args.dtype,
        "layer_index": int(args.layer_index),
        "pooler": args.pooler,
        "length_sort_window": args.length_sort_window,
        "sort_modes": args.sort_modes,
        "settings": settings,
        "length_limits": ["none" if value is None else value for value in length_limits],
        "models": {},
    }

    total = len(args.models)
    for index, model_class in enumerate(args.models, start=1):
        payload["models"][model_class] = _benchmark_model(
            model_class,
            repeat_records,
            settings=settings,
            length_limits=length_limits,
            repeats=args.repeats,
            device=args.device,
            dtype=args.dtype,
            layer_index=args.layer_index,
            pooler=args.pooler,
            fail_fast=args.fail_fast,
            length_sort_window=args.length_sort_window,
            progress_index=index,
            progress_total=total,
        )
        _maybe_release_cuda_cache(args.device)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_report(payload)
    print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
