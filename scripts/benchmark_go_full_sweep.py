"""Generate cached ESM2 embeddings and benchmark a full GO layer sweep.

Generate the shared cache:
    poetry run python scripts/benchmark_go_full_sweep.py generate \
        --data-root notebooks/data/probing --output-dir /tmp/go_bp_full_esm2_35m

Evaluate the cache:
    poetry run python scripts/benchmark_go_full_sweep.py evaluate \
        --data-root notebooks/data/probing --output-dir /tmp/go_bp_full_esm2_35m \
        --run-name after
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import resource
import time
from typing import Any, Mapping, Sequence, cast

from CBBIO import (
    EmbeddingWriter,
    GenerationInput,
    Generator,
    IterableBatcher,
    PredictionSpec,
    Task,
    TransferProbe,
    load_embedding_records_h5,
    load_go_dataset,
    pooler_factory,
    run_embedding_generation,
    run_task_on_layer,
)


_MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
_DATASET_NAME = "go_bp_full"
_TARGET = "go_bp"


@dataclass(frozen=True)
class _GenerationSummary:
    model_name: str
    dataset_name: str
    layers: tuple[int, ...]
    protein_count: int
    truncated_count: int
    max_sequence_length: int
    record_count: int
    error_count: int
    elapsed_seconds: float
    cache_path: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "evaluate"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="after")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--max-sequence-length", type=int, default=1022)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--length-sort-window", type=int, default=4096)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument(
        "--disable-profile",
        action="store_true",
        help="Measure wall time without cProfile instrumentation.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cache_path(output_dir: Path) -> Path:
    return output_dir / "go_bp_full_esm2_35m_layers_0_4_8_12_mean.h5"


def _generation_inputs(dataset: Any, *, max_sequence_length: int) -> list[GenerationInput]:
    return [
        GenerationInput(
            id=example.id,
            sequence=example.sequence[:max_sequence_length],
        )
        for example in dataset.examples
    ]


def _generate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(output_dir)
    dataset = load_go_dataset(args.data_root, name=_DATASET_NAME)
    layers = tuple(sorted(set(int(layer) for layer in args.layers)))
    truncated_count = sum(
        len(example.sequence) > int(args.max_sequence_length)
        for example in dataset.examples
    )
    inputs = _generation_inputs(
        dataset,
        max_sequence_length=int(args.max_sequence_length),
    )
    generator = Generator(
        model_class="esm2",
        name=_MODEL_NAME,
        device=args.device,
    )
    batcher = IterableBatcher(
        inputs,
        max_batch_tokens=int(args.max_batch_tokens),
        length_sort_window=int(args.length_sort_window),
        max_sequence_length=int(args.max_sequence_length),
    )
    writer = EmbeddingWriter(
        format="h5",
        path=cache_path,
        compression=None,
        write_batch_size=512,
        flush_interval=100,
    )
    last_reported = 0

    def _progress(event: dict[str, Any]) -> None:
        nonlocal last_reported
        if event.get("event") != "batch_completed":
            print(json.dumps(event, sort_keys=True), flush=True)
            return
        written = int(event.get("written_count", 0))
        if written - last_reported >= 10_000:
            last_reported = written
            print(json.dumps(event, sort_keys=True), flush=True)

    result = run_embedding_generation(
        generator,
        batcher,
        writer,
        layer_index=layers,
        pooler=pooler_factory("mean"),
        fail_fast=True,
        progress_callback=_progress,
    )
    expected_records = len(inputs) * len(layers)
    if result.record_count != expected_records or result.error_count:
        raise RuntimeError(
            f"Embedding cache is incomplete: expected {expected_records} records, "
            f"got {result.record_count} with {result.error_count} errors."
        )
    summary = _GenerationSummary(
        model_name=_MODEL_NAME,
        dataset_name=_DATASET_NAME,
        layers=layers,
        protein_count=len(inputs),
        truncated_count=truncated_count,
        max_sequence_length=int(args.max_sequence_length),
        record_count=result.record_count,
        error_count=result.error_count,
        elapsed_seconds=result.elapsed_seconds,
        cache_path=str(cache_path),
    )
    _write_json(output_dir / "generation.json", asdict(summary))
    print(json.dumps(asdict(summary), sort_keys=True), flush=True)


def _embeddings_for_layer(cache_path: Path, layer_index: int) -> dict[str, Sequence[float]]:
    records = load_embedding_records_h5(
        cache_path,
        layer_index=layer_index,
        pool_method="mean",
    )
    return {
        record.id: cast(Sequence[float], record.embedding)
        for record in records
    }


def _evaluate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser()
    cache_path = _cache_path(output_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}")
    dataset = load_go_dataset(args.data_root, name=_DATASET_NAME)
    task = Task(
        name=f"{_DATASET_NAME}_esm2_35m_transfer",
        dataset=dataset,
        prediction=PredictionSpec(
            target=_TARGET,
            objective="multilabel",
            level="protein",
        ),
        probe=TransferProbe(
            k=int(args.neighbors),
            neighbor_selection="knn",
            scoring="weighted_voting",
            search_backend="torch_gpu",
            search_device=args.device,
        ),
    )
    run_dir = output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for layer_index in sorted(set(int(layer) for layer in args.layers)):
        embeddings = _embeddings_for_layer(cache_path, layer_index)
        if len(embeddings) != len(dataset.examples):
            raise RuntimeError(
                f"Layer {layer_index} has {len(embeddings)} embeddings for "
                f"{len(dataset.examples)} dataset proteins."
            )
        profiler = None if args.disable_profile else cProfile.Profile()
        if profiler is not None:
            profiler.enable()
        started = time.perf_counter()
        result = run_task_on_layer(
            task=task,
            embeddings=embeddings,
            model_reference=_MODEL_NAME,
            layer_index=layer_index,
        )
        wall_seconds = time.perf_counter() - started
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(run_dir / f"layer_{layer_index}.prof")
        row: dict[str, Any] = {
            "run_name": args.run_name,
            "model_name": _MODEL_NAME,
            "dataset_name": _DATASET_NAME,
            "layer_index": layer_index,
            "train_count": result.train_count,
            "val_count": result.val_count,
            "test_count": result.test_count,
            "wall_seconds": wall_seconds,
            "reported_elapsed_seconds": result.elapsed_seconds,
            "profile_enabled": profiler is not None,
            "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "metrics": result.metrics,
        }
        rows.append(row)
        _write_json(run_dir / f"layer_{layer_index}.json", row)
        print(json.dumps(row, sort_keys=True), flush=True)
    _write_json(
        run_dir / "summary.json",
        {
            "run_name": args.run_name,
            "model_name": _MODEL_NAME,
            "dataset_name": _DATASET_NAME,
            "layers": rows,
            "total_wall_seconds": sum(float(row["wall_seconds"]) for row in rows),
        },
    )


def main() -> None:
    """Generate embeddings or evaluate the full GO layer sweep."""
    args = _parse_args()
    if args.command == "generate":
        _generate(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    main()
