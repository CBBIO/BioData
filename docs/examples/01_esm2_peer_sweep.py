# Prompt:
# Write a CLI script named sweep_esm2_peer.py. It should take --fasta, --data-dir,
# --out-dir, --device, and --max-batch-tokens. Use CBBIO.available_generator_models("esm2")
# to discover ESM-2 models, generate mean-pooled embeddings with CBBIO.Generator,
# CBBIO.FastaBatcher, CBBIO.EmbeddingWriter, and CBBIO.run_embedding_generation,
# then run all ready PEER protein-level datasets with CBBIO.LinearProbe. Sweep every
# generated layer. Save per-task metrics to peer_esm2_sweep.csv and save one HDF5
# embedding file per model. Include argparse, type annotations, logging, and graceful
# skips for datasets that cannot be downloaded.

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from CBBIO import (
    EmbeddingRecord,
    EmbeddingWriter,
    FastaBatcher,
    Generator,
    LinearProbe,
    PredictionSpec,
    Task,
    available_generator_models,
    list_dataset_catalog,
    load_dataset,
    load_embedding_records_h5,
    pooler_factory,
    run_embedding_generation,
    run_task_on_layer,
)


LOGGER = logging.getLogger("esm2_peer_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def records_by_layer(records: Sequence[EmbeddingRecord]) -> dict[int, dict[str, Sequence[float]]]:
    grouped: dict[int, dict[str, Sequence[float]]] = {}
    for record in records:
        if len(record.shape) != 1:
            continue
        grouped.setdefault(record.layer_index, {})[record.id] = record.embedding
    return grouped


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["model_name", "dataset_id", "layer_index", "metric", "score"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    datasets = list_dataset_catalog(collection="peer", level="protein", status="ready", has_loader=True)
    model_names = available_generator_models("esm2")
    if not isinstance(model_names, list):
        raise TypeError("available_generator_models('esm2') did not return a list.")

    for model_name in model_names:
        LOGGER.info("Generating embeddings for %s", model_name)
        h5_path = out_dir / f"{model_name.replace('/', '_')}.h5"
        generator = Generator(model_class="esm2", name=model_name, device=args.device)
        batcher = FastaBatcher(args.fasta, max_batch_tokens=args.max_batch_tokens, max_sequence_length=4000)
        writer = EmbeddingWriter(format="h5", path=h5_path)
        run_embedding_generation(generator, batcher, writer, layer_index=None, pooler=pooler_factory("mean"))

        by_layer = records_by_layer(load_embedding_records_h5(h5_path, pool_method="mean"))
        for dataset_meta in datasets:
            try:
                dataset = load_dataset(dataset_meta.id, args.data_dir, download=True)
            except Exception as exc:
                LOGGER.warning("Skipping %s: %s", dataset_meta.id, exc)
                continue

            task = Task(
                name=f"{dataset_meta.id}_linear",
                dataset=dataset,
                prediction=PredictionSpec(
                    target=dataset_meta.target,
                    objective=dataset_meta.objective,
                    level="protein",
                ),
                probe=LinearProbe(epochs=args.epochs, seed=7),
            )
            for layer_index, embeddings in by_layer.items():
                result = run_task_on_layer(
                    task=task,
                    embeddings=embeddings,
                    model_reference=model_name,
                    layer_index=layer_index,
                )
                for metric, score in result.metrics.items():
                    rows.append(
                        {
                            "model_name": model_name,
                            "dataset_id": dataset_meta.id,
                            "layer_index": layer_index,
                            "metric": metric,
                            "score": score,
                        }
                    )

    write_rows(out_dir / "peer_esm2_sweep.csv", rows)


if __name__ == "__main__":
    main()
