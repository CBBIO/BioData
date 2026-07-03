# Prompt:
# Create a residue-level probing benchmark for phosphorylation. Load dbPTM,
# PhosphoELM, and MusiteDeep residue datasets through CBBIO catalog or source loaders,
# generate unpooled residue embeddings with ProtT5, Ankh3, and ESM-2, and train both
# CBBIO.LinearProbe and CBBIO.MlpProbe for every requested layer. Validate that
# embedding matrix lengths match sequence lengths before training. Save metrics,
# confusion counts, and skipped protein ids. Add focused pytest tests for the
# length-validation helper.

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from CBBIO import (
    EmbeddingWriter,
    Generator,
    GenerationInput,
    IterableBatcher,
    LinearProbe,
    MlpProbe,
    PredictionSpec,
    ResidueDataset,
    Task,
    list_dataset_catalog,
    load_dataset,
    run_embedding_generation,
    run_task_on_layer,
)


LOGGER = logging.getLogger("residue_ptm_probe_matrix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def validate_residue_lengths(dataset: ResidueDataset, embeddings: Mapping[str, Sequence[Sequence[float]]]) -> list[str]:
    skipped: list[str] = []
    for example in dataset.examples:
        matrix = embeddings.get(example.id)
        if matrix is None or len(matrix) != len(example.sequence):
            skipped.append(example.id)
    return skipped


def write_metric_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "model_class", "probe", "layer_index", "metric", "score"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        item
        for item in list_dataset_catalog(level="residue", objective="binary", status="ready", has_loader=True)
        if "phosph" in item.target or "ptm" in item.tags or item.collection in {"dbptm", "phosphoelm", "musitedeep"}
    ]
    model_classes = ["prott5", "ankh3", "esm2"]
    rows: list[dict[str, Any]] = []

    for dataset_meta in datasets:
        try:
            dataset = load_dataset(dataset_meta.id, args.data_dir, download=True)
        except Exception as exc:
            LOGGER.warning("Skipping dataset %s: %s", dataset_meta.id, exc)
            continue
        inputs = [GenerationInput(id=example.id, sequence=example.sequence) for example in dataset.examples]

        for model_class in model_classes:
            writer = EmbeddingWriter(format="memory")
            try:
                run_embedding_generation(
                    Generator(model_class=model_class, device=args.device),
                    IterableBatcher(inputs, batch_size=4),
                    writer,
                    layer_index=args.layers,
                    pooler=None,
                    fail_fast=False,
                )
            except Exception as exc:
                LOGGER.warning("Skipping model %s: %s", model_class, exc)
                continue

            for layer_index in args.layers:
                embeddings = {record.id: record.embedding for record in writer.records if record.layer_index == layer_index}
                skipped = validate_residue_lengths(dataset, embeddings)
                if skipped:
                    (out_dir / f"{dataset_meta.id.replace(':', '_')}_{model_class}_L{layer_index}_skipped.txt").write_text("\n".join(skipped))
                valid_embeddings = {key: value for key, value in embeddings.items() if key not in set(skipped)}
                for probe_name, probe in {
                    "linear": LinearProbe(epochs=args.epochs),
                    "mlp": MlpProbe(epochs=args.epochs, hidden_dim=128),
                }.items():
                    task = Task(
                        name=f"{dataset_meta.id}_{model_class}_{probe_name}",
                        dataset=dataset,
                        prediction=PredictionSpec(target=dataset_meta.target, objective="binary", level="residue"),
                        probe=probe,
                    )
                    result = run_task_on_layer(task=task, embeddings=valid_embeddings, model_reference=model_class, layer_index=layer_index)
                    for metric, score in result.metrics.items():
                        rows.append({"dataset_id": dataset_meta.id, "model_class": model_class, "probe": probe_name, "layer_index": layer_index, "metric": metric, "score": score})

    write_metric_rows(out_dir / "residue_ptm_probe_matrix.csv", rows)


if __name__ == "__main__":
    main()
