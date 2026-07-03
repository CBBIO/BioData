# Prompt:
# Build a reproducible CAFA baseline pipeline. Load CAFA5 training-only and released
# target-subset datasets with CBBIO.load_dataset, generate protein-level embeddings
# for ESM-C and AMPLIFY, train no model with CBBIO.TransferProbe, and evaluate GO
# protein-centric metrics including weighted Fmax when information accretion weights
# are available. The script should compare search backends "numpy", "faiss_cpu",
# and "torch_gpu" when installed, skip missing optional dependencies with clear
# messages, and write one JSON report per dataset, model, layer, and backend.

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from CBBIO import (
    EmbeddingWriter,
    Generator,
    IterableBatcher,
    PredictionSpec,
    Task,
    TransferProbe,
    GenerationInput,
    list_dataset_catalog,
    load_dataset,
    pooler_factory,
    run_embedding_generation,
    run_task_on_layer,
)


LOGGER = logging.getLogger("cafa_transfer_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", default=[0])
    return parser.parse_args()


def dataset_inputs(dataset: Any) -> list[GenerationInput]:
    return [GenerationInput(id=example.id, sequence=example.sequence) for example in dataset.examples]


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        item
        for item in list_dataset_catalog(level="protein", objective="multilabel", status="ready", has_loader=True)
        if item.collection == "cafa"
    ]
    model_specs = [("esmc", "esmc_300m"), ("amplify", "nvidia/AMPLIFY_120M")]
    backends = ["numpy", "faiss_cpu", "torch_gpu"]

    for dataset_meta in datasets:
        try:
            dataset = load_dataset(dataset_meta.id, args.data_dir, download=True)
        except Exception as exc:
            LOGGER.warning("Skipping dataset %s: %s", dataset_meta.id, exc)
            continue

        for model_class, model_name in model_specs:
            writer = EmbeddingWriter(format="memory")
            try:
                run_embedding_generation(
                    Generator(model_class=model_class, name=model_name, device=args.device),
                    IterableBatcher(dataset_inputs(dataset), batch_size=8),
                    writer,
                    layer_index=args.layers,
                    pooler=pooler_factory("mean"),
                    fail_fast=False,
                )
            except Exception as exc:
                LOGGER.warning("Skipping model %s: %s", model_name, exc)
                continue

            for layer_index in args.layers:
                embeddings = {record.id: record.embedding for record in writer.records if record.layer_index == layer_index}
                for backend in backends:
                    task = Task(
                        name=f"{dataset_meta.id}_{model_class}_{backend}",
                        dataset=dataset,
                        prediction=PredictionSpec(
                            target=dataset_meta.target,
                            objective=dataset_meta.objective,
                            level="protein",
                            metrics=dataset_meta.metrics,
                        ),
                        probe=TransferProbe(search_backend=backend),
                    )
                    try:
                        result = run_task_on_layer(
                            task=task,
                            embeddings=embeddings,
                            model_reference=model_name,
                            layer_index=layer_index,
                        )
                    except Exception as exc:
                        LOGGER.warning("Skipping backend %s for %s: %s", backend, dataset_meta.id, exc)
                        continue
                    report_path = out_dir / f"{dataset_meta.id.replace(':', '_')}_{model_class}_L{layer_index}_{backend}.json"
                    write_report(report_path, {"metrics": result.metrics, "metadata": dict(result.metadata or {})})


if __name__ == "__main__":
    main()
