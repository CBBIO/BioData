"""Profile GO and EC metrics on real splits with cached protein embeddings.

The benchmark transfers labels between real proteins selected from the requested
datasets. Scores come from cosine-nearest neighbours in a precomputed embedding
cache, so ontology sizes, label sparsity, score ties, and class imbalance match
the production workloads.

Run with:
    poetry run python scripts/benchmark_function_metrics.py \
        --data-root notebooks/data/probing \
        --embedding-cache /tmp/biodata_embedding_cache/go_bp_full_esm2_8m_train2048_test1024_maxlen384_layers6.pkl
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import pickle
import pstats
import resource
import time
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np

from CBBIO import (
    evaluate_go_metric_context,
    load_cafa5_dataset,
    load_clean_dataset,
    load_ec_dataset,
    load_go,
    load_go_dataset,
    multilabel_metrics,
    prepare_go_metric_context,
    read_information_accretion_weights,
)


@dataclass(frozen=True)
class _BenchmarkCase:
    name: str
    target: str
    loader: Callable[..., Any]
    loader_name: str
    metric_kind: str
    threshold_count: int = 101


_CASES = (
    *(
        _BenchmarkCase(
            name=f"go_{aspect}_full",
            target=f"go_{aspect}",
            loader=load_go_dataset,
            loader_name=f"go_{aspect}_full",
            metric_kind="go",
        )
        for aspect in ("bp", "cc", "mf")
    ),
    *(
        _BenchmarkCase(
            name=f"ec_{level}_full",
            target=f"ec_{level}",
            loader=load_ec_dataset,
            loader_name=f"ec_{level}_full",
            metric_kind="ec",
        )
        for level in (1, 2, 3, 4)
    ),
    *(
        _BenchmarkCase(
            name=f"clean_ec_{level}_split50_fold0",
            target=f"ec_{level}",
            loader=load_clean_dataset,
            loader_name=f"ec_{level}_split50_fold0",
            metric_kind="ec",
        )
        for level in (1, 2, 3, 4)
    ),
    *(
        _BenchmarkCase(
            name=f"cafa5_no_knowledge_{aspect}",
            target=f"go_{aspect}",
            loader=load_cafa5_dataset,
            loader_name=f"cafa5_no_knowledge_{aspect}",
            metric_kind="cafa",
            threshold_count=1001,
        )
        for aspect in ("bp", "cc", "mf")
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("notebooks/data/probing"))
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=[case.name for case in _CASES])
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _load_embedding_cache(path: Path) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
    with path.expanduser().open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - explicitly supplied local benchmark cache
    if not isinstance(payload, Mapping):
        raise ValueError("Embedding cache must contain a mapping or a layer-to-mapping payload.")
    if payload and all(isinstance(key, int) for key in payload):
        payload = payload[max(payload)]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("Embedding cache contains no embeddings.")
    embeddings: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    for protein_id, vector in payload.items():
        embeddings[_canonical_id(str(protein_id))] = np.asarray(vector, dtype=np.float32)
    return embeddings


def _canonical_id(protein_id: str) -> str:
    return protein_id.rsplit(":", maxsplit=1)[-1].removeprefix("UniRef50_")


def _embedded_examples(
    examples: Sequence[Any],
    embeddings: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]],
) -> list[Any]:
    return [example for example in examples if _canonical_id(example.id) in embeddings]


def _class_names(dataset: Any, target: str) -> list[str]:
    return sorted(
        {
            str(label)
            for split in ("train", "test")
            for example in dataset.by_split(split)
            for label in example.labels[target]
        }
    )


def _transfer_scores(
    *,
    train_examples: Sequence[Any],
    test_examples: Sequence[Any],
    embeddings: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]],
    class_names: Sequence[str],
    target: str,
    neighbors: int,
) -> dict[str, np.ndarray[Any, np.dtype[np.float32]]]:
    class_indices = {name: index for index, name in enumerate(class_names)}
    train_matrix = np.asarray(
        [embeddings[_canonical_id(example.id)] for example in train_examples],
        dtype=np.float32,
    )
    test_matrix = np.asarray(
        [embeddings[_canonical_id(example.id)] for example in test_examples],
        dtype=np.float32,
    )
    train_matrix /= np.maximum(np.linalg.norm(train_matrix, axis=1, keepdims=True), 1e-12)
    test_matrix /= np.maximum(np.linalg.norm(test_matrix, axis=1, keepdims=True), 1e-12)
    similarities = test_matrix @ train_matrix.T
    neighbor_count = min(int(neighbors), len(train_examples))
    nearest = np.argpartition(similarities, -neighbor_count, axis=1)[:, -neighbor_count:]
    scores: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {}
    for row_index, example in enumerate(test_examples):
        row = np.zeros(len(class_names), dtype=np.float32)
        neighbor_indices = nearest[row_index]
        weights = np.maximum(similarities[row_index, neighbor_indices], 0.0)
        denominator = float(np.sum(weights))
        if denominator <= 0.0:
            weights = np.ones(neighbor_count, dtype=np.float32)
            denominator = float(neighbor_count)
        for neighbor_index, weight in zip(neighbor_indices, weights):
            for label in train_examples[int(neighbor_index)].labels[target]:
                row[class_indices[str(label)]] += float(weight) / denominator
        scores[example.id] = row
    return scores


def _ec_metric_call(
    *,
    test_examples: Sequence[Any],
    scores: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]],
    class_names: Sequence[str],
    target: str,
) -> Callable[[], dict[str, float]]:
    class_indices = {name: index for index, name in enumerate(class_names)}
    true_matrix = np.zeros((len(test_examples), len(class_names)), dtype=np.int8)
    score_matrix = np.asarray([scores[example.id] for example in test_examples], dtype=np.float32)
    for row_index, example in enumerate(test_examples):
        for label in example.labels[target]:
            true_matrix[row_index, class_indices[str(label)]] = 1
    pred_matrix = (score_matrix >= 0.5).astype(np.int8, copy=False)

    def evaluate() -> dict[str, float]:
        return multilabel_metrics(
            cast(Sequence[Sequence[int]], true_matrix),
            cast(Sequence[Sequence[int]], pred_matrix),
            cast(Sequence[Sequence[float]], score_matrix),
        )

    return evaluate


def _go_metric_call(
    *,
    case: _BenchmarkCase,
    test_examples: Sequence[Any],
    scores: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]],
    class_names: Sequence[str],
) -> Callable[[], dict[str, float]]:
    metadata = test_examples[0].metadata or {}
    ontology_path = Path(str(metadata["go_obo_path"]))
    term_weights = (
        read_information_accretion_weights(Path(str(metadata["ia_path"])))
        if case.metric_kind == "cafa"
        else None
    )
    terms_of_interest = None
    terms_of_interest_path = metadata.get("toi_path")
    if isinstance(terms_of_interest_path, str) and terms_of_interest_path:
        terms_of_interest = [
            line.strip().split()[0]
            for line in Path(terms_of_interest_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    truth = {
        example.id: [str(term) for term in (example.metadata or {})["labels_propagated"]]
        for example in test_examples
    }
    context = prepare_go_metric_context(
        class_names=class_names,
        true_labels=truth,
        ontology=load_go(ontology_path),
        term_weights=term_weights,
        terms_of_interest=terms_of_interest,
        threshold_count=case.threshold_count,
    )

    def evaluate() -> dict[str, float]:
        return evaluate_go_metric_context(scores, context=context)

    return evaluate


def _benchmark_case(
    case: _BenchmarkCase,
    *,
    data_root: Path,
    embeddings: Mapping[str, np.ndarray[Any, np.dtype[np.float32]]],
    neighbors: int,
    repeat: int,
    profile: bool,
) -> dict[str, object]:
    dataset = case.loader(data_root, name=case.loader_name)
    train_examples = _embedded_examples(dataset.by_split("train"), embeddings)
    test_examples = _embedded_examples(dataset.by_split("test"), embeddings)
    if not train_examples or not test_examples:
        raise ValueError(f"{case.name} has no cached train/test embedding overlap.")
    class_names = _class_names(dataset, case.target)
    scores = _transfer_scores(
        train_examples=train_examples,
        test_examples=test_examples,
        embeddings=embeddings,
        class_names=class_names,
        target=case.target,
        neighbors=neighbors,
    )
    evaluate = (
        _ec_metric_call(
            test_examples=test_examples,
            scores=scores,
            class_names=class_names,
            target=case.target,
        )
        if case.metric_kind == "ec"
        else _go_metric_call(
            case=case,
            test_examples=test_examples,
            scores=scores,
            class_names=class_names,
        )
    )
    evaluate()
    elapsed: list[float] = []
    result: dict[str, float] = {}
    profiler = cProfile.Profile() if profile else None
    for index in range(max(1, int(repeat))):
        if profiler is not None and index == 0:
            profiler.enable()
        started = time.perf_counter()
        result = evaluate()
        elapsed.append(time.perf_counter() - started)
        if profiler is not None and index == 0:
            profiler.disable()
    if profiler is not None:
        pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(20)
    return {
        "case": case.name,
        "train_proteins": len(train_examples),
        "test_proteins": len(test_examples),
        "classes": len(class_names),
        "score_values": len(test_examples) * len(class_names),
        "median_seconds": float(np.median(elapsed)),
        "min_seconds": min(elapsed),
        "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "metric": "go_weighted_fmax" if case.metric_kind == "cafa" else (
            "go_fmax" if case.metric_kind == "go" else "fmax"
        ),
        "metric_value": result[
            "go_weighted_fmax" if case.metric_kind == "cafa" else (
                "go_fmax" if case.metric_kind == "go" else "fmax"
            )
        ],
    }


def main() -> None:
    """Run selected real-data function-metric benchmarks."""
    args = _parse_args()
    embeddings = _load_embedding_cache(args.embedding_cache)
    selected = set(args.case or [case.name for case in _CASES])
    results: list[dict[str, object]] = []
    for case in _CASES:
        if case.name not in selected:
            continue
        result = _benchmark_case(
            case,
            data_root=args.data_root,
            embeddings=embeddings,
            neighbors=args.neighbors,
            repeat=args.repeat,
            profile=args.profile,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        gc.collect()
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
