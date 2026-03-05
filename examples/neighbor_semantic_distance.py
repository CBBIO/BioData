#!/usr/bin/env python3
"""Plot nearest-neighbor embedding distance vs GO semantic distance."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

try:
    from CBBIO.BioData import BioDataClient, GOAnnotation, NotFoundError
    from CBBIO.GO import GOError, GOOntology, load_go
    from CBBIO.types import DistanceMetric, SimilarityMethod
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from CBBIO.BioData import BioDataClient, GOAnnotation, NotFoundError
    from CBBIO.GO import GOError, GOOntology, load_go
    from CBBIO.types import DistanceMetric, SimilarityMethod


@dataclass
class PairResult:
    protein_id: str
    neighbor_id: str
    embedding_distance: float
    semantic_distance: Optional[float] = None
    protein_terms: int = 0
    neighbor_terms: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each input protein, find nearest neighbor (excluding self), "
            "compute GO semantic distance, and plot against "
            "embedding distance."
        )
    )
    parser.add_argument(
        "--protein-id",
        action="append",
        default=[],
        help="Protein ID to include (repeatable).",
    )
    parser.add_argument(
        "--protein-ids-file",
        help="Text file with one protein ID per line (# comments supported).",
    )
    parser.add_argument(
        "--obo",
        required=True,
        help="Path to GO OBO file (for example go-basic.obo).",
    )
    parser.add_argument(
        "--embedding",
        default="protT5",
        help="Embedding type name in sequence_embedding_type.name.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Embedding layer index.",
    )
    parser.add_argument(
        "--metric",
        choices=["l2", "cosine", "inner_product"],
        default="cosine",
        help="Embedding distance metric.",
    )
    parser.add_argument(
        "--semantic-method",
        choices=["lin", "schlicker", "resnik"],
        default="lin",
        help="GO term semantic similarity method.",
    )
    parser.add_argument(
        "--plot",
        default="neighbor_semantic_vs_embedding.png",
        help="Output image path.",
    )
    parser.add_argument("--csv", help="Optional CSV output path.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive plot window.",
    )
    args = parser.parse_args()

    if not args.protein_id and not args.protein_ids_file:
        parser.error("Provide --protein-id and/or --protein-ids-file")

    return args


def _load_protein_ids(
    inline_ids: Sequence[str],
    ids_file: Optional[str],
) -> List[str]:
    values: List[str] = [
        value.strip() for value in inline_ids if value.strip()
    ]

    if ids_file:
        file_path = Path(ids_file)
        with file_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                values.append(line)

    deduped: List[str] = []
    seen: Set[str] = set()
    for protein_id in values:
        if protein_id in seen:
            continue
        seen.add(protein_id)
        deduped.append(protein_id)

    return deduped


def _to_go_set_map(
    annotations: Dict[str, List[GOAnnotation]],
) -> Dict[str, Set[str]]:
    rows: Dict[str, Set[str]] = {}
    for protein_id, values in annotations.items():
        rows[protein_id] = {value.go_id for value in values}
    return rows


def _pairwise_similarity_bma(
    go: GOOntology,
    terms_a: Sequence[str],
    terms_b: Sequence[str],
    method: SimilarityMethod,
) -> Optional[float]:
    if not terms_a or not terms_b:
        return None

    scores_a: List[float] = []
    for go_id_a in terms_a:
        row_scores: List[float] = []
        for go_id_b in terms_b:
            row_scores.append(
                float(
                    go.semantic_similarity(
                        go_id_a,
                        go_id_b,
                        method=method,
                    )
                )
            )
        if row_scores:
            scores_a.append(max(row_scores))

    scores_b: List[float] = []
    for go_id_b in terms_b:
        row_scores = []
        for go_id_a in terms_a:
            row_scores.append(
                float(
                    go.semantic_similarity(
                        go_id_b,
                        go_id_a,
                        method=method,
                    )
                )
            )
        if row_scores:
            scores_b.append(max(row_scores))

    if not scores_a or not scores_b:
        return None

    avg_a = sum(scores_a) / float(len(scores_a))
    avg_b = sum(scores_b) / float(len(scores_b))
    return (avg_a + avg_b) / 2.0


def _similarity_to_distance(
    similarity: float,
    method: SimilarityMethod,
) -> float:
    if method in {"lin", "schlicker"}:
        bounded = min(1.0, max(0.0, similarity))
        return 1.0 - bounded

    non_negative = max(0.0, similarity)
    return 1.0 / (1.0 + non_negative)


def _plot_results(
    results: Sequence[PairResult],
    output_path: str,
    metric: DistanceMetric,
    semantic_method: SimilarityMethod,
    show_plot: bool,
) -> int:
    valid = [
        row
        for row in results
        if row.semantic_distance is not None
        and math.isfinite(row.semantic_distance)
    ]
    if not valid:
        print("No valid points with GO semantic distance available.")
        return 0

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: "
            "poetry add matplotlib"
        ) from exc

    x_values = [row.embedding_distance for row in valid]
    y_values = [
        row.semantic_distance
        for row in valid
        if row.semantic_distance is not None
    ]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        x_values,
        y_values,
        alpha=0.8,
        s=30,
        edgecolors="black",
        linewidths=0.4,
    )
    plt.xlabel(f"Embedding distance ({metric})")
    plt.ylabel(f"GO semantic distance ({semantic_method})")
    plt.title("Nearest-neighbor embedding vs GO semantic distance")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    plt.close()

    print(f"Saved plot: {output_path}")
    return len(valid)


def _write_csv(path: str, rows: Sequence[PairResult]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protein_id",
                "neighbor_id",
                "embedding_distance",
                "semantic_distance",
                "protein_terms",
                "neighbor_terms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.protein_id,
                    row.neighbor_id,
                    row.embedding_distance,
                    (
                        ""
                        if row.semantic_distance is None
                        else row.semantic_distance
                    ),
                    row.protein_terms,
                    row.neighbor_terms,
                ]
            )


def main() -> int:
    args = parse_args()
    protein_ids = _load_protein_ids(args.protein_id, args.protein_ids_file)
    if not protein_ids:
        print("No protein IDs found.", file=sys.stderr)
        return 2

    with BioDataClient() as db:
        embedding_type = db.get_embedding_type_by_name(args.embedding)
        if embedding_type is None:
            print(
                f"Embedding type not found: {args.embedding}",
                file=sys.stderr,
            )
            return 2

        results: List[PairResult] = []
        neighbor_ids: Set[str] = set()

        for protein_id in protein_ids:
            query_embedding = db.get_protein_embedding(
                protein_id,
                embedding_type.id,
                layer_index=args.layer,
                as_numpy=False,
            )
            if query_embedding is None:
                print(
                    f"Skipping {protein_id}: missing embedding "
                    f"(type={embedding_type.id}, layer={args.layer}).",
                    file=sys.stderr,
                )
                continue

            neighbors = db.find_nearest_neighbors(
                query_embedding,
                embedding_type.id,
                layer_index=args.layer,
                k=1,
                metric=args.metric,
                exclude_protein_ids=[protein_id],
            )
            if not neighbors:
                print(
                    f"Skipping {protein_id}: no neighbor found.",
                    file=sys.stderr,
                )
                continue

            nearest = neighbors[0]
            results.append(
                PairResult(
                    protein_id=protein_id,
                    neighbor_id=nearest.protein_id,
                    embedding_distance=nearest.distance,
                )
            )
            neighbor_ids.add(nearest.protein_id)

        all_ids = sorted(set(protein_ids).union(neighbor_ids))
        annotations = db.fetch_go_annotations(all_ids)

    if not results:
        print("No protein-neighbor pairs were generated.")
        return 0

    go = load_go(args.obo)
    go_sets = _to_go_set_map(annotations)

    counts_input: Dict[str, Set[str]] = {}
    for protein_id, go_ids in go_sets.items():
        filtered = {go_id for go_id in go_ids if go.has_term(go_id)}
        if filtered:
            counts_input[protein_id] = filtered

    if not counts_input:
        print("No valid GO terms found for semantic calculations.")
        return 0

    go.prepare_term_counts(counts_input)

    for row in results:
        source_terms = sorted(counts_input.get(row.protein_id, set()))
        neighbor_terms = sorted(counts_input.get(row.neighbor_id, set()))
        row.protein_terms = len(source_terms)
        row.neighbor_terms = len(neighbor_terms)

        if not source_terms or not neighbor_terms:
            continue

        similarity = _pairwise_similarity_bma(
            go,
            source_terms,
            neighbor_terms,
            method=args.semantic_method,
        )
        if similarity is None:
            continue
        row.semantic_distance = _similarity_to_distance(
            similarity,
            args.semantic_method,
        )

    if args.csv:
        _write_csv(args.csv, results)
        print(f"Saved table: {args.csv}")

    valid_points = _plot_results(
        results,
        output_path=args.plot,
        metric=args.metric,
        semantic_method=args.semantic_method,
        show_plot=args.show,
    )
    print(
        f"Processed {len(results)} pairs; "
        f"plotted {valid_points} points with valid GO distance."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except NotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except GOError as exc:
        print(f"GO error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
