#!/usr/bin/env python3
"""Plot nearest-neighbor embedding distance vs GO semantic distance."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

try:
    from CBBIO.BioData import BioDataClient, NotFoundError
    from CBBIO.GO import GOError, GOOntology, load_go
    from CBBIO.types import DistanceMetric, SimilarityMethod
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from CBBIO.BioData import BioDataClient, NotFoundError
    from CBBIO.GO import GOError, GOOntology, load_go
    from CBBIO.types import DistanceMetric, SimilarityMethod


@dataclass
class PairResult:
    protein_id: str
    neighbor_id: str
    embedding_distance: float
    protein_species: str = ""
    neighbor_species: str = ""
    semantic_similarity: Optional[float] = None
    protein_terms: str = ""
    neighbor_terms: str = ""


GO_CATEGORIES: Sequence[Tuple[str, str]] = (
    ("mf", "MF"),
    ("bp", "BP"),
    ("cc", "CC"),
)


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
        "--k",
        type=int,
        default=1,
        help="Number of nearest neighbors per query protein.",
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
    parser.add_argument(
        "--use-ann",
        action="store_true",
        help="Use ANN/index-friendly nearest-neighbor query path.",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print step-by-step timing diagnostics.",
    )
    args = parser.parse_args()

    if not args.protein_id and not args.protein_ids_file:
        parser.error("Provide --protein-id and/or --protein-ids-file")
    if args.k < 1:
        parser.error("--k must be >= 1")

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


def _distance_to_score(distance: float) -> float:
    return 1.0 - distance


def _linear_fit_with_r2(
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> Optional[Tuple[float, float, float]]:
    if len(x_values) < 2 or len(y_values) < 2:
        return None

    x_mean = sum(x_values) / float(len(x_values))
    y_mean = sum(y_values) / float(len(y_values))
    ss_xx = sum((x - x_mean) ** 2 for x in x_values)
    if ss_xx == 0.0:
        return None

    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_hat = [slope * x + intercept for x in x_values]
    ss_res = sum((y - yh) ** 2 for y, yh in zip(y_values, y_hat))
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0

    return slope, intercept, r2


def _plot_results(
    results: Sequence[PairResult],
    output_path: str,
    metric: DistanceMetric,
    semantic_method: SimilarityMethod,
    category_label: str,
    show_plot: bool,
) -> int:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: "
            "poetry add matplotlib"
        ) from exc

    valid = [
        row
        for row in results
        if row.semantic_similarity is not None
        and math.isfinite(row.semantic_similarity)
    ]
    if not valid:
        plt.figure(figsize=(8, 6))
        plt.xlabel(f"Embedding score (1-d, {metric})")
        plt.ylabel(f"GO semantic similarity ({semantic_method})")
        plt.title(
            f"Nearest-neighbor embedding score vs GO semantic similarity ({category_label})"
        )
        plt.text(
            0.5,
            0.5,
            "No valid points",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        if show_plot:
            plt.show()
        plt.close()

        print(
            f"No valid points with GO semantic similarity available for "
            f"{category_label}. Saved empty plot: {output_path}"
        )
        return 0

    x_values = [_distance_to_score(row.embedding_distance) for row in valid]
    y_values = [
        row.semantic_similarity
        for row in valid
        if row.semantic_similarity is not None
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

    fit = _linear_fit_with_r2(x_values, y_values)
    if fit is not None:
        slope, intercept, r2 = fit
        x_min = min(x_values)
        x_max = max(x_values)
        line_x = [x_min, x_max]
        line_y = [slope * x + intercept for x in line_x]
        plt.plot(
            line_x,
            line_y,
            color="crimson",
            linewidth=1.6,
            label=f"Linear fit (R^2={r2:.3f})",
        )
        plt.legend()

    plt.xlabel(f"Embedding score (1-d, {metric})")
    plt.ylabel(f"GO semantic similarity ({semantic_method})")
    plt.title(
        f"Nearest-neighbor embedding score vs GO semantic similarity ({category_label})"
    )
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
                "protein_species",
                "neighbor_id",
                "neighbor_species",
                "embedding_distance",
                "semantic_similarity",
                "protein_go_terms",
                "neighbor_go_terms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.protein_id,
                    row.protein_species,
                    row.neighbor_id,
                    row.neighbor_species,
                    row.embedding_distance,
                    (
                        ""
                        if row.semantic_similarity is None
                        else row.semantic_similarity
                    ),
                    row.protein_terms,
                    row.neighbor_terms,
                ]
            )


def _with_suffix(path: Optional[str], suffix: str) -> Optional[str]:
    if path is None:
        return None
    base = Path(path)
    return str(base.with_name(f"{base.stem}_{suffix}{base.suffix}"))


def _log_timing(enabled: bool, label: str, started_at: float) -> None:
    if not enabled:
        return
    elapsed = time.perf_counter() - started_at
    print(f"[timing] {label}: {elapsed:.3f}s")


def _fetch_species_map(db: BioDataClient, protein_ids: Sequence[str]) -> Dict[str, str]:
    ids = [str(value) for value in protein_ids]
    if not ids:
        return {}
    rows = db.query_all(
        """
        SELECT id, organism
        FROM protein
        WHERE id = ANY(%s);
        """,
        (ids,),
    )
    return {
        str(row["id"]): ("" if row.get("organism") is None else str(row["organism"]))
        for row in rows
    }


def main() -> int:
    started_total = time.perf_counter()
    args = parse_args()

    started = time.perf_counter()
    protein_ids = _load_protein_ids(args.protein_id, args.protein_ids_file)
    _log_timing(args.timings, "load protein ids", started)
    if not protein_ids:
        print("No protein IDs found.", file=sys.stderr)
        return 2

    started = time.perf_counter()
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

        started_neighbors = time.perf_counter()
        embeddings_by_protein = db.get_protein_embeddings(
            protein_ids,
            embedding_type.id,
            layer_index=args.layer,
            as_numpy=False,
        )

        for protein_id in protein_ids:
            query_embedding = embeddings_by_protein.get(protein_id)
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
                k=args.k,
                metric=args.metric,
                exclude_protein_ids=[protein_id],
                use_ann=args.use_ann,
            )
            if not neighbors:
                print(
                    f"Skipping {protein_id}: no neighbor found.",
                    file=sys.stderr,
                )
                continue

            for neighbor in neighbors:
                results.append(
                    PairResult(
                        protein_id=protein_id,
                        neighbor_id=neighbor.protein_id,
                        embedding_distance=neighbor.distance,
                    )
                )
                neighbor_ids.add(neighbor.protein_id)
        _log_timing(args.timings, "db: embeddings + nearest neighbors", started_neighbors)

        started_annotations = time.perf_counter()
        all_ids = sorted(set(protein_ids).union(neighbor_ids))
        local_go_ids = db.fetch_protein_go_ids(all_ids)
        _log_timing(args.timings, "db: fetch local GO IDs", started_annotations)

        started_global_go = time.perf_counter()
        global_go_ids = db.fetch_protein_go_ids()
        _log_timing(args.timings, "db: fetch global GO IDs", started_global_go)

        started_species = time.perf_counter()
        species_by_protein = _fetch_species_map(db, all_ids)
        _log_timing(args.timings, "db: fetch species", started_species)
    _log_timing(args.timings, "db total", started)

    if not results:
        print("No protein-neighbor pairs were generated.")
        return 0

    for row in results:
        row.protein_species = species_by_protein.get(row.protein_id, "")
        row.neighbor_species = species_by_protein.get(row.neighbor_id, "")

    started = time.perf_counter()
    go = load_go(args.obo)
    _log_timing(args.timings, "load GO ontology", started)

    started = time.perf_counter()
    local_by_category = go.split_annotations_by_category(local_go_ids)
    global_by_category = go.split_annotations_by_category(global_go_ids)
    _log_timing(args.timings, "split GO terms by category (local+global)", started)

    total_valid_points = 0
    for category_key, category_label in GO_CATEGORIES:
        started_category = time.perf_counter()
        counts_input = global_by_category.get(category_key, {})
        local_sets = local_by_category.get(category_key, {})

        category_rows = [
            PairResult(
                protein_id=row.protein_id,
                neighbor_id=row.neighbor_id,
                embedding_distance=row.embedding_distance,
                protein_species=row.protein_species,
                neighbor_species=row.neighbor_species,
            )
            for row in results
        ]

        if counts_input:
            started_prepare_counts = time.perf_counter()
            go.prepare_term_counts(counts_input)
            _log_timing(
                args.timings,
                f"{category_label}: prepare term counts",
                started_prepare_counts,
            )

        started_similarity = time.perf_counter()
        for row in category_rows:
            source_terms = sorted(local_sets.get(row.protein_id, set()))
            neighbor_terms = sorted(local_sets.get(row.neighbor_id, set()))
            row.protein_terms = go.format_term_names(source_terms)
            row.neighbor_terms = go.format_term_names(neighbor_terms)

            if not source_terms or not neighbor_terms:
                continue

            similarity = go.group_similarity(
                source_terms,
                neighbor_terms,
                method=args.semantic_method,
                aggregate="bma",
            )
            if similarity is None:
                continue
            row.semantic_similarity = similarity
        _log_timing(
            args.timings,
            f"{category_label}: similarity aggregation",
            started_similarity,
        )

        csv_path = _with_suffix(args.csv, category_key)
        if csv_path:
            started_csv = time.perf_counter()
            _write_csv(csv_path, category_rows)
            print(f"Saved table ({category_label}): {csv_path}")
            _log_timing(args.timings, f"{category_label}: write CSV", started_csv)

        plot_path = _with_suffix(args.plot, category_key)
        started_plot = time.perf_counter()
        valid_points = _plot_results(
            category_rows,
            output_path=plot_path or args.plot,
            metric=args.metric,
            semantic_method=args.semantic_method,
            category_label=category_label,
            show_plot=args.show,
        )
        _log_timing(args.timings, f"{category_label}: plot", started_plot)
        total_valid_points += valid_points
        print(
            f"Processed {len(category_rows)} pairs for {category_label}; "
            f"plotted {valid_points} points with valid GO similarity."
        )
        _log_timing(args.timings, f"{category_label}: total", started_category)

    print(
        f"Processed {len(results)} unique protein-neighbor pairs "
        f"across {len(GO_CATEGORIES)} GO categories; "
        f"total plotted points with valid GO similarity: {total_valid_points}."
    )
    _log_timing(args.timings, "total runtime", started_total)
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
