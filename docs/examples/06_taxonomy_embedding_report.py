# Prompt:
# Write a report generator that groups proteins by taxonomy lineage. Use
# CBBIO.load_taxonomy to load NCBI taxdump, CBBIO.connect to fetch protein taxonomy
# ids and embeddings, and CBBIO.Taxonomy.compute_taxon_ic_and_lin_maps to compute
# taxonomy information content and pairwise Lin similarity for query species. For
# each species pair, compare taxonomy similarity with mean embedding distance. Output
# a concise markdown report with tables and plots, and keep plotting code optional if
# matplotlib is unavailable.

from __future__ import annotations

import argparse
import itertools
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from CBBIO import connect, load_taxonomy
from CBBIO.Taxonomy import compute_taxon_ic_and_lin_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxdump-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--embedding-type-id", type=int, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--taxon-id", action="append", required=True)
    return parser.parse_args()


def fetch_taxon_embeddings(client: Any, taxon_ids: Sequence[str], embedding_type_id: int, layer_index: int) -> dict[str, list[Sequence[float]]]:
    rows = client.query_all(
        """
        SELECT p.taxonomy_id, se.embedding
        FROM protein p
        JOIN sequence s ON s.id = p.sequence_id
        JOIN sequence_embeddings se ON se.sequence_id = s.id
        WHERE p.taxonomy_id = ANY(%s)
          AND se.embedding_type_id = %s
          AND se.layer_index = %s;
        """,
        (list(taxon_ids), embedding_type_id, layer_index),
    )
    grouped: dict[str, list[Sequence[float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["taxonomy_id"]), []).append(row["embedding"])
    return grouped


def mean_vector(vectors: Sequence[Sequence[float]]) -> list[float]:
    width = len(vectors[0])
    return [sum(vector[index] for vector in vectors) / len(vectors) for index in range(width)]


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return 1.0 if left_norm == 0.0 or right_norm == 0.0 else 1.0 - numerator / (left_norm * right_norm)


def main() -> None:
    args = parse_args()
    taxonomy = load_taxonomy(args.taxdump_dir)
    client = connect()
    grouped = fetch_taxon_embeddings(client, args.taxon_id, args.embedding_type_id, args.layer_index)
    means = {taxon_id: mean_vector(vectors) for taxon_id, vectors in grouped.items() if vectors}
    _, lin_map = compute_taxon_ic_and_lin_maps(taxonomy, query_tax_ids=args.taxon_id, subject_tax_ids=args.taxon_id, observed_tax_ids=args.taxon_id)

    lines = ["# Taxonomy Embedding Report", "", "| Taxon A | Taxon B | Taxonomy Lin | Embedding distance |", "|---|---|---|---|"]
    for left, right in itertools.combinations(sorted(means), 2):
        lin = lin_map.get(left, {}).get(right, 0.0)
        distance = cosine_distance(means[left], means[right])
        lines.append(f"| {left} | {right} | {lin:.4f} | {distance:.4f} |")
    Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
