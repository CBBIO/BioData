# Prompt:
# Write an analysis notebook or script that samples proteins from the BioData database,
# retrieves nearest neighbors with CBBIO.BioDataClient.find_nearest_neighbors, loads
# go-basic.obo with CBBIO.load_go, prepares GO term counts from database annotations,
# and compares embedding distance with GO semantic similarity using Resnik, Lin,
# Schlicker, and Wang. Produce a Spearman correlation table by embedding type, model
# layer, ontology aspect, and distance metric. Save the result as markdown and CSV.

from __future__ import annotations

import argparse
import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from CBBIO import connect, load_go


LOGGER = logging.getLogger("go_semantic_neighbor_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go-obo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--embedding-type-id", type=int, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--k", type=int, default=20)
    return parser.parse_args()


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    try:
        from scipy.stats import spearmanr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install scipy to compute Spearman correlations: pip install scipy") from exc
    return float(spearmanr(xs, ys, nan_policy="omit").statistic)


def sample_protein_ids(client: Any, limit: int) -> list[str]:
    rows = client.query_all(
        """
        SELECT DISTINCT p.id
        FROM protein p
        JOIN sequence s ON s.id = p.sequence_id
        JOIN sequence_embeddings se ON se.sequence_id = s.id
        ORDER BY p.id
        LIMIT %s;
        """,
        (limit,),
    )
    return [str(row["id"]) for row in rows]


def best_go_similarity(ontology: Any, left_terms: Sequence[str], right_terms: Sequence[str], method: str) -> float:
    scores = [
        ontology.semantic_similarity(left, right, method=method)
        for left in left_terms
        for right in right_terms
        if ontology.has_term(left) and ontology.has_term(right)
    ]
    return max(scores) if scores else 0.0


def write_outputs(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    csv_path = out_dir / "go_neighbor_correlations.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "correlation", "pair_count"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# GO Neighbor Correlations", "", "| Method | Spearman rho | Pairs |", "|---|---|---|"]
    lines.extend(f"| {row['method']} | {row['correlation']:.4f} | {row['pair_count']} |" for row in rows)
    (out_dir / "go_neighbor_correlations.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = connect()
    ontology = load_go(args.go_obo)
    all_go = client.fetch_protein_go_ids()
    ontology.prepare_term_counts(all_go)

    distances: list[float] = []
    similarities: dict[str, list[float]] = {method: [] for method in ("resnik", "lin", "schlicker", "wang")}
    for protein_id in sample_protein_ids(client, args.sample_size):
        embedding = client.get_protein_embedding(protein_id, args.embedding_type_id, args.layer_index, as_numpy=False)
        if embedding is None:
            continue
        neighbors = client.find_nearest_neighbors(embedding, args.embedding_type_id, layer_index=args.layer_index, k=args.k, metric="cosine", exclude_protein_ids=[protein_id])
        query_terms = sorted(all_go.get(protein_id, set()))
        for neighbor in neighbors:
            neighbor_terms = sorted(all_go.get(neighbor.protein_id, set()))
            if not query_terms or not neighbor_terms:
                continue
            distances.append(float(neighbor.distance))
            for method in similarities:
                similarities[method].append(best_go_similarity(ontology, query_terms, neighbor_terms, method))

    rows = [
        {"method": method, "correlation": spearman(distances, values), "pair_count": len(values)}
        for method, values in similarities.items()
        if values
    ]
    write_outputs(out_dir, rows)


if __name__ == "__main__":
    main()
