# Prompt:
# Build a script that takes a query FASTA and a BioData database connection, aligns
# query sequences to database sequences with CBBIO.align_sequences, generates query
# embeddings with ESM-2 and ProteinGLM, retrieves database nearest neighbors for
# matching embedding types, and combines sequence identity, embedding distance, GO
# similarity, and taxonomy Lin similarity into a ranked candidate table. The output
# should include one TSV for machine use and one markdown report explaining the top
# candidates and any missing annotations.

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from CBBIO import (
    EmbeddingWriter,
    FastaBatcher,
    Generator,
    align_sequences,
    connect,
    load_go,
    pooler_factory,
    run_embedding_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-fasta", required=True)
    parser.add_argument("--go-obo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--esm2-type-id", type=int, required=True)
    parser.add_argument("--proteinglm-type-id", type=int, required=True)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def generate_queries(fasta: str, model_class: str, device: str, layer_index: int) -> dict[str, Sequence[float]]:
    writer = EmbeddingWriter(format="memory")
    run_embedding_generation(
        Generator(model_class=model_class, device=device),
        FastaBatcher(fasta, max_batch_tokens=32768, max_sequence_length=4000),
        writer,
        layer_index=layer_index,
        pooler=pooler_factory("mean"),
        fail_fast=False,
    )
    return {record.id: record.embedding for record in writer.records}


def best_go_similarity(ontology: Any, left_terms: Sequence[str], right_terms: Sequence[str]) -> float:
    scores = [
        ontology.semantic_similarity(left, right, method="lin")
        for left in left_terms
        for right in right_terms
        if ontology.has_term(left) and ontology.has_term(right)
    ]
    return max(scores) if scores else 0.0


def write_reports(out_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with (out_dir / "candidates.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["query_id", "candidate_id"])
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Cross-Modal Similarity Explorer", "", "| Query | Candidate | Model | Distance | Identity | GO Lin |", "|---|---|---|---|---|---|"]
    for row in rows[:25]:
        lines.append(f"| {row['query_id']} | {row['candidate_id']} | {row['model']} | {row['embedding_distance']:.4f} | {row['sequence_identity']:.4f} | {row['go_similarity']:.4f} |")
    (out_dir / "candidates.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = connect()
    ontology = load_go(args.go_obo)
    all_go = client.fetch_protein_go_ids()
    ontology.prepare_term_counts(all_go)

    rows: list[dict[str, Any]] = []
    for model_class, type_id in {"esm2": args.esm2_type_id, "proteinglm": args.proteinglm_type_id}.items():
        query_embeddings = generate_queries(args.query_fasta, model_class, args.device, args.layer_index)
        for query_id, embedding in query_embeddings.items():
            query_sequence = client.get_protein_sequence(query_id) or ""
            query_terms = sorted(all_go.get(query_id, set()))
            neighbors = client.find_nearest_neighbors(embedding, type_id, layer_index=args.layer_index, k=args.k, metric="cosine")
            for neighbor in neighbors:
                candidate_sequence = client.get_protein_sequence(neighbor.protein_id) or ""
                alignment = align_sequences(query_sequence, candidate_sequence, mode="global") if query_sequence and candidate_sequence else None
                candidate_terms = sorted(all_go.get(neighbor.protein_id, set()))
                rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": neighbor.protein_id,
                        "model": model_class,
                        "embedding_distance": float(neighbor.distance),
                        "sequence_identity": float(alignment.identity) if alignment is not None else 0.0,
                        "go_similarity": best_go_similarity(ontology, query_terms, candidate_terms),
                    }
                )
    rows.sort(key=lambda row: (row["embedding_distance"], -row["go_similarity"], -row["sequence_identity"]))
    write_reports(out_dir, rows)


if __name__ == "__main__":
    main()
