# Prompt:
# Write a script that connects to the BioData PostgreSQL database with CBBIO.connect,
# finds all proteins annotated to Wnt signaling or its descendant GO terms, retrieves
# or generates ProstT5 mean-pooled embeddings for those proteins, computes pairwise
# cosine distances, and builds a NetworkX graph where edges connect proteins below a
# configurable distance threshold. Save graph.graphml and nodes.tsv with UniProt id,
# organism, taxonomy id, GO terms, and embedding metadata. Use public CBBIO APIs where
# available and keep database SQL isolated in one typed helper.

from __future__ import annotations

import argparse
import csv
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from CBBIO import GenerationInput, Generator, IterableBatcher, EmbeddingWriter, connect, load_go, pooler_factory, run_embedding_generation


LOGGER = logging.getLogger("wnt_signaling_graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go-obo", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--wnt-go-id", default="GO:0016055")
    return parser.parse_args()


def fetch_proteins_for_go_terms(client: Any, go_ids: Sequence[str]) -> list[dict[str, Any]]:
    return client.query_all(
        """
        SELECT DISTINCT p.id, p.organism, p.taxonomy_id, s.sequence
        FROM protein p
        JOIN sequence s ON s.id = p.sequence_id
        JOIN protein_go_term_annotation pga ON pga.protein_id = p.id
        WHERE pga.go_id = ANY(%s)
        ORDER BY p.id;
        """,
        (list(go_ids),),
    )


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - numerator / (left_norm * right_norm)


def write_nodes(path: Path, proteins: Sequence[Mapping[str, Any]], go_by_protein: Mapping[str, Sequence[Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_id", "organism", "taxonomy_id", "go_terms"])
        writer.writeheader()
        for protein in proteins:
            protein_id = str(protein["id"])
            writer.writerow(
                {
                    "protein_id": protein_id,
                    "organism": protein.get("organism", ""),
                    "taxonomy_id": protein.get("taxonomy_id", ""),
                    "go_terms": ";".join(annotation.go_id for annotation in go_by_protein.get(protein_id, [])),
                }
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ontology = load_go(args.go_obo)
    go_ids = ontology.descendants(args.wnt_go_id, include_self=True)
    client = connect()
    proteins = fetch_proteins_for_go_terms(client, go_ids)
    records = [GenerationInput(id=str(row["id"]), sequence=str(row["sequence"])) for row in proteins]

    writer = EmbeddingWriter(format="memory")
    run_embedding_generation(
        Generator(model_class="prostt5", device=args.device),
        IterableBatcher(records, batch_size=8),
        writer,
        layer_index=0,
        pooler=pooler_factory("mean"),
    )
    embeddings = {record.id: record.embedding for record in writer.records}
    go_by_protein = client.fetch_go_annotations([str(row["id"]) for row in proteins])

    try:
        import networkx as nx
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install networkx to write graph output: pip install networkx") from exc

    graph = nx.Graph()
    for protein in proteins:
        graph.add_node(str(protein["id"]), organism=str(protein.get("organism", "")), taxonomy_id=str(protein.get("taxonomy_id", "")))

    ids = sorted(embeddings)
    for index, left_id in enumerate(ids):
        for right_id in ids[index + 1:]:
            distance = cosine_distance(embeddings[left_id], embeddings[right_id])
            if distance <= args.threshold:
                graph.add_edge(left_id, right_id, distance=distance)

    nx.write_graphml(graph, out_dir / "graph.graphml")
    write_nodes(out_dir / "nodes.tsv", proteins, go_by_protein)


if __name__ == "__main__":
    main()
