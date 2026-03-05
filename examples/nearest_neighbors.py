#!/usr/bin/env python3
"""Run nearest-neighbor search with GO lookup against BioData.

Example:
    python examples/nearest_neighbors.py \
      --query P12345 \
      --embedding-type-name esm2_layer0 \
      --layer 0 \
      --k 10 \
      --metric cosine
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from CBBIO.BioData import BioDataClient, NotFoundError
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from CBBIO.BioData import BioDataClient, NotFoundError


def _resolve_embedding_type_id(
    client: BioDataClient,
    embedding_type_id: Optional[int],
    embedding_type_name: Optional[str],
) -> int:
    if embedding_type_id is not None:
        return embedding_type_id

    if not embedding_type_name:
        raise ValueError("Provide either --embedding-type-id or --embedding-type-name.")

    emb_type = client.get_embedding_type_by_name(embedding_type_name)
    if emb_type is None:
        raise ValueError(f"Embedding type not found: {embedding_type_name!r}")
    return emb_type.id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BioData nearest-neighbor + GO annotation example")
    parser.add_argument("--query", required=True, help="Query UniProt/protein ID (e.g. P12345)")
    parser.add_argument("--embedding-type-id", type=int, help="sequence_embedding_type.id")
    parser.add_argument("--embedding-type-name", help="sequence_embedding_type.name (e.g. esm2_layer0)")
    parser.add_argument("--layer", type=int, default=0, help="Embedding layer index (default: 0)")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors (default: 10)")
    parser.add_argument(
        "--metric",
        choices=["l2", "cosine", "inner_product"],
        default="l2",
        help="Distance metric (default: l2)",
    )
    parser.add_argument("--include-query", action="store_true", help="Include query ID in returned neighbors")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with BioDataClient() as client:
        health = client.health_check()
        print("Health:", health)

        emb_type_id = _resolve_embedding_type_id(client, args.embedding_type_id, args.embedding_type_name)
        layers = client.list_available_layers(emb_type_id)
        if args.layer not in layers:
            print(f"Warning: layer {args.layer} not found for embedding type {emb_type_id}. Available: {layers}")

        try:
            neighbors, annotations = client.neighbors_with_go(
                query_uniprot_id=args.query,
                embedding_type_id=emb_type_id,
                layer_index=args.layer,
                k=args.k,
                metric=args.metric,
                include_query=args.include_query,
            )
        except NotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    print("\nNeighbors (protein_id, distance):")
    for item in neighbors:
        print(f"  {item.protein_id}\t{item.distance:.6f}")

    print("\nGO annotations:")
    for item in neighbors:
        anns = annotations.get(item.protein_id, [])
        print(f"\n  {item.protein_id}")
        if not anns:
            print("    (no GO terms)")
            continue
        for ann in anns:
            print(f"    {ann.go_id} [{ann.category}] ({ann.evidence_code}) - {ann.description}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
