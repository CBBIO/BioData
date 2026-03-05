#!/usr/bin/env python3
"""Lookup proteins by protein ID or accession code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from CBBIO.BioData import BioDataClient
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from CBBIO.BioData import BioDataClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BioData protein lookup example")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--protein-id", help="protein.id value")
    group.add_argument("--accession", help="accession.code value")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with BioDataClient() as client:
        if args.protein_id:
            protein = client.get_protein(args.protein_id)
            if protein is None:
                print(f"Protein not found: {args.protein_id}")
                return 2
        else:
            protein = client.get_protein_by_accession(args.accession)
            if protein is None:
                print(f"No protein found for accession: {args.accession}")
                return 2

        protein_id = str(protein["id"])
        accessions = client.list_accessions_for_protein(protein_id)
        go_annotations = client.get_protein_go_annotations(protein_id)
        structures = client.get_protein_structures(protein_id)

    payload = {
        "protein": protein,
        "accessions": accessions,
        "go_annotations": go_annotations,
        "structures": structures,
    }

    if args.pretty:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
