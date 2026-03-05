#!/usr/bin/env python3
"""Fetch aggregated protein context (accessions, GO, structures, chains, states)."""

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
    parser = argparse.ArgumentParser(description="BioData protein context example")
    parser.add_argument("--protein-id", required=True, help="protein.id value")
    parser.add_argument("--include-3di", action="store_true", help="Include structure_3di rows")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with BioDataClient() as client:
        context = client.get_protein_context(
            args.protein_id,
            include_3di=args.include_3di,
        )

    if context is None:
        print(f"Protein not found: {args.protein_id}")
        return 2

    if args.pretty:
        print(json.dumps(context, indent=2, default=str))
    else:
        print(json.dumps(context, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
