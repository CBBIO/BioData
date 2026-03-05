#!/usr/bin/env python3
"""Traverse structure -> chains -> states -> optional 3Di rows."""

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
    parser = argparse.ArgumentParser(description="BioData structure drilldown example")
    parser.add_argument("--structure-id", required=True, help="structure.id value")
    parser.add_argument("--include-3di", action="store_true", help="Include 3Di records per state")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with BioDataClient() as client:
        chains = client.get_structure_chains(args.structure_id)
        if not chains:
            print(f"No chains found for structure: {args.structure_id}")
            return 2

        states_by_chain = {}
        structure_3di_by_state = {}

        for chain in chains:
            chain_id = int(chain["id"])
            states = client.get_chain_states(chain_id)
            states_by_chain[chain_id] = states

            if args.include_3di:
                for state in states:
                    state_id = int(state["id"])
                    structure_3di_by_state[state_id] = client.get_state_3di_embeddings(state_id)

    payload = {
        "structure_id": args.structure_id,
        "chains": chains,
        "states_by_chain": states_by_chain,
    }
    if args.include_3di:
        payload["structure_3di_by_state"] = structure_3di_by_state

    if args.pretty:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
