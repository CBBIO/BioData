#!/usr/bin/env python3
"""Generate a deterministic random protein ID list and FASTA from BioData."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Force local repo import before site-packages when running this script directly.
_SCRIPT_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from CBBIO.BioData import BioDataClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample proteins from the BioData DB and write IDs plus FASTA.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10_000,
        help="Number of proteins to sample. Default: 10000.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="benchmark-10000",
        help="Stable sampling seed used in SQL ordering. Default: benchmark-10000.",
    )
    parser.add_argument(
        "--ids-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "random_protein_ids_10000.txt",
        help="Output path for protein IDs.",
    )
    parser.add_argument(
        "--fasta-out",
        type=Path,
        default=_SCRIPT_REPO_ROOT / "random_protein_ids_10000.fasta",
        help="Output path for FASTA records.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=80,
        help="FASTA sequence line width. Default: 80.",
    )
    return parser.parse_args()


def _wrap_sequence(sequence: str, width: int) -> str:
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def _sample_protein_ids(client: BioDataClient, *, count: int, seed: str) -> list[str]:
    rows = client.query_all(
        """
        SELECT p.id
        FROM protein p
        JOIN sequence s ON s.id = p.sequence_id
        WHERE s.sequence IS NOT NULL
          AND s.sequence <> ''
        ORDER BY md5(p.id || %s)
        LIMIT %s;
        """,
        (seed, count),
    )
    return [str(row["id"]) for row in rows if row.get("id") is not None]


def _write_ids(path: Path, protein_ids: Sequence[str]) -> None:
    path.write_text("\n".join(protein_ids) + "\n", encoding="utf-8")


def _write_fasta(path: Path, protein_ids: Sequence[str], sequences: dict[str, str], width: int) -> None:
    lines: list[str] = []
    for protein_id in protein_ids:
        sequence = sequences.get(protein_id)
        if not sequence:
            continue
        lines.append(f">{protein_id}")
        lines.append(_wrap_sequence(sequence, width))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.line_width < 1:
        raise SystemExit("--line-width must be >= 1")

    with BioDataClient() as client:
        protein_ids = _sample_protein_ids(client, count=args.count, seed=args.seed)
        if not protein_ids:
            raise SystemExit("No proteins found in database")
        sequences = client.get_protein_sequences(protein_ids)

    missing_ids = [protein_id for protein_id in protein_ids if protein_id not in sequences]
    if missing_ids:
        raise SystemExit(
            f"Sampled {len(protein_ids)} proteins but only fetched {len(sequences)} sequences; "
            f"missing {len(missing_ids)} sequences"
        )

    args.ids_out.parent.mkdir(parents=True, exist_ok=True)
    args.fasta_out.parent.mkdir(parents=True, exist_ok=True)
    _write_ids(args.ids_out, protein_ids)
    _write_fasta(args.fasta_out, protein_ids, sequences, args.line_width)

    print(f"Wrote {len(protein_ids)} protein IDs to {args.ids_out}")
    print(f"Wrote {len(protein_ids)} FASTA records to {args.fasta_out}")


if __name__ == "__main__":
    main()
