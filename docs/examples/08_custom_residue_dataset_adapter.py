# Prompt:
# Add support for a new residue-level CSV dataset with columns protein_id, sequence,
# split, and binding_site_mask. Implement a typed loader that returns CBBIO.ResidueDataset,
# register it in the probing dataset catalog as status "ready", document it in the
# relevant docs, and add unit tests with a tiny fixture. Follow docs/AddingProbingDataSource.md
# and keep imports in tests and examples from the public CBBIO namespace unless an
# internal adapter base class is required.

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from CBBIO import ResidueDataset, ResidueExample


def parse_binary_mask(value: str, *, protein_id: str, sequence: str) -> list[int]:
    labels = [int(token) for token in value.replace(";", ",").replace(" ", ",").split(",") if token != ""]
    if len(labels) != len(sequence):
        raise ValueError(
            f"binding_site_mask for {protein_id!r} has {len(labels)} values but sequence has {len(sequence)} residues."
        )
    if any(label not in {0, 1} for label in labels):
        raise ValueError(f"binding_site_mask for {protein_id!r} must contain only 0 and 1.")
    return labels


def load_binding_site_csv(path: str | Path, *, target: str = "binding_site") -> ResidueDataset:
    examples: list[ResidueExample] = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            protein_id = str(row["protein_id"]).strip()
            sequence = str(row["sequence"]).strip()
            examples.append(
                ResidueExample(
                    id=protein_id,
                    sequence=sequence,
                    labels={target: parse_binary_mask(str(row["binding_site_mask"]), protein_id=protein_id, sequence=sequence)},
                    split=str(row["split"]).strip().lower(),
                )
            )
    return ResidueDataset(examples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_binding_site_csv(args.csv_path)
    counts = dataset.split_counts()
    print(f"loaded={sum(counts.values())} train={counts['train']} val={counts['val']} test={counts['test']}")


if __name__ == "__main__":
    main()
