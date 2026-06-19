"""Load BioLiP residue-level ligand-binding datasets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import gzip
import hashlib
from pathlib import Path
import re
from typing import Any, Literal, cast

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ResidueDataset, ResidueExample, SplitName


BioLipLigandClass = Literal["all", "dna", "rna", "pep", "other"]

BIOLIP_DOWNLOAD_URLS = (
    "https://zhanggroup.org/BioLiP/download/BioLiP_nr.txt.gz",
    "https://zhanggroup.org/BioLiP/data/protein_nr.fasta.gz",
)


def load_biolip_dataset(
    annotation_path: str | Path,
    *,
    protein_fasta: str | Path | None = None,
    target: str | None = None,
    split: SplitName | None = "train",
    ligand_class: BioLipLigandClass = "all",
) -> ResidueDataset:
    """Load BioLiP annotations into residue-level ligand-binding labels.

    Args:
        annotation_path: BioLiP annotation table, optionally gzip-compressed.
        protein_fasta: Optional receptor sequences indexed by PDB-chain identifier.
        target: Label name stored in each residue example.
        split: Dataset split, or ``None`` to assign deterministic splits.
        ligand_class: Ligand subset to retain.

    Returns:
        Residue-level ligand-binding dataset.

    Raises:
        EmbeddingInputError: If annotations or receptor sequences are inconsistent.
    """
    resolved_ligand_class = _normalize_ligand_class(ligand_class)
    resolved_target = target or _target_for_ligand_class(resolved_ligand_class)
    fasta_sequences = (
        dict(_iter_fasta(protein_fasta))
        if protein_fasta is not None and Path(protein_fasta).exists()
        else {}
    )
    grouped: dict[str, dict[str, Any]] = {}
    opener = gzip.open if str(annotation_path).endswith(".gz") else open
    with opener(annotation_path, "rt", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t") if "\t" in line else line.split()
            if len(columns) < 21:
                raise EmbeddingInputError(
                    f"BioLiP row {line_index} has {len(columns)} columns; expected at least 21."
                )
            if not _ligand_matches(columns[4], resolved_ligand_class):
                continue
            record_id = f"{columns[0]}{columns[1]}"
            sequence = fasta_sequences.get(record_id, columns[20])
            group = grouped.setdefault(
                record_id,
                {"sequence": sequence, "labels": [0] * len(sequence)},
            )
            if group["sequence"] != sequence:
                raise EmbeddingInputError(
                    f"Conflicting BioLiP sequences for receptor {record_id!r}."
                )
            for position in _binding_positions(columns[8]):
                _mark_position(cast(list[int], group["labels"]), position=position)

    examples = [
        ResidueExample(
            id=record_id,
            sequence=str(group["sequence"]),
            labels={resolved_target: cast(list[int], group["labels"])},
            split=split or "train",
            metadata={"source": "biolip", "ligand_class": resolved_ligand_class},
        )
        for record_id, group in grouped.items()
    ]
    if split is None:
        examples = _split_examples(examples)
    return ResidueDataset(examples)


def _iter_fasta(path: str | Path) -> Iterable[tuple[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records.append((current_id, "".join(chunks)))
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            records.append((current_id, "".join(chunks)))
    return records


def _normalize_ligand_class(value: str) -> BioLipLigandClass:
    normalized = str(value).strip().lower()
    if normalized in {"all", "dna", "rna", "other"}:
        return cast(BioLipLigandClass, normalized)
    if normalized in {"pep", "peptide"}:
        return "pep"
    raise EmbeddingInputError(
        "BioLiP ligand_class must be one of: all, dna, rna, pep, other."
    )


def _target_for_ligand_class(ligand_class: BioLipLigandClass) -> str:
    targets = {
        "dna": "dna_binding_site",
        "rna": "rna_binding_site",
        "pep": "peptide_binding_site",
        "other": "other_ligand_binding_site",
    }
    return targets.get(ligand_class, "ligand_binding_site")


def _ligand_matches(value: str, ligand_class: BioLipLigandClass) -> bool:
    ligand = str(value).strip().lower()
    if ligand_class == "all":
        return True
    if ligand_class == "pep":
        return ligand == "peptide"
    if ligand_class == "other":
        return ligand not in {"dna", "rna", "peptide"}
    return ligand == ligand_class


def _binding_positions(value: str) -> list[int]:
    positions: list[int] = []
    for token in value.replace(";", " ").split():
        match = re.search(r"(-?\d+)$", token)
        if match is not None:
            positions.append(int(match.group(1)))
    return positions


def _mark_position(labels: list[int], *, position: int) -> None:
    if position < 1 or position > len(labels):
        raise EmbeddingInputError(
            f"Residue interval {position}-{position} is outside sequence length {len(labels)}."
        )
    labels[position - 1] = 1


def _split_examples(examples: Sequence[ResidueExample]) -> list[ResidueExample]:
    ordered = sorted(examples, key=lambda example: _stable_hash(example.id))
    counts = _split_counts(len(ordered))
    split_names: list[SplitName] = [
        split_name for split_name, count in counts for _ in range(count)
    ]
    return [
        ResidueExample(
            id=example.id,
            sequence=example.sequence,
            labels=example.labels,
            split=split_name,
            mask=example.mask,
            metadata=example.metadata,
        )
        for example, split_name in zip(ordered, split_names)
    ]


def _split_counts(total: int) -> list[tuple[SplitName, int]]:
    if total <= 0:
        return [("train", 0), ("val", 0), ("test", 0)]
    validation = int(round(total * 0.1))
    test = int(round(total * 0.1))
    if total >= 10:
        validation = max(1, validation)
        test = max(1, test)
    elif total >= 2:
        test = max(1, test)
    train = total - validation - test
    while train < 1 and validation > 0:
        validation -= 1
        train += 1
    while train < 1 and test > 0:
        test -= 1
        train += 1
    return [("train", train), ("val", validation), ("test", test)]


def _stable_hash(value: str) -> str:
    return hashlib.sha256(f"biolip-split-v1:{value}".encode("utf-8")).hexdigest()


__all__ = [
    "BIOLIP_DOWNLOAD_URLS",
    "BioLipLigandClass",
    "load_biolip_dataset",
]
