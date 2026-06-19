"""Download and load MusiteDeep residue-level PTM datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import gzip
import json
from pathlib import Path
from typing import cast
import urllib.request

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ResidueDataset, ResidueExample, SplitName


MUSITEDEEP_TESTDATA_API_URL = (
    "https://api.github.com/repos/duolinwang/MusiteDeep/contents/testdata?ref=master"
)


def download_musitedeep_testdata(
    root: str | Path,
    *,
    file_names: Sequence[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download FASTA files from the MusiteDeep GitHub testdata directory.

    Args:
        root: Local dataset root.
        file_names: Optional subset of FASTA filenames.
        force: Download files even when local copies exist.

    Returns:
        Downloaded or existing FASTA paths.

    Raises:
        EmbeddingInputError: If requested files are absent from the repository.
    """
    output_dir = Path(root).expanduser() / "musitedeep" / "testdata"
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {str(name) for name in file_names} if file_names is not None else None
    paths: list[Path] = []
    for record in _github_contents(MUSITEDEEP_TESTDATA_API_URL):
        name = record.get("name")
        download_url = record.get("download_url")
        item_type = record.get("type")
        if not isinstance(name, str) or not name.endswith(".fasta"):
            continue
        if wanted is not None and name not in wanted:
            continue
        if item_type != "file" or not isinstance(download_url, str):
            continue
        path = output_dir / name
        if force or not path.exists():
            urllib.request.urlretrieve(download_url, path)
        paths.append(path)
    if wanted is not None:
        missing = sorted(wanted - {path.name for path in paths})
        if missing:
            raise EmbeddingInputError(
                f"MusiteDeep testdata files not found on GitHub: {', '.join(missing)}."
            )
    return paths


def load_musitedeep_testdata_dataset(
    root: str | Path,
    *,
    target: str = "phosphorylation",
    file_names: Sequence[str] | None = None,
    download: bool = False,
) -> ResidueDataset:
    """Load MusiteDeep GitHub testdata FASTA files as one residue dataset."""
    data_dir = Path(root).expanduser() / "musitedeep" / "testdata"
    paths = (
        download_musitedeep_testdata(root, file_names=file_names)
        if download
        else _local_fastas(data_dir, file_names=file_names)
    )
    examples: list[ResidueExample] = []
    for path in paths:
        split: SplitName = "test" if "test" in path.stem.lower() else "train"
        partial = load_musitedeep_fasta(path, target=target, split=split)
        examples.extend(
            ResidueExample(
                id=f"{path.stem}:{example.id}",
                sequence=example.sequence,
                labels=example.labels,
                split=example.split,
                mask=example.mask,
                metadata={"source": "musitedeep_testdata", "file": path.name},
            )
            for example in partial.examples
        )
    return ResidueDataset(examples)


def load_musitedeep_fasta(
    path: str | Path,
    *,
    target: str = "ptm_site",
    split: SplitName = "train",
) -> ResidueDataset:
    """Load MusiteDeep FASTA where ``#`` marks the preceding residue."""
    examples: list[ResidueExample] = []
    for record_id, raw_sequence in _iter_fasta(path):
        sequence, labels = _parse_marked_sequence(raw_sequence)
        examples.append(
            ResidueExample(
                id=record_id,
                sequence=sequence,
                labels={target: labels},
                split=split,
                metadata={"source": "musitedeep"},
            )
        )
    return ResidueDataset(examples)


def _github_contents(api_url: str) -> list[Mapping[str, object]]:
    request = urllib.request.Request(
        api_url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "CBBIO"},
    )
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise EmbeddingInputError("GitHub contents API did not return a file list.")
    records: list[Mapping[str, object]] = []
    for item in cast(list[object], payload):
        if isinstance(item, Mapping):
            records.append(cast(Mapping[str, object], item))
    return records


def _local_fastas(root: Path, *, file_names: Sequence[str] | None) -> list[Path]:
    paths = (
        [root / name for name in file_names]
        if file_names is not None
        else sorted(root.glob("*.fasta"))
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise EmbeddingInputError(
            "Missing MusiteDeep testdata FASTA files. Pass download=True or download them first. "
            f"Missing: {', '.join(missing[:5])}."
        )
    if not paths:
        raise EmbeddingInputError(f"No MusiteDeep FASTA files found under {root}.")
    return paths


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


def _parse_marked_sequence(raw_sequence: str) -> tuple[str, list[int]]:
    residues: list[str] = []
    labels: list[int] = []
    for char in raw_sequence:
        if char == "#":
            if not labels:
                raise EmbeddingInputError(
                    "MusiteDeep marker '#' appears before any residue."
                )
            labels[-1] = 1
            continue
        residues.append(char)
        labels.append(0)
    return "".join(residues), labels


__all__ = [
    "MUSITEDEEP_TESTDATA_API_URL",
    "download_musitedeep_testdata",
    "load_musitedeep_fasta",
    "load_musitedeep_testdata_dataset",
]
