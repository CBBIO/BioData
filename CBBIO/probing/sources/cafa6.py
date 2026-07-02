"""Load CAFA6 Kaggle protein-function prediction training data."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

from ..datasets import ProteinDataset
from ..splitters import DatasetSplitter, HashDatasetSplitter
from ._cafa import (
    CAFA_COLLECTION_METADATA,
    CafaAspect,
    CafaChallengeSpec,
    build_cafa_datasets,
    download_cafa_dataset,
    load_cafa_dataset,
)


Cafa6Aspect: TypeAlias = CafaAspect

CAFA6_COLLECTION_METADATA = CAFA_COLLECTION_METADATA
_CAFA6_SPEC = CafaChallengeSpec(
    version="cafa6",
    competition_slug="cafa-6-protein-function-prediction",
    root_alias="cafa_6_protein_function_prediction",
    default_splitter=HashDatasetSplitter(salt="cafa6-train-v1"),
)

CAFA6_DATASETS = build_cafa_datasets(_CAFA6_SPEC)


def load_cafa6_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
    splitter: DatasetSplitter | None = None,
    go_obo_path: str | Path | None = None,
    ia_path: str | Path | None = None,
) -> ProteinDataset:
    """Load a CAFA6 aspect dataset from local Kaggle competition files."""
    return load_cafa_dataset(
        root,
        spec=_CAFA6_SPEC,
        name=name,
        split=split,
        target=target,
        splitter=splitter,
        go_obo_path=go_obo_path,
        ia_path=ia_path,
    )


def download_cafa6_dataset(root: str | Path, *, force: bool = False) -> list[Path]:
    """Download and unzip CAFA6 competition files with the Kaggle CLI."""
    return download_cafa_dataset(root, spec=_CAFA6_SPEC, force=force)


__all__ = [
    "CAFA_COLLECTION_METADATA",
    "CAFA6_COLLECTION_METADATA",
    "CAFA6_DATASETS",
    "Cafa6Aspect",
    "download_cafa6_dataset",
    "load_cafa6_dataset",
]
