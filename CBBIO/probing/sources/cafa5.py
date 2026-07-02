"""Load CAFA5 Kaggle protein-function prediction training data."""

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


Cafa5Aspect: TypeAlias = CafaAspect

CAFA5_COLLECTION_METADATA = CAFA_COLLECTION_METADATA
_CAFA5_SPEC = CafaChallengeSpec(
    version="cafa5",
    competition_slug="cafa-5-protein-function-prediction",
    root_alias="cafa_5_protein_function_prediction",
    default_splitter=HashDatasetSplitter(salt="cafa5-train-v1"),
)

CAFA5_DATASETS = build_cafa_datasets(_CAFA5_SPEC)


def load_cafa5_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
    splitter: DatasetSplitter | None = None,
    go_obo_path: str | Path | None = None,
    ia_path: str | Path | None = None,
) -> ProteinDataset:
    """Load a CAFA5 aspect dataset from local Kaggle competition files."""
    return load_cafa_dataset(
        root,
        spec=_CAFA5_SPEC,
        name=name,
        split=split,
        target=target,
        splitter=splitter,
        go_obo_path=go_obo_path,
        ia_path=ia_path,
    )


def download_cafa5_dataset(root: str | Path, *, force: bool = False) -> list[Path]:
    """Download and unzip CAFA5 competition files with the Kaggle CLI."""
    return download_cafa_dataset(root, spec=_CAFA5_SPEC, force=force)


__all__ = [
    "CAFA5_COLLECTION_METADATA",
    "CAFA5_DATASETS",
    "Cafa5Aspect",
    "download_cafa5_dataset",
    "load_cafa5_dataset",
]
