"""Load CAFA5 Kaggle protein-function prediction training and target data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
from typing import TypeAlias

from CBBIO.embeddings import EmbeddingInputError

from ..collection_types import DatasetMetadata
from ..datasets import ProteinDataset, ProteinExample
from ..splitters import DatasetSplitter, HashDatasetSplitter
from ._cafa import (
    CAFA_ASPECT_NAMES,
    CAFA_ASPECTS,
    CAFA_COLLECTION_METADATA,
    CafaAspect,
    CafaChallengeSpec,
    build_cafa_datasets,
    cafa_example_metadata,
    download_cafa_dataset,
    load_cafa_dataset,
    parse_cafa_dataset_name,
    read_cafa_fasta_sequences,
    read_cafa_terms,
    read_cafa_train_terms,
    resolve_cafa_go_obo_path,
    resolve_cafa_ia_path,
    resolve_cafa_root,
    resolve_cafa_splits,
)


Cafa5Aspect: TypeAlias = CafaAspect
Cafa5Subset: TypeAlias = str

LOGGER = logging.getLogger(__name__)

CAFA5_TARGET_SUBSETS: Mapping[Cafa5Subset, str] = {
    "limited": "eval_terms_limited_2025_03.tsv",
    "limited_after_t0": "eval_terms_limited_2025_03_publishedaftert0.tsv",
    "no_knowledge": "eval_terms_no_knowledge_2025_03.tsv",
    "no_knowledge_after_t0": "eval_terms_no_knowledge_2025_03_publishedaftert0.tsv",
    "partial": "eval_terms_partial_2025_03.tsv",
    "partial_after_t0": "eval_terms_partial_2025_03_publishedaftert0.tsv",
}
_CAFA5_SUBSET_DISPLAY_NAMES: Mapping[Cafa5Subset, str] = {
    "limited": "limited knowledge",
    "limited_after_t0": "limited knowledge published after t0",
    "no_knowledge": "no knowledge",
    "no_knowledge_after_t0": "no knowledge published after t0",
    "partial": "partial knowledge",
    "partial_after_t0": "partial knowledge published after t0",
}
_CAFA5_SUBSET_KNOWN_FILES: Mapping[Cafa5Subset, str] = {
    "partial": "known_t0.tsv",
    "partial_after_t0": "known_publishedaftert0.tsv",
}

CAFA5_COLLECTION_METADATA = CAFA_COLLECTION_METADATA
_CAFA5_SPEC = CafaChallengeSpec(
    version="cafa5",
    competition_slug="cafa-5-protein-function-prediction",
    root_alias="cafa_5_protein_function_prediction",
    default_splitter=HashDatasetSplitter(salt="cafa5-train-v1"),
)


def _build_cafa5_subset_datasets() -> tuple[DatasetMetadata, ...]:
    datasets: list[DatasetMetadata] = []
    for subset in CAFA5_TARGET_SUBSETS:
        for aspect in CAFA_ASPECTS:
            name = f"cafa5_{subset}_{aspect}"
            datasets.append(
                DatasetMetadata(
                    id=f"cafa:{name}",
                    name=name,
                    display_name=f"CAFA5 {_CAFA5_SUBSET_DISPLAY_NAMES[subset]} {CAFA_ASPECT_NAMES[aspect]}",
                    collection="cafa",
                    source="CAFA5 released target files",
                    category="function_prediction",
                    task_class="function",
                    preferred_metric="go_weighted_fmax",
                    description=(
                        "Source: CAFA5 released target files. "
                        "Class: function_prediction. "
                        "Split system: Kaggle training proteins as train and released CAFA5 target subset as test."
                    ),
                    level="protein",
                    objective="multilabel",
                    target=f"go_{aspect}",
                    status="adapter",
                    metrics=(
                        "go_weighted_fmax",
                        "go_weighted_precision_at_fmax",
                        "go_weighted_recall_at_fmax",
                        "go_fmax",
                    ),
                    homepage="https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data",
                    download_adapter="download_cafa5_dataset",
                    import_adapter="load_cafa5_dataset",
                    loader="load_cafa5_dataset",
                    tags=(
                        "protein",
                        "function",
                        "gene_ontology",
                        "go",
                        "cafa",
                        "cafa5",
                        subset,
                        _cafa5_setting_tag(subset),
                        aspect,
                    ),
                    notes=(
                        "Requires Train/train_terms.tsv, Train/train_sequences.fasta, "
                        f"and Test (Targets)/{CAFA5_TARGET_SUBSETS[subset]}. "
                        "If test target sequences are not present in the training FASTA, add a "
                        "testsuperset.fasta file under Test/ or Test (Targets)/."
                    ),
                )
            )
    return tuple(datasets)


def _cafa5_setting_tag(subset: Cafa5Subset) -> str:
    if subset.startswith("no_knowledge"):
        return "nk"
    if subset.startswith("limited"):
        return "lk"
    return "pk"


CAFA5_DATASETS = (*build_cafa_datasets(_CAFA5_SPEC), *_build_cafa5_subset_datasets())


def load_cafa5_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
    splitter: DatasetSplitter | None = None,
    go_obo_path: str | Path | None = None,
    ia_path: str | Path | None = None,
    require_complete_subset_sequences: bool = False,
) -> ProteinDataset:
    """Load a CAFA5 aspect dataset from local Kaggle competition files."""
    aspect, subset = _parse_cafa5_dataset_name(name)
    if subset is not None:
        if splitter is not None:
            raise EmbeddingInputError("CAFA5 target-subset datasets define fixed train/test splits.")
        return _load_cafa5_subset_dataset(
            root,
            aspect=aspect,
            subset=subset,
            split=split,
            target=target,
            go_obo_path=go_obo_path,
            ia_path=ia_path,
            require_complete_subset_sequences=require_complete_subset_sequences,
        )
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


def _parse_cafa5_dataset_name(name: str) -> tuple[CafaAspect, Cafa5Subset | None]:
    try:
        return parse_cafa_dataset_name(name, spec=_CAFA5_SPEC), None
    except EmbeddingInputError:
        pass
    normalized = str(name).strip().lower().replace("-", "_")
    for prefix in ("cafa5:", "cafa:"):
        if normalized.startswith(prefix):
            normalized = normalized.partition(":")[2]
    for subset in CAFA5_TARGET_SUBSETS:
        prefix = f"cafa5_{subset}_"
        if normalized.startswith(prefix):
            aspect = normalized.removeprefix(prefix)
            if aspect in CAFA_ASPECTS:
                return aspect, subset
    supported = ", ".join(
        [f"cafa5_{aspect}" for aspect in CAFA_ASPECTS]
        + [f"cafa5_{subset}_{aspect}" for subset in CAFA5_TARGET_SUBSETS for aspect in CAFA_ASPECTS]
    )
    raise EmbeddingInputError(f"CAFA5 dataset name must be one of: {supported}.")


def _load_cafa5_subset_dataset(
    root: str | Path,
    *,
    aspect: CafaAspect,
    subset: Cafa5Subset,
    split: str | Sequence[str] | None,
    target: str | None,
    go_obo_path: str | Path | None,
    ia_path: str | Path | None,
    require_complete_subset_sequences: bool,
) -> ProteinDataset:
    expected_target = f"go_{aspect}"
    resolved_target = target or expected_target
    if resolved_target != expected_target:
        raise EmbeddingInputError(f"CAFA5 subset dataset only supports target {expected_target!r}.")

    root_path = resolve_cafa_root(Path(root).expanduser(), spec=_CAFA5_SPEC)
    sequence_path = root_path / "Train" / "train_sequences.fasta"
    terms_path = root_path / "Train" / "train_terms.tsv"
    target_terms_path = root_path / "Test (Targets)" / CAFA5_TARGET_SUBSETS[subset]
    sequences = _read_cafa5_sequences(root_path, sequence_path=sequence_path)
    train_labels = read_cafa_train_terms(terms_path, aspect=aspect, spec=_CAFA5_SPEC)
    test_labels = read_cafa_terms(
        target_terms_path,
        aspect=aspect,
        spec=_CAFA5_SPEC,
        file_label=target_terms_path.name,
    )
    ontology_path = resolve_cafa_go_obo_path(root_path, go_obo_path=go_obo_path, spec=_CAFA5_SPEC)
    information_accretion_path = resolve_cafa_ia_path(root_path, ia_path=ia_path, spec=_CAFA5_SPEC)
    known_terms_path = _resolve_cafa5_known_terms_path(root_path, subset=subset)
    terms_of_interest_path = _resolve_cafa5_terms_of_interest_path(root_path)

    train_examples = [
        ProteinExample(
            id=protein_id,
            sequence=sequences[protein_id],
            labels={resolved_target: labels},
            split="train",
            metadata=cafa_example_metadata(
                spec=_CAFA5_SPEC,
                aspect=aspect,
                labels=labels,
                terms_path=terms_path,
                sequence_path=sequence_path,
                ontology_path=ontology_path,
                ia_path=information_accretion_path,
            ),
        )
        for protein_id, labels in train_labels.items()
        if protein_id in sequences
    ]
    missing_test_ids = sorted(protein_id for protein_id in test_labels if protein_id not in sequences)
    _handle_missing_subset_sequences(
        missing_test_ids,
        subset=subset,
        root_path=root_path,
        require_complete_subset_sequences=require_complete_subset_sequences,
    )
    test_examples = [
        ProteinExample(
            id=protein_id,
            sequence=sequences[protein_id],
            labels={resolved_target: labels},
            split="test",
            metadata={
                **cafa_example_metadata(
                    spec=_CAFA5_SPEC,
                    aspect=aspect,
                    labels=labels,
                    terms_path=target_terms_path,
                    sequence_path=sequence_path,
                    ontology_path=ontology_path,
                    ia_path=information_accretion_path,
                ),
                "subset": subset,
                "evaluation_setting": _cafa5_setting_tag(subset),
                "target_terms_path": str(target_terms_path),
                **({"toi_path": str(terms_of_interest_path)} if terms_of_interest_path is not None else {}),
                **({"known_terms_path": str(known_terms_path)} if known_terms_path is not None else {}),
            },
        )
        for protein_id, labels in test_labels.items()
        if protein_id in sequences
    ]
    dataset = ProteinDataset([*train_examples, *test_examples])
    selected_splits = resolve_cafa_splits(split, spec=_CAFA5_SPEC)
    if selected_splits is None:
        return dataset
    return ProteinDataset(example for example in dataset.examples if example.split in selected_splits)


def _resolve_cafa5_known_terms_path(root: Path, *, subset: Cafa5Subset) -> Path | None:
    name = _CAFA5_SUBSET_KNOWN_FILES.get(subset)
    if name is None:
        return None
    path = root / "Test (Targets)" / name
    if not path.exists():
        raise EmbeddingInputError(f"CAFA5 partial-knowledge subset requires known terms file: {path}.")
    return path


def _resolve_cafa5_terms_of_interest_path(root: Path) -> Path | None:
    path = root / "Test (Targets)" / "toi_2025_03.tsv"
    if path.exists():
        return path
    return None


def _read_cafa5_sequences(root: Path, *, sequence_path: Path) -> dict[str, str]:
    sequences = read_cafa_fasta_sequences(sequence_path, spec=_CAFA5_SPEC)
    for candidate in _cafa5_test_sequence_candidates(root):
        if candidate.exists():
            sequences.update(read_cafa_fasta_sequences(candidate, spec=_CAFA5_SPEC))
    return sequences


def _cafa5_test_sequence_candidates(root: Path) -> tuple[Path, ...]:
    return (
        root / "Test" / "testsuperset.fasta",
        root / "Test (Targets)" / "testsuperset.fasta",
        root / "testsuperset.fasta",
    )


def _handle_missing_subset_sequences(
    missing_test_ids: Sequence[str],
    *,
    subset: Cafa5Subset,
    root_path: Path,
    require_complete_subset_sequences: bool,
) -> None:
    if not missing_test_ids:
        return
    candidates = ", ".join(str(path) for path in _cafa5_test_sequence_candidates(root_path))
    message = (
        f"CAFA5 subset {subset!r} is missing sequences for {len(missing_test_ids)} target proteins. "
        f"Add testsuperset.fasta at one of: {candidates}."
    )
    if require_complete_subset_sequences:
        raise EmbeddingInputError(message)
    LOGGER.warning("%s Skipping missing target proteins.", message)


def download_cafa5_dataset(root: str | Path, *, force: bool = False) -> list[Path]:
    """Download and unzip CAFA5 competition files with the Kaggle CLI."""
    return download_cafa_dataset(root, spec=_CAFA5_SPEC, force=force)


__all__ = [
    "CAFA5_COLLECTION_METADATA",
    "CAFA5_DATASETS",
    "CAFA5_TARGET_SUBSETS",
    "Cafa5Aspect",
    "Cafa5Subset",
    "download_cafa5_dataset",
    "load_cafa5_dataset",
]
