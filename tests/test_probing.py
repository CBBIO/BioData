from __future__ import annotations

import io
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

from CBBIO import (
    EmbeddingDependencyError,
    LinearProbe,
    MlpProbe,
    MmseqsRedundancyHit,
    MmseqsRedundancyReport,
    PredictionSpec,
    HashDatasetSplitter,
    HoldoutDatasetSplitter,
    ProbeBackend,
    ProbeBackendInput,
    ProbeBackendOutput,
    ProbeSpec,
    ProteinDataset,
    ProteinExample,
    ResidueDataset,
    ResidueExample,
    StratifiedDatasetSplitter,
    Task,
    CAFA5_TARGET_SUBSETS,
    get_dataset_collection,
    get_dataset_catalog_entry,
    download_dbptm_benchmark,
    download_cafa5_dataset,
    download_disprot_current_json,
    download_disprot_current_tsv,
    download_musitedeep_testdata,
    download_residue_source,
    filter_redundant_to_test_mmseqs,
    get_dbptm_benchmark,
    get_dtu_service,
    list_dbptm_benchmarks,
    list_dataset_catalog,
    list_dataset_collections,
    list_dtu_services,
    get_peer_task,
    get_residue_source,
    list_peer_native_datasets,
    list_peer_tasks,
    list_residue_dataset_catalog,
    list_residue_sources,
    load_biolip_dataset,
    load_cafa5_dataset,
    load_clean_dataset,
    load_dataset,
    load_dbptm_benchmark_archive,
    load_dbptm_benchmark_dataset,
    load_disprot_json,
    load_disprot_tsv,
    load_ec_dataset,
    load_ecbench_dataset,
    load_go_dataset,
    load_flip_csv,
    load_flip_dataset,
    load_go,
    load_interval_residue_tsv,
    load_musitedeep_fasta,
    load_musitedeep_testdata_dataset,
    load_peer_dataset,
    load_phosphoelm_dataset,
    load_residue_label_table,
    load_residue_label_csv,
    load_residue_source_dataset,
    read_ec_manifest,
    read_go_manifest,
    read_information_accretion_weights,
    run_task_on_layer,
    search_dataset_catalog,
    train_and_evaluate_residue_probe,
    TransferProbe,
)
from CBBIO.probing.metrics import binary_metrics
from CBBIO.probing.metrics import cafa6_weighted_fmax_mean
from CBBIO.probing.metrics import go_combined_protein_centric_metrics
from CBBIO.probing.metrics import go_fixed_threshold_protein_centric_metrics
from CBBIO.probing.metrics import go_protein_centric_metrics
from CBBIO.probing.metrics import go_weighted_protein_centric_metrics
from CBBIO.probing.metrics import multilabel_metrics
from CBBIO.probing.metrics import multiclass_metrics
from CBBIO.probing.metrics import spearmanr
from CBBIO.probing.sources.cafa6 import download_cafa6_dataset
from CBBIO.probing.sources.cafa6 import load_cafa6_dataset
from CBBIO.embeddings import EmbeddingInputError


torch = pytest.importorskip("torch")
peer_module = sys.modules[load_peer_dataset.__module__]
cafa_download_module = sys.modules[download_cafa6_dataset.__globals__["download_cafa_dataset"].__module__]


class _FixedProbeBackend(ProbeBackend):
    def __init__(self, output: ProbeBackendOutput) -> None:
        self.output = output
        self.data: ProbeBackendInput | None = None

    def fit_predict(self, data: ProbeBackendInput) -> ProbeBackendOutput:
        """Capture canonical input and return fixed predictions."""
        self.data = data
        return self.output


def test_hash_dataset_splitter_assigns_stable_protein_splits() -> None:
    examples = [
        ProteinExample(f"protein_{index}", "ACDE", {"active": index % 2}, "train")
        for index in range(12)
    ]
    splitter = HashDatasetSplitter(salt="test-split-v1")

    first = splitter.split_examples(examples)
    second = splitter.split_examples(examples)

    assert ProteinDataset(first).split_counts() == {"train": 10, "val": 1, "test": 1}
    assert {example.id: example.split for example in first} == {
        example.id: example.split for example in second
    }


def test_hash_dataset_splitter_rebuilds_residue_dataset() -> None:
    dataset = ResidueDataset(
        ResidueExample(
            f"residue_{index}",
            "ACDE",
            {"site": [0, 1, 0, 0]},
            "train",
        )
        for index in range(12)
    )

    split_dataset = HashDatasetSplitter(salt="residue-split-v1").split_dataset(dataset)

    assert isinstance(split_dataset, ResidueDataset)
    assert split_dataset.split_counts() == {"train": 10, "val": 1, "test": 1}


def test_stratified_dataset_splitter_balances_large_metadata_groups() -> None:
    examples = [
        ResidueExample(
            f"{family}_{index}",
            "ACDE",
            {"site": [0, 1, 0, 0]},
            "train",
            metadata={"family": family},
        )
        for family in ("kinase", "phosphatase")
        for index in range(10)
    ]

    split_examples = StratifiedDatasetSplitter(
        metadata_fields=("family",),
        salt="family-split-v1",
    ).split_examples(examples)
    families_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for example in split_examples:
        assert example.metadata is not None
        families_by_split[example.split].add(str(example.metadata["family"]))

    assert ResidueDataset(split_examples).split_counts() == {"train": 16, "val": 2, "test": 2}
    assert families_by_split == {
        "train": {"kinase", "phosphatase"},
        "val": {"kinase", "phosphatase"},
        "test": {"kinase", "phosphatase"},
    }


def test_stratified_dataset_splitter_uses_fallback_for_rare_groups() -> None:
    examples = [
        ResidueExample(
            f"{subfamily}_{index}",
            "ACDE",
            {"site": [0, 1, 0, 0]},
            "train",
            metadata={"family": "enzyme", "subfamily": subfamily},
        )
        for subfamily in ("a", "b")
        for index in range(6)
    ]

    split_examples = StratifiedDatasetSplitter(
        metadata_fields=("family", "subfamily"),
        fallback_metadata_fields=("family",),
        salt="rare-family-split-v1",
    ).split_examples(examples)

    assert ResidueDataset(split_examples).split_counts() == {"train": 10, "val": 1, "test": 1}


def test_stratified_dataset_splitter_handles_missing_metadata() -> None:
    examples = [
        ResidueExample(
            f"missing_{index}",
            "ACDE",
            {"site": [0, 1, 0, 0]},
            "train",
            metadata=None,
        )
        for index in range(12)
    ]

    split_examples = StratifiedDatasetSplitter(
        metadata_fields=("family",),
        salt="missing-metadata-split-v1",
    ).split_examples(examples)

    assert ResidueDataset(split_examples).split_counts() == {"train": 10, "val": 1, "test": 1}


def test_holdout_dataset_splitter_assigns_metadata_matches_to_test() -> None:
    examples = [
        ProteinExample(
            f"protein_{index}",
            "ACDE",
            {"active": index % 2},
            "train",
            metadata={"species": "heldout" if index < 3 else "trainable"},
        )
        for index in range(20)
    ]
    splitter = HoldoutDatasetSplitter(
        metadata_field="species",
        holdout_values=("heldout",),
        salt="species-holdout-v1",
    )

    first = splitter.split_examples(examples)
    second = splitter.split_examples(examples)
    test_ids = {example.id for example in first if example.split == "test"}

    assert test_ids == {"protein_0", "protein_1", "protein_2"}
    assert ProteinDataset(first).split_counts() == {"train": 15, "val": 2, "test": 3}
    assert {example.id: example.split for example in first} == {
        example.id: example.split for example in second
    }


def test_holdout_dataset_splitter_selects_metadata_classes_deterministically() -> None:
    examples = [
        ProteinExample(
            f"{species}_{index}",
            "ACDE",
            {"active": index % 2},
            "train",
            metadata={"species": species},
        )
        for species in ("human", "mouse", "yeast")
        for index in range(4)
    ]
    splitter = HoldoutDatasetSplitter(
        metadata_field="species",
        salt="random-species-holdout-v1",
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    )

    first = splitter.split_examples(examples)
    second = splitter.split_examples(examples)
    heldout_species = {
        str(example.metadata["species"])
        for example in first
        if example.metadata is not None and example.split == "test"
    }

    assert len(heldout_species) == 1
    assert ProteinDataset(first).split_counts() == {"train": 5, "val": 3, "test": 4}
    assert all(
        str(example.metadata["species"]) in heldout_species
        for example in first
        if example.metadata is not None and example.split == "test"
    )
    assert {example.id: example.split for example in first} == {
        example.id: example.split for example in second
    }


def test_holdout_dataset_splitter_selects_rarest_metadata_classes_first() -> None:
    examples = [
        *[
            ProteinExample(
                f"human_{index}",
                "ACDE",
                {"active": index % 2},
                "train",
                metadata={"species": "human"},
            )
            for index in range(8)
        ],
        *[
            ProteinExample(
                f"mouse_{index}",
                "ACDE",
                {"active": index % 2},
                "train",
                metadata={"species": "mouse"},
            )
            for index in range(3)
        ],
        ProteinExample("yeast_0", "ACDE", {"active": 1}, "train", metadata={"species": "yeast"}),
    ]

    split_examples = HoldoutDatasetSplitter(
        metadata_field="species",
        holdout_strategy="rarest_first",
        salt="rarest-species-holdout-v1",
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    ).split_examples(examples)
    test_species = {
        str(example.metadata["species"])
        for example in split_examples
        if example.metadata is not None and example.split == "test"
    }

    assert test_species == {"mouse", "yeast"}
    assert ProteinDataset(split_examples).split_counts() == {"train": 5, "val": 3, "test": 4}


def test_holdout_dataset_splitter_can_disable_class_holdout() -> None:
    examples = [
        ProteinExample(
            f"protein_{index}",
            "ACDE",
            {"active": index % 2},
            "train",
            metadata={"species": "human" if index < 6 else "mouse"},
        )
        for index in range(12)
    ]

    split_examples = HoldoutDatasetSplitter(
        metadata_field="species",
        holdout_strategy="none",
        salt="no-class-holdout-v1",
        ratios={"train": 0.5, "val": 0.25, "test": 0.25},
    ).split_examples(examples)

    assert ProteinDataset(split_examples).split_counts() == {"train": 6, "val": 3, "test": 3}


def test_custom_probe_backend_uses_canonical_protein_splits_and_metrics() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train_0", "ACDE", {"active": 0}, "train"),
            ProteinExample("train_1", "FGHI", {"active": 1}, "train"),
            ProteinExample("test_0", "KLMN", {"active": 0}, "test"),
            ProteinExample("test_1", "PQRS", {"active": 1}, "test"),
        ]
    )
    backend = _FixedProbeBackend(
        ProbeBackendOutput(
            predictions={"test_0": 0, "test_1": 1},
            scores={"test_0": 0.1, "test_1": 0.9},
        )
    )
    task = Task(
        name="custom_binary",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=backend,
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "train_0": [0.0, 0.1],
            "train_1": [0.9, 1.0],
            "test_0": [0.1, 0.2],
            "test_1": [0.8, 0.9],
        },
    )

    assert backend.data is not None
    assert backend.data.train_ids == ("train_0", "train_1")
    assert backend.data.test_ids == ("test_0", "test_1")
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.metrics["auroc"] == pytest.approx(1.0)
    assert result.predictions == {"test_0": 0, "test_1": 1}


def test_custom_probe_backend_applies_residue_masks_before_evaluation() -> None:
    dataset = ResidueDataset(
        [
            ResidueExample("train", "ACD", {"site": [0, 1, 0]}, "train"),
            ResidueExample(
                "test",
                "EFG",
                {"site": [0, 0, 1]},
                "test",
                mask=[True, False, True],
            ),
        ]
    )
    backend = _FixedProbeBackend(
        ProbeBackendOutput(
            predictions={"test": [0, 1, 1]},
            scores={"test": [0.1, 0.9, 0.8]},
        )
    )
    task = Task(
        name="custom_residue_binary",
        dataset=dataset,
        prediction=PredictionSpec(target="site", objective="binary", level="residue"),
        probe=backend,
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "train": [[0.0], [1.0], [0.0]],
            "test": [[0.0], [1.0], [1.0]],
        },
    )

    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.predictions == {"test:1": 0, "test:3": 1}
    assert result.scores == {"test:1": 0.1, "test:3": 0.8}


def test_custom_binary_probe_raises_when_scores_are_missing() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train", "ACDE", {"active": 0}, "train"),
            ProteinExample("test", "FGHI", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="custom_binary_without_scores",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=_FixedProbeBackend(ProbeBackendOutput(predictions={"test": 1})),
    )

    with pytest.raises(EmbeddingInputError, match="scores"):
        run_task_on_layer(
            task=task,
            embeddings={"train": [0.0], "test": [1.0]},
        )


def test_custom_regression_probe_normalizes_predictions_and_metrics() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train", "ACDE", {"value": 1.0}, "train"),
            ProteinExample("test", "FGHI", {"value": 2.5}, "test"),
        ]
    )
    task = Task(
        name="custom_regression",
        dataset=dataset,
        prediction=PredictionSpec(target="value", objective="regression"),
        probe=_FixedProbeBackend(
            ProbeBackendOutput(predictions={"test": "2.5"})
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={"train": [0.0], "test": [1.0]},
    )

    assert result.predictions == {"test": 2.5}
    assert result.metrics["mae"] == pytest.approx(0.0)


def test_custom_multiclass_probe_returns_canonical_class_names() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train_0", "ACDE", {"class": "alpha"}, "train"),
            ProteinExample("train_1", "FGHI", {"class": "beta"}, "train"),
            ProteinExample("test", "KLMN", {"class": "beta"}, "test"),
        ]
    )
    task = Task(
        name="custom_multiclass",
        dataset=dataset,
        prediction=PredictionSpec(target="class", objective="multiclass"),
        probe=_FixedProbeBackend(ProbeBackendOutput(predictions={"test": 1})),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={"train_0": [0.0], "train_1": [1.0], "test": [1.0]},
    )

    assert result.predictions == {"test": "beta"}
    assert result.metrics["accuracy"] == pytest.approx(1.0)


def test_transfer_probe_scores_multilabel_classes_by_inverse_cosine_distance() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("near", "ACDE", {"terms": ["a"]}, "train"),
            ProteinExample("far", "FGHI", {"terms": ["b"]}, "train"),
            ProteinExample("test", "KLMN", {"terms": ["a"]}, "test"),
        ]
    )
    task = Task(
        name="transfer_multilabel",
        dataset=dataset,
        prediction=PredictionSpec(
            target="terms",
            objective="multilabel",
            classes=("a", "b"),
        ),
        probe=TransferProbe(k=2, threshold=0.6),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "near": [0.9, 0.4358898944],
            "far": [0.8, 0.6],
            "test": [1.0, 0.0],
        },
    )

    assert result.scores == {"test": pytest.approx([2.0 / 3.0, 1.0 / 3.0])}
    assert result.predictions == {"test": ["a"]}


def test_transfer_probe_uses_half_threshold_for_default_multilabel_predictions() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("near", "ACDE", {"terms": ["a"]}, "train"),
            ProteinExample("far", "FGHI", {"terms": ["b"]}, "train"),
            ProteinExample("test", "KLMN", {"terms": ["a"]}, "test"),
        ]
    )
    task = Task(
        name="transfer_default_multilabel",
        dataset=dataset,
        prediction=PredictionSpec(
            target="terms",
            objective="multilabel",
            classes=("a", "b"),
        ),
        probe=TransferProbe(k=2),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "near": [0.9, 0.4358898944],
            "far": [0.8, 0.6],
            "test": [1.0, 0.0],
        },
    )

    assert result.scores == {"test": pytest.approx([2.0 / 3.0, 1.0 / 3.0])}
    assert result.predictions == {"test": ["a"]}


def test_transfer_probe_uses_ten_neighbors_for_knn_selection() -> None:
    dataset = ProteinDataset(
        [
            *[
                ProteinExample(f"positive_{index}", "ACDE", {"active": 1}, "train")
                for index in range(10)
            ],
            ProteinExample("negative", "FGHI", {"active": 0}, "train"),
            ProteinExample("test", "KLMN", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="transfer_default_k",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=TransferProbe(neighbor_selection="knn"),
    )
    embeddings = {
        **{
            f"positive_{index}": [1.0, float(index + 1) / 100.0]
            for index in range(10)
        },
        "negative": [-1.0, 0.0],
        "test": [1.0, 0.0],
    }

    result = run_task_on_layer(task=task, embeddings=embeddings)

    assert result.predictions == {"test": 1}
    assert result.scores == {"test": pytest.approx(1.0)}


def test_transfer_probe_returns_neighbor_metadata_when_requested() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("near", "ACDE", {"active": 1}, "train"),
            ProteinExample("far", "FGHI", {"active": 0}, "train"),
            ProteinExample("test", "KLMN", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="transfer_neighbors",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=TransferProbe(
            distance="euclidean",
            neighbor_selection="knn",
            k=2,
            scoring="voting",
            return_neighbors=True,
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "near": [0.1, 0.0],
            "far": [2.0, 0.0],
            "test": [0.0, 0.0],
        },
    )

    assert result.metadata is not None
    neighbors = result.metadata["transfer_neighbors"]["test"]
    assert neighbors[0] == {"id": "near", "distance": pytest.approx(0.1), "weight": 1.0, "label": 1}
    assert neighbors[1] == {"id": "far", "distance": pytest.approx(2.0), "weight": 1.0, "label": 0}


def test_transfer_probe_uses_faiss_cpu_for_knn_selection_when_requested() -> None:
    pytest.importorskip("faiss")
    dataset = ProteinDataset(
        [
            ProteinExample("near", "ACDE", {"active": 1}, "train"),
            ProteinExample("far", "FGHI", {"active": 0}, "train"),
            ProteinExample("test", "KLMN", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="transfer_faiss_cpu",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=TransferProbe(
            neighbor_selection="knn",
            k=1,
            search_backend="faiss_cpu",
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "near": [1.0, 0.0],
            "far": [0.0, 1.0],
            "test": [0.9, 0.1],
        },
    )

    assert result.scores == {"test": pytest.approx(1.0)}
    assert result.predictions == {"test": 1}


def test_transfer_probe_scores_all_neighbors_with_unweighted_votes() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("positive", "ACDE", {"active": 1}, "train"),
            ProteinExample("negative", "FGHI", {"active": 0}, "train"),
            ProteinExample("test", "KLMN", {"active": 0}, "test"),
        ]
    )
    task = Task(
        name="transfer_all_voting",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=TransferProbe(
            distance="euclidean",
            neighbor_selection="all",
            scoring="voting",
            threshold=0.6,
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "positive": [0.0, 0.0],
            "negative": [10.0, 0.0],
            "test": [0.1, 0.0],
        },
    )

    assert result.scores == {"test": pytest.approx(0.5)}
    assert result.predictions == {"test": 0}


def test_transfer_probe_selects_neighbors_by_distance_cutoff() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("near", "ACDE", {"active": 1}, "train"),
            ProteinExample("far", "FGHI", {"active": 0}, "train"),
            ProteinExample("test", "KLMN", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="transfer_cutoff",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=TransferProbe(
            distance="euclidean",
            neighbor_selection="cutoff_distance",
            distance_cutoff=0.2,
            scoring="voting",
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "near": [0.1, 0.0],
            "far": [10.0, 0.0],
            "test": [0.0, 0.0],
        },
    )

    assert result.scores == {"test": pytest.approx(1.0)}
    assert result.predictions == {"test": 1}


def test_transfer_probe_raises_when_cutoff_selects_no_neighbors() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train", "ACDE", {"active": 1}, "train"),
            ProteinExample("test", "KLMN", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="transfer_empty_cutoff",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=TransferProbe(
            distance="euclidean",
            neighbor_selection="cutoff_distance",
            distance_cutoff=0.1,
        ),
    )

    with pytest.raises(EmbeddingInputError, match="no neighbors"):
        run_task_on_layer(
            task=task,
            embeddings={"train": [1.0, 0.0], "test": [0.0, 0.0]},
        )


def test_transfer_probe_raises_when_k_is_not_positive() -> None:
    with pytest.raises(EmbeddingInputError, match="k"):
        TransferProbe(k=0)


def test_transfer_probe_raises_when_accelerated_backend_is_used_without_knn() -> None:
    with pytest.raises(EmbeddingInputError, match="knn"):
        TransferProbe(search_backend="faiss_cpu", neighbor_selection="all")


def test_transfer_probe_raises_when_embedding_is_zero_vector() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train", "ACDE", {"active": 1}, "train"),
            ProteinExample("test", "KLMN", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="transfer_zero_vector",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=TransferProbe(),
    )

    with pytest.raises(EmbeddingInputError, match="zero vectors"):
        run_task_on_layer(
            task=task,
            embeddings={"train": [0.0, 0.0], "test": [1.0, 0.0]},
        )


@pytest.mark.parametrize(
    "probe",
    [
        pytest.param(LinearProbe(epochs=2), id="linear"),
        pytest.param(MlpProbe(epochs=2, hidden_dim=4), id="mlp"),
    ],
)
def test_builtin_probe_backends_use_the_canonical_task_interface(
    probe: ProbeBackend,
) -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train_0", "ACDE", {"active": 0}, "train"),
            ProteinExample("train_1", "FGHI", {"active": 1}, "train"),
            ProteinExample("test_0", "KLMN", {"active": 0}, "test"),
            ProteinExample("test_1", "PQRS", {"active": 1}, "test"),
        ]
    )
    task = Task(
        name="built_in_backend",
        dataset=dataset,
        prediction=PredictionSpec(target="active", objective="binary"),
        probe=probe,
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "train_0": [0.0, 0.1],
            "train_1": [0.9, 1.0],
            "test_0": [0.1, 0.2],
            "test_1": [0.8, 0.9],
        },
    )

    assert set(result.predictions) == {"test_0", "test_1"}
    assert "accuracy" in result.metrics


def test_spearmanr_handles_ranks_reverse_ranks_and_ties() -> None:
    assert spearmanr([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert spearmanr([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    assert spearmanr([1.0, 1.0, 2.0, 3.0], [2.0, 2.0, 1.0, 4.0]) == pytest.approx(1.0 / 3.0)
    assert spearmanr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_binary_task_uses_dataset_target_and_probe_objective() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("n0", "ACDE", {"is_enzyme": 0}, "train"),
            ProteinExample("n1", "ACDE", {"is_enzyme": 0}, "train"),
            ProteinExample("p0", "ACDE", {"is_enzyme": 1}, "train"),
            ProteinExample("p1", "ACDE", {"is_enzyme": 1}, "train"),
            ProteinExample("nt", "ACDE", {"is_enzyme": 0}, "test"),
            ProteinExample("pt", "ACDE", {"is_enzyme": 1}, "test"),
        ]
    )
    embeddings = {
        "n0": [0.0, 0.0],
        "n1": [0.0, 0.5],
        "p0": [2.0, 2.0],
        "p1": [2.0, 2.5],
        "nt": [0.0, 0.2],
        "pt": [2.5, 2.0],
    }
    task = Task(
        name="enzyme_binary",
        dataset=dataset,
        prediction=PredictionSpec(target="is_enzyme", objective="binary"),
        probe=ProbeSpec(epochs=200, learning_rate=0.1, seed=11),
    )

    result = run_task_on_layer(
        task=task,
        embeddings=embeddings,
        model_reference="unit/model",
        layer_index=3,
    )

    assert result.task_name == "enzyme_binary"
    assert result.model_reference == "unit/model"
    assert result.layer_index == 3
    assert result.train_count == 4
    assert result.test_count == 2
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["f1"] == 1.0
    assert result.predictions == {"nt": 0, "pt": 1}
    assert result.scores is not None
    assert set(result.scores) == {"nt", "pt"}


def test_binary_metrics_include_imbalance_aware_scores() -> None:
    metrics = binary_metrics(
        y_true=[1, 1, 0, 0],
        y_pred=[1, 0, 0, 0],
        y_score=[0.9, 0.8, 0.4, 0.1],
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["mcc"] == pytest.approx(1 / 3**0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_multilabel_metrics_report_micro_and_macro_f1() -> None:
    metrics = multilabel_metrics(
        y_true=[[1, 0], [0, 1], [1, 1]],
        y_pred=[[1, 0], [1, 0], [1, 1]],
        y_score=[[0.9, 0.1], [0.6, 0.4], [0.8, 0.7]],
    )

    assert metrics["exact_match"] == pytest.approx(2.0 / 3.0)
    assert metrics["f1"] == pytest.approx(0.75)
    assert metrics["micro_f1"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx((0.8 + 2.0 / 3.0) / 2.0)
    assert metrics["weighted_f1"] == pytest.approx((0.8 * 2.0 + (2.0 / 3.0) * 2.0) / 4.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["fmax"] == pytest.approx(8.0 / 9.0)
    assert metrics["fmax_threshold"] == pytest.approx(0.11)
    assert metrics["precision_at_fmax"] == pytest.approx(0.8)
    assert metrics["recall_at_fmax"] == pytest.approx(1.0)


def test_multilabel_metrics_do_not_propagate_hierarchical_labels() -> None:
    metrics = multilabel_metrics(
        y_true=[[1, 0]],
        y_pred=[[0, 1]],
        y_score=[[-1.0, 0.9]],
    )

    assert metrics["exact_match"] == pytest.approx(0.0)
    assert metrics["f1"] == pytest.approx(0.0)
    assert metrics["fmax"] == pytest.approx(0.0)


def test_multiclass_metrics_report_weighted_f1() -> None:
    metrics = multiclass_metrics(
        y_true=[0, 0, 0, 1, 2],
        y_pred=[0, 0, 1, 1, 1],
        class_count=3,
    )

    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["macro_f1"] == pytest.approx(((4.0 / 5.0) + (1.0 / 2.0) + 0.0) / 3.0)
    assert metrics["weighted_f1"] == pytest.approx(((4.0 / 5.0) * 3.0 + (1.0 / 2.0)) / 5.0)


def test_multilabel_task_trains_probe_with_label_sets() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("none", "ACDE", {"terms": []}, "train"),
            ProteinExample("a0", "ACDE", {"terms": ["a"]}, "train"),
            ProteinExample("a1", "ACDE", {"terms": ["a"]}, "train"),
            ProteinExample("b0", "ACDE", {"terms": ["b"]}, "train"),
            ProteinExample("b1", "ACDE", {"terms": ["b"]}, "train"),
            ProteinExample("ab0", "ACDE", {"terms": ["a", "b"]}, "train"),
            ProteinExample("ab1", "ACDE", {"terms": ["a", "b"]}, "train"),
            ProteinExample("test_a", "ACDE", {"terms": ["a"]}, "test"),
            ProteinExample("test_b", "ACDE", {"terms": ["b"]}, "test"),
            ProteinExample("test_ab", "ACDE", {"terms": ["a", "b"]}, "test"),
        ]
    )
    embeddings = {
        "none": [0.0, 0.0],
        "a0": [2.0, 0.0],
        "a1": [2.5, 0.1],
        "b0": [0.0, 2.0],
        "b1": [0.1, 2.5],
        "ab0": [2.0, 2.0],
        "ab1": [2.5, 2.5],
        "test_a": [3.0, 0.0],
        "test_b": [0.0, 3.0],
        "test_ab": [3.0, 3.0],
    }
    task = Task(
        name="multilabel_terms",
        dataset=dataset,
        prediction=PredictionSpec(target="terms", objective="multilabel"),
        probe=ProbeSpec(epochs=300, learning_rate=0.1, seed=17),
    )

    result = run_task_on_layer(task=task, embeddings=embeddings)

    assert result.metrics["exact_match"] == pytest.approx(1.0)
    assert result.metrics["micro_f1"] == pytest.approx(1.0)
    assert result.metrics["f1"] == pytest.approx(1.0)
    assert result.metrics["macro_f1"] == pytest.approx(1.0)
    assert result.metrics["weighted_f1"] == pytest.approx(1.0)
    assert result.predictions == {
        "test_a": ["a"],
        "test_b": ["b"],
        "test_ab": ["a", "b"],
    }
    assert result.scores is not None
    assert set(result.scores) == {"test_a", "test_b", "test_ab"}


def test_probe_standardizes_embedding_features_from_train_split() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("n0", "ACDE", {"label": 0}, "train"),
            ProteinExample("n1", "ACDE", {"label": 0}, "train"),
            ProteinExample("p0", "ACDE", {"label": 1}, "train"),
            ProteinExample("p1", "ACDE", {"label": 1}, "train"),
            ProteinExample("nt", "ACDE", {"label": 0}, "test"),
            ProteinExample("pt", "ACDE", {"label": 1}, "test"),
        ]
    )
    embeddings = {
        "n0": [1_000_000.0, 1_000_000.0],
        "n1": [1_000_000.0, 1_005_000.0],
        "p0": [1_000_000.0, 1_020_000.0],
        "p1": [1_000_000.0, 1_025_000.0],
        "nt": [1_000_000.0, 1_002_000.0],
        "pt": [1_000_000.0, 1_022_000.0],
    }
    task = Task(
        name="scaled_binary",
        dataset=dataset,
        prediction=PredictionSpec(target="label", objective="binary"),
        probe=ProbeSpec(epochs=100, learning_rate=0.1, seed=17),
    )

    result = run_task_on_layer(task=task, embeddings=embeddings)

    assert result.metrics["accuracy"] == 1.0
    assert result.predictions == {"nt": 0, "pt": 1}


def test_regression_task_trains_linear_probe() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("r0", "ACDE", {"score": 0.0}, "train"),
            ProteinExample("r1", "ACDE", {"score": 1.0}, "train"),
            ProteinExample("r2", "ACDE", {"score": 2.0}, "train"),
            ProteinExample("r3", "ACDE", {"score": 3.0}, "train"),
            ProteinExample("rt0", "ACDE", {"score": 1.5}, "test"),
            ProteinExample("rt1", "ACDE", {"score": 2.5}, "test"),
        ]
    )
    embeddings = {
        "r0": [0.0],
        "r1": [1.0],
        "r2": [2.0],
        "r3": [3.0],
        "rt0": [1.5],
        "rt1": [2.5],
    }
    task = Task(
        name="score_regression",
        dataset=dataset,
        prediction=PredictionSpec(target="score", objective="regression"),
        probe=ProbeSpec(epochs=300, learning_rate=0.05, seed=3),
    )

    result = run_task_on_layer(task=task, embeddings=embeddings)

    assert result.metrics["mae"] < 0.05
    assert result.metrics["rmse"] < 0.05
    assert result.metrics["spearmanr"] == pytest.approx(1.0)


def test_multiclass_task_returns_class_names() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("a0", "ACDE", {"family": "a"}, "train"),
            ProteinExample("a1", "ACDE", {"family": "a"}, "train"),
            ProteinExample("b0", "ACDE", {"family": "b"}, "train"),
            ProteinExample("b1", "ACDE", {"family": "b"}, "train"),
            ProteinExample("c0", "ACDE", {"family": "c"}, "train"),
            ProteinExample("c1", "ACDE", {"family": "c"}, "train"),
            ProteinExample("at", "ACDE", {"family": "a"}, "test"),
            ProteinExample("bt", "ACDE", {"family": "b"}, "test"),
            ProteinExample("ct", "ACDE", {"family": "c"}, "test"),
        ]
    )
    embeddings = {
        "a0": [3.0, 0.0, 0.0],
        "a1": [2.5, 0.0, 0.0],
        "b0": [0.0, 3.0, 0.0],
        "b1": [0.0, 2.5, 0.0],
        "c0": [0.0, 0.0, 3.0],
        "c1": [0.0, 0.0, 2.5],
        "at": [2.8, 0.0, 0.0],
        "bt": [0.0, 2.8, 0.0],
        "ct": [0.0, 0.0, 2.8],
    }
    task = Task(
        name="family_multiclass",
        dataset=dataset,
        prediction=PredictionSpec(target="family", objective="multiclass"),
        probe=ProbeSpec(epochs=250, learning_rate=0.1, seed=5),
    )

    result = run_task_on_layer(task=task, embeddings=embeddings)

    assert result.metrics["accuracy"] == 1.0
    assert result.predictions == {"at": "a", "bt": "b", "ct": "c"}


def test_dataset_owns_labels_and_prediction_selects_target() -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("p0", "ACDE", {"other": 1}, "train"),
            ProteinExample("p1", "ACDE", {"other": 0}, "test"),
        ]
    )
    task = Task(
        name="missing_target",
        dataset=dataset,
        prediction=PredictionSpec(target="is_enzyme", objective="binary"),
    )

    with pytest.raises(EmbeddingInputError, match="Target 'is_enzyme' is missing"):
        run_task_on_layer(
            task=task,
            embeddings={"p0": [0.0], "p1": [1.0]},
        )


def test_residue_level_task_trains_probe_over_valid_positions() -> None:
    dataset = ResidueDataset(
        [
            ResidueExample("r0", "ACDE", {"site": [0, 0, 1, 1]}, "train"),
            ResidueExample("r1", "ACDE", {"site": [0, 1, 1, 0]}, "train", mask=[True, True, True, False]),
            ResidueExample("rt", "ACDE", {"site": [0, 1, 1, 0]}, "test"),
        ]
    )
    embeddings = {
        "r0": [[0.0], [0.2], [2.0], [2.2]],
        "r1": [[0.1], [2.1], [2.3], [0.0]],
        "rt": [[0.1], [2.0], [2.2], [0.0]],
    }
    task = Task(
        name="residue_site",
        dataset=dataset,
        prediction=PredictionSpec(target="site", objective="binary", level="residue"),
        probe=ProbeSpec(epochs=200, learning_rate=0.1, seed=13),
    )

    result = run_task_on_layer(task=task, embeddings=embeddings, layer_index=2)

    assert result.layer_index == 2
    assert result.train_count == 2
    assert result.test_count == 1
    assert result.metrics["accuracy"] == 1.0
    assert result.predictions == {"rt:1": 0, "rt:2": 1, "rt:3": 1, "rt:4": 0}


def test_filter_redundant_to_test_mmseqs_removes_train_and_val_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train_high", "ACDEFG", {"target": 1}, "train"),
            ProteinExample("val_boundary", "ACDEYY", {"target": 0}, "val"),
            ProteinExample("train_keep", "YYYYYY", {"target": 0}, "train"),
            ProteinExample("test_ref", "ACDEFG", {"target": 1}, "test"),
        ]
    )
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        Path(command[4]).write_text(
            "\n".join(
                [
                    "train_high\ttest_ref\t100.0\t1.0\t1.0\t6\t6\t6\t1e-20\t99",
                    "val_boundary\ttest_ref\t30.0\t0.8\t0.8\t5\t6\t6\t1e-5\t30",
                    "train_keep\ttest_ref\t29.9\t1.0\t1.0\t6\t6\t6\t1e-4\t20",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("CBBIO.probing.datasets.shutil.which", lambda name: "/usr/bin/mmseqs" if name == "mmseqs" else None)
    monkeypatch.setattr("CBBIO.probing.datasets.subprocess.run", fake_run)

    filtered, report = filter_redundant_to_test_mmseqs(dataset, cutoff=30.0, min_coverage=0.8, threads=2)

    assert isinstance(report, MmseqsRedundancyReport)
    assert [example.id for example in filtered.examples] == ["train_keep", "test_ref"]
    assert report.removed_count == 2
    assert report.retained_count == 2
    assert report.reference_count == 1
    assert [(hit.removed_id, hit.reference_id, hit.percent_identity) for hit in report.removed] == [
        ("train_high", "test_ref", 100.0),
        ("val_boundary", "test_ref", 30.0),
    ]
    assert isinstance(report.removed[0], MmseqsRedundancyHit)
    command = commands[0]
    assert command[:2] == ["/usr/bin/mmseqs", "easy-search"]
    assert "--min-seq-id" in command
    assert command[command.index("--min-seq-id") + 1] == "0.3"
    assert "-c" in command
    assert command[command.index("-c") + 1] == "0.8"
    assert "--cov-mode" in command
    assert command[command.index("--cov-mode") + 1] == "0"
    assert "--threads" in command
    assert command[command.index("--threads") + 1] == "2"


def test_filter_redundant_to_test_mmseqs_keeps_hits_below_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train_short_hit", "ACDEFG", {"target": 1}, "train"),
            ProteinExample("test_ref", "ACDEFG", {"target": 1}, "test"),
        ]
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[4]).write_text(
            "train_short_hit\ttest_ref\t100.0\t0.79\t1.0\t4\t6\t6\t1e-10\t42\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("CBBIO.probing.datasets.shutil.which", lambda name: "/usr/bin/mmseqs" if name == "mmseqs" else None)
    monkeypatch.setattr("CBBIO.probing.datasets.subprocess.run", fake_run)

    filtered, report = filter_redundant_to_test_mmseqs(dataset, cutoff=30.0, min_coverage=0.8)

    assert [example.id for example in filtered.examples] == ["train_short_hit", "test_ref"]
    assert report.removed_count == 0
    assert report.removed == ()


def test_filter_redundant_to_test_mmseqs_preserves_residue_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = ResidueDataset(
        [
            ResidueExample("remove_me", "STYK", {"site": [1, 0, 0, 1]}, "train", mask=[True, True, True, True]),
            ResidueExample("keep_me", "AAAA", {"site": [0, 0, 0, 0]}, "train", mask=[True, False, True, True]),
            ResidueExample("test_ref", "STYK", {"site": [1, 0, 0, 1]}, "test", mask=[True, True, True, True]),
        ]
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[4]).write_text(
            "remove_me\ttest_ref\t100.0\t1.0\t1.0\t4\t4\t4\t1e-12\t80\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("CBBIO.probing.datasets.shutil.which", lambda name: "/usr/bin/mmseqs" if name == "mmseqs" else None)
    monkeypatch.setattr("CBBIO.probing.datasets.subprocess.run", fake_run)

    filtered, report = filter_redundant_to_test_mmseqs(dataset, cutoff=30.0, min_coverage=0.8)

    assert isinstance(filtered, ResidueDataset)
    assert [example.id for example in filtered.examples] == ["keep_me", "test_ref"]
    assert filtered.target_values("site") == {"keep_me": [0, 0, 0, 0], "test_ref": [1, 0, 0, 1]}
    assert filtered.examples[0].mask == [True, False, True, True]
    assert report.removed_count == 1


def test_filter_redundant_to_test_mmseqs_requires_mmseqs(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = ProteinDataset(
        [
            ProteinExample("train", "ACDE", {"target": 0}, "train"),
            ProteinExample("test", "ACDE", {"target": 1}, "test"),
        ]
    )
    monkeypatch.setattr("CBBIO.probing.datasets.shutil.which", lambda _name: None)

    with pytest.raises(EmbeddingDependencyError, match="MMseqs2"):
        filter_redundant_to_test_mmseqs(dataset)


def test_filter_redundant_to_test_mmseqs_requires_reference_split(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = ProteinDataset([ProteinExample("train", "ACDE", {"target": 0}, "train")])
    monkeypatch.setattr("CBBIO.probing.datasets.shutil.which", lambda name: "/usr/bin/mmseqs" if name == "mmseqs" else None)

    with pytest.raises(EmbeddingInputError, match="no examples in reference split"):
        filter_redundant_to_test_mmseqs(dataset)


def test_load_flip_csv_translates_peer_splits(tmp_path) -> None:
    csv_path = tmp_path / "two_vs_rest.csv"
    csv_path.write_text(
        "\n".join(
            [
                "sequence,target,set,validation",
                "AAAA,1.0,train,False",
                "CCCC,2.0,train,True",
                "DDDD,3.0,test,False",
                "EEEE,4.0,train,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_flip_csv(csv_path)

    assert [example.id for example in dataset.examples] == ["0", "3", "1", "2"]
    assert [example.split for example in dataset.examples] == ["train", "train", "val", "test"]
    assert dataset.target_values("target") == {"0": 1.0, "3": 4.0, "1": 2.0, "2": 3.0}


def test_load_flip_dataset_uses_named_split_and_mutation_region(tmp_path) -> None:
    split_dir = tmp_path / "aav" / "splits"
    split_dir.mkdir(parents=True)
    sequence = "A" * 474 + "BCDE" + "F" * 200
    (split_dir / "two_vs_many.csv").write_text(
        "\n".join(
            [
                "sequence,target,set,validation",
                f"{sequence},1.0,train,False",
                f"{sequence},2.0,test,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_flip_dataset(
        tmp_path,
        name="aav",
        split="two_vs_many",
        keep_mutation_region=True,
    )

    assert [example.sequence[:4] for example in dataset.examples] == ["BCDE", "BCDE"]
    assert {len(example.sequence) for example in dataset.examples} == {200}


def test_load_flip_dataset_rejects_unknown_split(tmp_path) -> None:
    with pytest.raises(EmbeddingInputError, match="Unsupported FLIP split"):
        load_flip_dataset(tmp_path, name="gb1", split="two_vs_many")


def test_peer_task_registry_contains_full_peer_table_metadata() -> None:
    tasks = list_peer_tasks()

    assert len(tasks) == 17
    assert get_peer_task("GB1").split_counts == (381, 43, 8309)
    assert get_peer_task("Thermo").name == "thermostability"
    assert get_peer_task("SSP").level == "residue"
    assert get_peer_task("Cont").level == "residue_pair"
    assert get_peer_task("BDB").category == "protein_ligand_interaction_prediction"
    assert [task.name for task in list_peer_tasks(level="protein_ligand")] == ["pdbbind", "bindingdb"]


def test_load_residue_label_csv_supports_ptm_like_binary_labels(tmp_path) -> None:
    csv_path = tmp_path / "ptm.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,sequence,labels,split,mask",
                "p0,STYK,1001,train,1111",
                "p1,STYK,0 1 0 0,test,1110",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_residue_label_csv(csv_path, target="phosphorylation", mask_field="mask")

    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 1}
    assert dataset.target_values("phosphorylation") == {"p0": [1, 0, 0, 1], "p1": [0, 1, 0, 0]}
    assert dataset.examples[1].mask == [True, True, True, False]


def test_load_residue_label_csv_supports_multiclass_strings_and_json(tmp_path) -> None:
    csv_path = tmp_path / "ssp.csv"
    csv_path.write_text(
        "\n".join(
            [
                "id,sequence,labels,split",
                "p0,ACDE,HECC,train",
                "p1,ACDE,\"[\"\"H\"\", \"\"E\"\", \"\"C\"\", \"\"C\"\"]\",test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_residue_label_csv(csv_path, target="secondary_structure")

    assert dataset.target_values("secondary_structure") == {
        "p0": ["H", "E", "C", "C"],
        "p1": ["H", "E", "C", "C"],
    }


def test_residue_label_csv_validates_label_length(tmp_path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(
        "\n".join(["id,sequence,labels,split", "p0,ACDE,101,train"]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(EmbeddingInputError, match="does not match sequence length"):
        load_residue_label_csv(csv_path)


def test_residue_dataset_catalog_lists_ptm_and_binding_sources() -> None:
    ptm = {item.name for item in list_residue_dataset_catalog(category="ptm")}
    binding = {item.name for item in list_residue_dataset_catalog(category="binding")}

    assert {"dbptm", "musitedeep"}.issubset(ptm)
    assert {"biolip", "metalpdb", "scannet_binding"}.issubset(binding)


def _write_ec_dataset_layout(root: Path) -> None:
    ec_root = root / "ec_main_head"
    manifest_dir = ec_root / "manifests"
    manifest_dir.mkdir(parents=True)
    for track in ("head", "main", "full"):
        (manifest_dir / f"ec_full_{track}.json").write_text(
            json.dumps({"task": "ec_full", "track": track, "split_counts": {"train": 2}})
            + "\n",
            encoding="utf-8",
        )
        for split in ("train", "val", "test"):
            split_dir = ec_root / "tasks" / "ec_full" / track
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
    records = {
        ("head", "train"): [
            _ec_jsonl_record(
                "ec_full:head:p1",
                "ACDE",
                ["1.1.1.1"],
                ec_level1=["1"],
                ec_level2=["1.1"],
                ec_level3=["1.1.1"],
            ),
            _ec_jsonl_record(
                "ec_full:head:p2",
                "FGHI",
                ["2.7.7.4", "3.5.4.10"],
                ec_level1=["2", "3"],
                ec_level2=["2.7", "3.5"],
                ec_level3=["2.7.7", "3.5.4"],
            ),
            _ec_jsonl_record(
                "ec_full:head:p5",
                "TVWY",
                ["1.1.1.5"],
                ec_level1=["1"],
                ec_level2=["1.1"],
                ec_level3=["1.1.1"],
            ),
            _ec_jsonl_record(
                "ec_full:head:p7",
                "AAAA",
                ["1.1.2.1"],
                ec_level1=["1"],
                ec_level2=["1.1"],
                ec_level3=["1.1.2"],
            ),
            _ec_jsonl_record(
                "ec_full:head:p8",
                "CCCC",
                ["1.2.1.1"],
                ec_level1=["1"],
                ec_level2=["1.2"],
                ec_level3=["1.2.1"],
            ),
        ],
        ("head", "test"): [
            _ec_jsonl_record(
                "ec_full:head:p6",
                "LMNP",
                ["1.1.1.6"],
                ec_level1=["1"],
                ec_level2=["1.1"],
                ec_level3=["1.1.1"],
            )
        ],
        ("main", "val"): [
            _ec_jsonl_record(
                "ec_full:main:p3",
                "KLMN",
                ["3.5.4.10"],
                ec_level1=["3"],
                ec_level2=["3.5"],
                ec_level3=["3.5.4"],
            )
        ],
        ("main", "test"): [
            _ec_jsonl_record(
                "ec_full:main:p4",
                "PQRS",
                ["1.1.1.1"],
                ec_level1=["1"],
                ec_level2=["1.1"],
                ec_level3=["1.1.1"],
            )
        ],
        ("full", "train"): [
            _ec_jsonl_record(
                "ec_full:full:p9",
                "MMMM",
                ["9.9.9.9"],
                ec_level1=["9"],
                ec_level2=["9.9"],
                ec_level3=["9.9.9"],
            )
        ],
    }
    for (track, split), split_records in records.items():
        path = ec_root / "tasks" / "ec_full" / track / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in split_records),
            encoding="utf-8",
        )


def _write_ec_holdout_budget_layout(root: Path) -> None:
    split_dir = root / "ec_main_head" / "tasks" / "ec_full" / "head"
    split_dir.mkdir(parents=True)
    for split in ("val", "test"):
        (split_dir / f"{split}.jsonl").write_text("", encoding="utf-8")

    records: list[dict[str, object]] = []
    child_counts = {
        "1.1.1.1": 1,
        "1.1.1.2": 2,
        "1.1.1.3": 3,
        "1.1.1.4": 20,
    }
    index = 0
    for exact_label, count in child_counts.items():
        for _ in range(count):
            index += 1
            records.append(
                _ec_jsonl_record(
                    f"ec_full:head:budget{index}",
                    "ACDE",
                    [exact_label],
                    ec_level1=["1"],
                    ec_level2=["1.1"],
                    ec_level3=["1.1.1"],
                )
            )
    (split_dir / "train.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _ec_jsonl_record(
    record_id: str,
    sequence: str,
    labels_exact: list[str],
    *,
    ec_level1: list[str],
    ec_level2: list[str],
    ec_level3: list[str],
) -> dict[str, object]:
    return {
        "cluster_id": record_id.rpartition(":")[2],
        "id": record_id,
        "labels_exact": labels_exact,
        "metadata": {
            "ec_level1": ec_level1,
            "ec_level2": ec_level2,
            "ec_level3": ec_level3,
            "track": record_id.split(":")[1],
        },
        "rep_accession": record_id.rpartition(":")[2],
        "sequence": sequence,
    }


def _write_clean_dataset_layout(root: Path) -> None:
    clean_root = root / "CLEAN" / "CLEAN_all_train_valid_splits"
    split_dir = clean_root / "split30"
    split_dir.mkdir(parents=True)
    (split_dir / "split30_train_split_0.csv").write_text(
        "\n".join(
            [
                "Entry\tEC number\tSequence",
                "clean_train_1\t1.1.1.1\tACDE",
                "clean_train_2\t1.1.1.2;1.1.2.1\tFGHI",
                "clean_train_partial\t3.4.24.-\tKLMN",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (split_dir / "split30_test_split_0_curate.csv").write_text(
        "\n".join(
            [
                "ID\tEC\tSequences",
                "clean_test_1\t2.7.7.4\tPQRS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (clean_root / "halogenase.csv").write_text(
        "\n".join(
            [
                "Entry\tEC number\tSequence",
                "clean_halogenase_1\t4.2.3.158\tTVWY",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (clean_root / "price.csv").write_text(
        "\n".join(
            [
                "Entry\tEC number\tSequence",
                "clean_price_1\t5.4.2.1\tAAAA",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (clean_root / "new.csv").write_text(
        "\n".join(
            [
                "Entry\tEC number\tSequence",
                "clean_new_1\t6.1.1.10\tCCCC",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_ecbench_dataset_layout(root: Path) -> None:
    ecbench_root = root / "ec-benchmark"
    ecbench_root.mkdir()
    (ecbench_root / "train_30.csv").write_text(
        "\n".join(
            [
                "id,seq,ec_number",
                "ecbench_train_1,ACDE,1.1.1.1",
                "ecbench_train_2,FGHI,1.1.1.2;1.1.2.1",
                "ecbench_train_partial,KLMN,3.4.24.-",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ecbench_root / "train_100.csv").write_text(
        "\n".join(
            [
                "id,seq,ec_number",
                "ecbench_train100_1,ACDE,6.1.1.10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ecbench_root / "test_ec.csv").write_text(
        "\n".join(
            [
                "id,seq,ec_number",
                "ecbench_test_1,PQRS,2.7.7.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for test_set, label in {
        "halogenase": "4.2.3.158",
        "price": "5.4.2.1",
        "new": "6.1.1.10",
    }.items():
        (ecbench_root / f"{test_set}.csv").write_text(
            "\n".join(
                [
                    "id,seq,ec_number",
                    f"ecbench_{test_set}_1,TVWY,{label}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def _write_go_dataset_layout(root: Path) -> None:
    go_root = root / "go_main_head"
    manifest_dir = go_root / "manifests"
    manifest_dir.mkdir(parents=True)
    ontology_dir = go_root / "go_ontology"
    ontology_dir.mkdir()
    (ontology_dir / "go.obo").write_text(
        """format-version: 1.2

[Term]
id: GO:0008150
name: biological process
namespace: biological_process

[Term]
id: GO:0009987
name: cellular process
namespace: biological_process
is_a: GO:0008150 ! biological process

[Term]
id: GO:0008151
name: unrelated process
namespace: biological_process
is_a: GO:0008150 ! biological process

[Term]
id: GO:0003674
name: molecular function
namespace: molecular_function
""",
        encoding="utf-8",
    )
    for aspect in ("bp", "cc", "mf"):
        for track in ("head", "main", "full"):
            (manifest_dir / f"go_{aspect}_{track}.json").write_text(
                json.dumps({"task": f"go_{aspect}", "track": track}) + "\n",
                encoding="utf-8",
            )
            split_dir = go_root / "tasks" / f"go_{aspect}" / track
            split_dir.mkdir(parents=True)
            for split in ("train", "val", "test"):
                (split_dir / f"{split}.jsonl").write_text("", encoding="utf-8")
    (go_root / "tasks" / "go_bp" / "head" / "train.jsonl").write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                _go_jsonl_record("go_bp:head:p1", "ACDE", ["GO:0008150", "GO:0009987"]),
                _go_jsonl_record("go_bp:head:p2", "FGHI", ["GO:0008150", "GO:0008151"]),
            ]
        ),
        encoding="utf-8",
    )
    (go_root / "tasks" / "go_mf" / "main" / "test.jsonl").write_text(
        json.dumps(_go_jsonl_record("go_mf:main:p3", "KLMN", ["GO:0003674"])) + "\n",
        encoding="utf-8",
    )
    (go_root / "tasks" / "go_bp" / "head" / "test.jsonl").write_text(
        json.dumps(_go_jsonl_record("go_bp:head:p3", "KLMN", ["GO:0008150", "GO:0009987"]))
        + "\n",
        encoding="utf-8",
    )
    (go_root / "tasks" / "go_bp" / "full" / "train.jsonl").write_text(
        json.dumps(_go_jsonl_record("go_bp:full:p4", "MMMM", ["GO:0008150", "GO:0008151"]))
        + "\n",
        encoding="utf-8",
    )


def _go_jsonl_record(record_id: str, sequence: str, labels_propagated: list[str]) -> dict[str, object]:
    cluster_id = record_id.rpartition(":")[2]
    return {
        "cluster_id": cluster_id,
        "id": record_id,
        "labels_asserted": labels_propagated[-1:],
        "labels_propagated": labels_propagated,
        "metadata": {"aspect": "biological_process", "track": record_id.split(":")[1]},
        "rep_accession": cluster_id,
        "sequence": sequence,
    }


def _write_cafa6_dataset_layout(root: Path) -> None:
    train_dir = root / "Train"
    train_dir.mkdir(parents=True)
    (train_dir / "train_sequences.fasta").write_text(
        "\n".join(
            [
                ">sp|P1|ONE protein one",
                "ACDE",
                ">P2 protein two",
                "FGHI",
                ">P3 protein three",
                "KLMN",
                ">P4 protein four",
                "PQRS",
                ">P5 protein five",
                "TVWY",
                ">P6 protein six",
                "AAAA",
                ">P7 protein seven",
                "CCCC",
                ">P8 protein eight",
                "DDDD",
                ">P9 protein nine",
                "EEEE",
                ">P10 protein ten",
                "FFFF",
                ">P11 protein eleven",
                "GGGG",
                ">P12 protein twelve",
                "HHHH",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (train_dir / "train_terms.tsv").write_text(
        "\n".join(
            [
                "EntryID\tterm\taspect",
                "P1\tGO:0009987\tP",
                "P1\tGO:0008151\tP",
                "P2\tGO:0009987\tP",
                "P3\tGO:0005575\tC",
                "P4\tGO:0003674\tF",
                "P5\tGO:0009987\tP",
                "P6\tGO:0009987\tP",
                "P7\tGO:0009987\tP",
                "P8\tGO:0009987\tP",
                "P9\tGO:0009987\tP",
                "P10\tGO:0009987\tP",
                "P11\tGO:0009987\tP",
                "P12\tGO:0009987\tP",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (train_dir / "go-basic.obo").write_text(
        """format-version: 1.2

[Term]
id: GO:0008150
name: biological process
namespace: biological_process

[Term]
id: GO:0009987
name: cellular process
namespace: biological_process
is_a: GO:0008150 ! biological process

[Term]
id: GO:0008151
name: unrelated process
namespace: biological_process
is_a: GO:0008150 ! biological process

[Term]
id: GO:0005575
name: cellular component
namespace: cellular_component

[Term]
id: GO:0003674
name: molecular function
namespace: molecular_function
""",
        encoding="utf-8",
    )
    (root / "IA.tsv").write_text(
        "\n".join(
            [
                "GO:0008150\t0.0",
                "GO:0009987\t2.0",
                "GO:0008151\t1.0",
                "GO:0005575\t1.5",
                "GO:0003674\t1.25",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_cafa5_target_subset_layout(root: Path) -> None:
    _write_cafa6_dataset_layout(root)
    target_dir = root / "Test (Targets)"
    target_dir.mkdir(parents=True)
    (target_dir / "eval_terms_partial_2025_03.tsv").write_text(
        "\n".join(
            [
                "EntryID\tterm\taspect",
                "T1\tGO:0009987\tBPO",
                "T2\tGO:0008151\tBPO",
                "T3\tGO:0005575\tCCO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (target_dir / "known_t0.tsv").write_text(
        "\n".join(
            [
                "EntryID\tterm\taspect",
                "T1\tGO:0008151\tBPO",
                "T2\tGO:0009987\tBPO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (target_dir / "toi_2025_03.tsv").write_text("GO:0009987\nGO:0008151\n", encoding="utf-8")
    (target_dir / "testsuperset.fasta").write_text(
        "\n".join(
            [
                ">T1 target one",
                "MMMM",
                ">T2 target two",
                "NNNN",
                ">T3 target three",
                "PPPP",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_cafa_lowercase_aspect_layout(root: Path) -> None:
    train_dir = root / "Train"
    train_dir.mkdir(parents=True)
    (train_dir / "train_sequences.fasta").write_text(
        "\n".join(
            [
                ">P1 protein one",
                "ACDE",
                ">P2 protein two",
                "FGHI",
                ">P3 protein three",
                "KLMN",
                ">P4 protein four",
                "PQRS",
                ">P5 protein five",
                "TVWY",
                ">P6 protein six",
                "AAAA",
                ">P7 protein seven",
                "CCCC",
                ">P8 protein eight",
                "DDDD",
                ">P9 protein nine",
                "EEEE",
                ">P10 protein ten",
                "FFFF",
                ">P11 protein eleven",
                "GGGG",
                ">P12 protein twelve",
                "HHHH",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (train_dir / "train_terms.tsv").write_text(
        "\n".join(
            [
                "Entry_ID\tTERM\tAspect",
                "P1\tGO:0009987\tp",
                "P2\tGO:0009987\tp",
                "P3\tGO:0005575\tc",
                "P4\tGO:0003674\tf",
                "P5\tGO:0009987\tp",
                "P6\tGO:0009987\tp",
                "P7\tGO:0009987\tp",
                "P8\tGO:0009987\tp",
                "P9\tGO:0009987\tp",
                "P10\tGO:0009987\tp",
                "P11\tGO:0009987\tp",
                "P12\tGO:0009987\tp",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_cafa6_dataset_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "Train/train_sequences.fasta",
            "\n".join(
                [
                    ">P1 protein one",
                    "ACDE",
                    ">P2 protein two",
                    "FGHI",
                    ">P3 protein three",
                    "KLMN",
                    ">P4 protein four",
                    "PQRS",
                    ">P5 protein five",
                    "TVWY",
                    ">P6 protein six",
                    "AAAA",
                    ">P7 protein seven",
                    "CCCC",
                    ">P8 protein eight",
                    "DDDD",
                    ">P9 protein nine",
                    "EEEE",
                    ">P10 protein ten",
                    "FFFF",
                    ">P11 protein eleven",
                    "GGGG",
                    ">P12 protein twelve",
                    "HHHH",
                ]
            )
            + "\n",
        )
        archive.writestr(
            "Train/train_terms.tsv",
            "\n".join(
                [
                    "EntryID\tterm\taspect",
                    "P1\tGO:0009987\tP",
                    "P1\tGO:0008151\tP",
                    "P2\tGO:0009987\tP",
                    "P3\tGO:0005575\tC",
                    "P4\tGO:0003674\tF",
                    "P5\tGO:0009987\tP",
                    "P6\tGO:0009987\tP",
                    "P7\tGO:0009987\tP",
                    "P8\tGO:0009987\tP",
                    "P9\tGO:0009987\tP",
                    "P10\tGO:0009987\tP",
                    "P11\tGO:0009987\tP",
                    "P12\tGO:0009987\tP",
                ]
            )
            + "\n",
        )
        archive.writestr(
        "Train/go-basic.obo",
            """format-version: 1.2

[Term]
id: GO:0008150
name: biological process
namespace: biological_process

[Term]
id: GO:0009987
name: cellular process
namespace: biological_process
is_a: GO:0008150 ! biological process
""",
        )
        archive.writestr("IA.tsv", "GO:0008150\t0.0\nGO:0009987\t2.0\n")


def test_dtu_catalog_lists_relevant_residue_services() -> None:
    residue_names = {service.name for service in list_dtu_services(level="residue")}
    ptm_names = {service.name for service in list_dtu_services(category="ptm")}
    dataset_names = {service.name for service in list_dtu_services(has_dataset=True)}

    assert {"netphos", "netoglyc", "signalp", "netsurfp", "deeptmhmm"}.issubset(residue_names)
    assert {"dictyoglyc", "netcglyc", "netnglyc", "netphos"}.issubset(ptm_names)
    assert {"signalp", "signalp_datasets", "dna2protss", "netnes"}.issubset(dataset_names)
    assert get_dtu_service("SignalP").target == "signal_peptide_cleavage_site"


def test_unified_dataset_catalog_is_filterable_and_searchable() -> None:
    ready_residue_ptm = list_dataset_catalog(level="residue", category="ptm", status="ready")
    structure_tasks = list_dataset_catalog(task_class="structure")
    f1_tasks = list_dataset_catalog(preferred_metric="f1")
    search_results = search_dataset_catalog("phosphorylation cdk residue", status="ready")

    assert get_dataset_catalog_entry("dbptm:phosphorylation_by_cdk").loader == "load_dbptm_benchmark_dataset"
    assert get_dataset_catalog_entry("disprot:all").status == "ready"
    assert get_dataset_catalog_entry("disprot:all").loader == "load_residue_source_dataset"
    assert get_dataset_catalog_entry("biolip:dna").status == "ready"
    assert get_dataset_catalog_entry("biolip:dna").target == "dna_binding_site"
    assert get_dataset_catalog_entry("phosphoelm:all").status == "ready"
    assert get_dataset_catalog_entry("phosphoelm:ltp").loader == "load_residue_source_dataset"
    assert get_dataset_catalog_entry("phosphoelm:htp").target == "phosphorylation_site"
    assert get_dataset_catalog_entry("phosphoelm:ltp").task_class == "ptms"
    assert get_dataset_catalog_entry("phosphoelm:ltp").preferred_metric == "f1"
    assert get_dataset_catalog_entry("peer:fold").task_class == "structure"
    assert get_dataset_catalog_entry("peer:fold").preferred_metric == "accuracy"
    assert get_dataset_catalog_entry("ec:ec_1_head").split_counts == (28147, 3519, 3519)
    assert get_dataset_catalog_entry("ec:ec_4_main").sample_count == 45405
    assert get_dataset_catalog_entry("ec:ec_4_main").loader == "load_ec_dataset"
    assert get_dataset_catalog_entry("ec:ec_4_main").preferred_metric == "fmax"
    assert "fmax" in get_dataset_catalog_entry("ec:ec_4_main").metrics
    assert "weighted_f1" in get_dataset_catalog_entry("ec:ec_4_main").metrics
    assert get_dataset_catalog_entry("ec:ec_4_full").sample_count == 60620
    assert get_dataset_catalog_entry("go:go_bp_head").split_counts == (20948, 2620, 2619)
    assert get_dataset_catalog_entry("go_mf_main").sample_count == 41081
    assert get_dataset_catalog_entry("go:go_mf_full").split_counts == (40395, 4994, 5057)
    assert get_dataset_catalog_entry("go:go_cc_main").loader == "load_go_dataset"
    assert get_dataset_catalog_entry("go:go_bp_head").preferred_metric == "go_fmax"
    assert get_dataset_catalog_entry("cafa:cafa5_bp").status == "adapter"
    assert get_dataset_catalog_entry("cafa:cafa5_bp").preferred_metric == "go_weighted_fmax"
    assert get_dataset_catalog_entry("cafa5:cafa5_cc").id == "cafa:cafa5_cc"
    assert get_dataset_catalog_entry("cafa:cafa5_partial_bp").target == "go_bp"
    assert get_dataset_catalog_entry("cafa:cafa5_partial_bp").loader == "load_cafa5_dataset"
    assert get_dataset_catalog_entry("cafa5_pk_bp").id == "cafa:cafa5_partial_bp"
    with pytest.raises(EmbeddingInputError, match="Unknown dataset"):
        get_dataset_catalog_entry("cafa:cafa6_bp")
    assert get_dataset_catalog_entry("ec_2_head").target == "ec_2"
    assert get_dataset_catalog_entry("ec:single_ec_4_main").objective == "multiclass"
    assert "weighted_f1" in get_dataset_catalog_entry("ec:single_ec_4_main").metrics
    assert get_dataset_catalog_entry("single_ec_4_main").split_counts == (34213, 4368, 4376)
    assert get_dataset_catalog_entry("single_ec_4_full").sample_count == 56165
    assert get_dataset_catalog_entry("ec:single_ec_1_head").target == "single_ec_1"
    assert get_dataset_catalog_entry("ec:ec_3_subclass_holdout_main").objective == "multiclass"
    assert get_dataset_catalog_entry("ec:ec_3_subclass_holdout_main").split_counts == (32143, 4040, 4021)
    assert get_dataset_catalog_entry("ec_3_subclass_holdout_head").target == "ec_3"
    assert get_dataset_catalog_entry("ec:ec_2_subclass_holdout_main").split_counts == (29773, 3771, 3738)
    assert get_dataset_catalog_entry("ec_2_subclass_holdout_head").target == "ec_2"
    assert get_dataset_catalog_entry("ec:ec_1_subclass_holdout_main").split_counts == (34592, 5057, 4485)
    assert get_dataset_catalog_entry("ec:ec_3_subclass_holdout_full").split_counts == (44304, 5542, 5543)
    assert get_dataset_catalog_entry("ec_1_subclass_holdout_head").target == "ec_1"
    assert get_dataset_catalog_entry("clean:ec_4_split30_fold0").loader == "load_clean_dataset"
    assert get_dataset_catalog_entry("clean:ec_4_split30_fold0").preferred_metric == "fmax"
    assert get_dataset_catalog_entry("clean:ec_4_split30_fold0_price").target == "ec_4"
    assert get_dataset_catalog_entry("ecbench:ec_4_train_100").loader == "load_ecbench_dataset"
    assert get_dataset_catalog_entry("ecbench:ec_4_train_100").preferred_metric == "fmax"
    assert "weighted_f1" in get_dataset_catalog_entry("ecbench:ec_4_train_100").metrics
    assert get_dataset_catalog_entry("ec-bench:ec_1_train_30_new").id == "ecbench:ec_1_train_30_new"
    assert get_dataset_catalog_entry("source:phosphoelm_ltp").id == "phosphoelm:ltp"
    assert get_dataset_catalog_entry("secondary_structure").id == "peer:secondary_structure"
    for entry in list_dataset_catalog():
        description = entry.description.lower()
        assert entry.task_class
        assert entry.preferred_metric
        assert "source:" in description
        assert "class:" in description
        assert "split system:" in description
    assert "dbptm:phosphorylation_by_cdk" in {entry.id for entry in ready_residue_ptm}
    assert "phosphoelm:all" in {entry.id for entry in ready_residue_ptm}
    assert "peer:fold" in {entry.id for entry in structure_tasks}
    assert "phosphoelm:ltp" in {entry.id for entry in f1_tasks}
    assert "ec:ec_4_main" in {entry.id for entry in list_dataset_catalog(collection="ec")}
    assert "ec:ec_4_full" in {entry.id for entry in list_dataset_catalog(collection="ec")}
    assert "go:go_mf_main" in {entry.id for entry in list_dataset_catalog(collection="go")}
    assert "go:go_mf_full" in {entry.id for entry in list_dataset_catalog(collection="go")}
    assert "cafa:cafa5_bp" in {entry.id for entry in list_dataset_catalog(collection="cafa")}
    assert "cafa:cafa5_partial_bp" in {entry.id for entry in list_dataset_catalog(collection="cafa")}
    assert "cafa:cafa6_bp" not in {entry.id for entry in list_dataset_catalog(collection="cafa")}
    assert "clean:ec_4_split30_fold0" in {entry.id for entry in list_dataset_catalog(collection="clean")}
    assert "ecbench:ec_4_train_100" in {entry.id for entry in list_dataset_catalog(collection="ecbench")}
    assert "ec:single_ec_4_main" in {entry.id for entry in list_dataset_catalog(collection="ec", objective="multiclass")}
    assert "ec:ec_3_subclass_holdout_main" in {
        entry.id for entry in list_dataset_catalog(collection="ec", objective="multiclass")
    }
    assert "ec:ec_2_subclass_holdout_main" in {
        entry.id for entry in list_dataset_catalog(collection="ec", objective="multiclass")
    }
    assert "ec:ec_1_subclass_holdout_main" in {
        entry.id for entry in list_dataset_catalog(collection="ec", objective="multiclass")
    }
    assert search_results[0].id == "dbptm:phosphorylation_by_cdk"
    assert "peer:secondary_structure" in {entry.id for entry in search_dataset_catalog("secondary structure")}
    assert "phosphoelm:ltp" in {entry.id for entry in search_dataset_catalog("ptms f1 ltp")}
    assert "ec:ec_3_head" in {entry.id for entry in search_dataset_catalog("enzyme commission ec 3 head")}
    assert "ec:single_ec_3_head" in {entry.id for entry in search_dataset_catalog("single enzyme commission ec 3 head")}
    assert "ec:ec_3_subclass_holdout_head" in {
        entry.id for entry in search_dataset_catalog("ec 3 subclass holdout head")
    }
    assert "ec:ec_2_subclass_holdout_head" in {
        entry.id for entry in search_dataset_catalog("ec 2 subclass holdout head")
    }
    assert "ec:ec_1_subclass_holdout_head" in {
        entry.id for entry in search_dataset_catalog("ec 1 subclass holdout head")
    }
    assert "go:go_bp_head" in {entry.id for entry in search_dataset_catalog("gene ontology bp head")}
    assert "go:go_bp_full" in {entry.id for entry in search_dataset_catalog("gene ontology bp full")}
    assert "cafa:cafa5_bp" in {entry.id for entry in search_dataset_catalog("cafa5 kaggle bp")}
    assert "cafa:cafa5_partial_bp" in {entry.id for entry in search_dataset_catalog("cafa5 partial bp")}
    assert "clean:ec_4_split30_fold0" in {entry.id for entry in search_dataset_catalog("clean ec 4 split30")}
    assert "ecbench:ec_4_train_100" in {entry.id for entry in search_dataset_catalog("ecbench ec 4 train 100")}


def test_dataset_catalog_is_flattened_from_registered_collections() -> None:
    collections = list_dataset_collections()
    published_ids = {
        dataset.id
        for collection in collections
        for dataset in collection.list_datasets()
    }

    assert published_ids == {dataset.id for dataset in list_dataset_catalog()}
    assert {collection.id for collection in collections} == {
        "peer",
        "dbptm",
        "ec",
        "clean",
        "ecbench",
        "go",
        "cafa",
        "musitedeep",
        "disprot",
        "biolip",
        "metalpdb",
        "scannet",
        "netsurfp",
        "phosphoelm",
        "dtu",
    }


def test_dataset_collection_groups_related_provider_datasets() -> None:
    collection = get_dataset_collection("biolip")

    assert {dataset.id for dataset in collection.list_datasets()} == {
        "biolip:all",
        "biolip:dna",
        "biolip:rna",
        "biolip:pep",
        "biolip:other",
    }


def test_load_dataset_routes_through_owning_collection(tmp_path: Path) -> None:
    data_dir = tmp_path / "disprot"
    data_dir.mkdir()
    (data_dir / "disprot_current.tsv").write_text(
        "\n".join(
            [
                "UniProt ACC\tDisProt ID\tStart\tEnd\tRegion sequence\tTerm ID\tTerm name",
                "P1\tDP1\t2\t4\tCDE\tIDPO:0000002\tdisorder",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_dataset("disprot:all", tmp_path)

    assert dataset.target_values("disorder") == {"DP1": [1, 1, 1]}


def test_load_ec_dataset_imports_generated_level_targets(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    level1 = load_dataset("ec:ec_1_head", tmp_path, split="train", download=True)
    level4 = load_ec_dataset(tmp_path / "ec_main_head", name="ec_4_main", split=("val", "test"))
    full = load_dataset("ec:ec_4_full", tmp_path, split="train", download=True)

    assert level1.split_counts() == {"train": 5, "val": 0, "test": 0}
    assert level1.target_values("ec_1") == {
        "ec_full:head:p1": ["1"],
        "ec_full:head:p2": ["2", "3"],
        "ec_full:head:p5": ["1"],
        "ec_full:head:p7": ["1"],
        "ec_full:head:p8": ["1"],
    }
    assert level4.split_counts() == {"train": 0, "val": 1, "test": 1}
    assert level4.target_values("ec_4") == {
        "ec_full:main:p3": ["3.5.4.10"],
        "ec_full:main:p4": ["1.1.1.1"],
    }
    assert full.target_values("ec_4") == {"ec_full:full:p9": ["9.9.9.9"]}


def test_load_clean_dataset_imports_level_targets_and_named_test_sets(tmp_path: Path) -> None:
    _write_clean_dataset_layout(tmp_path)

    native = load_dataset("clean:ec_3_split30_fold0", tmp_path)
    halogenase = load_clean_dataset(tmp_path, name="ec_4_split30_fold0_halogenase")

    assert native.split_counts() == {"train": 3, "val": 0, "test": 1}
    assert native.target_values("ec_3") == {
        "clean_train_1": ["1.1.1"],
        "clean_train_2": ["1.1.1", "1.1.2"],
        "clean_train_partial": ["3.4.24"],
        "clean_test_1": ["2.7.7"],
    }
    assert halogenase.split_counts() == {"train": 2, "val": 0, "test": 1}
    assert halogenase.target_values("ec_4") == {
        "clean_train_1": ["1.1.1.1"],
        "clean_train_2": ["1.1.1.2", "1.1.2.1"],
        "clean_halogenase_1": ["4.2.3.158"],
    }


def test_load_ecbench_dataset_imports_level_targets_and_named_test_sets(tmp_path: Path) -> None:
    _write_ecbench_dataset_layout(tmp_path)

    native = load_dataset("ec-bench:ec_3_train_30", tmp_path)
    price = load_ecbench_dataset(tmp_path, name="ecbench:ec4_train100_price")

    assert native.split_counts() == {"train": 3, "val": 0, "test": 1}
    assert native.target_values("ec_3") == {
        "ecbench_train_1": ["1.1.1"],
        "ecbench_train_2": ["1.1.1", "1.1.2"],
        "ecbench_train_partial": ["3.4.24"],
        "ecbench_test_1": ["2.7.7"],
    }
    assert price.split_counts() == {"train": 1, "val": 0, "test": 1}
    assert price.target_values("ec_4") == {
        "ecbench_train100_1": ["6.1.1.10"],
        "ecbench_price_1": ["5.4.2.1"],
    }


def test_load_go_dataset_imports_generated_asserted_targets(tmp_path: Path) -> None:
    _write_go_dataset_layout(tmp_path)

    bp_dataset = load_dataset("go:go_bp_head", tmp_path, split="train", download=True)
    mf_dataset = load_go_dataset(tmp_path / "go_main_head", name="go_mf_main", split="test")
    full_dataset = load_dataset("go:go_bp_full", tmp_path, split="train", download=True)

    assert bp_dataset.split_counts() == {"train": 2, "val": 0, "test": 0}
    assert bp_dataset.target_values("go_bp") == {
        "go_bp:head:p1": ["GO:0009987"],
        "go_bp:head:p2": ["GO:0008151"],
    }
    assert mf_dataset.target_values("go_mf") == {"go_mf:main:p3": ["GO:0003674"]}
    assert full_dataset.target_values("go_bp") == {"go_bp:full:p4": ["GO:0008151"]}


def test_run_task_on_layer_reports_propagated_go_fmax_for_custom_backend(tmp_path: Path) -> None:
    _write_go_dataset_layout(tmp_path)
    dataset = load_dataset("go:go_bp_head", tmp_path)
    task = Task(
        name="go_bp",
        dataset=dataset,
        prediction=PredictionSpec(target="go_bp", objective="multilabel"),
        probe=_FixedProbeBackend(
            ProbeBackendOutput(
                predictions={"go_bp:head:p3": ["GO:0009987"]},
                scores={"go_bp:head:p3": [0.2, 0.9]},
            )
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "go_bp:head:p1": [0.0],
            "go_bp:head:p2": [0.0],
            "go_bp:head:p3": [0.0],
        },
    )

    assert result.metrics["go_fmax"] == pytest.approx(1.0)
    assert result.metrics["go_fmax_threshold"] == pytest.approx(0.21)
    assert result.metrics["go_precision_at_fmax"] == pytest.approx(1.0)
    assert result.metrics["go_recall_at_fmax"] == pytest.approx(1.0)
    assert result.metrics["go_evaluated_protein_count"] == pytest.approx(1.0)
    assert "go_propagated_f1" not in result.metrics
    assert "go_propagated_threshold" not in result.metrics


def test_run_task_on_layer_reports_fixed_threshold_propagated_go_metrics_for_transfer_probe(tmp_path: Path) -> None:
    _write_go_dataset_layout(tmp_path)
    dataset = load_dataset("go:go_bp_head", tmp_path)
    task = Task(
        name="go_bp",
        dataset=dataset,
        prediction=PredictionSpec(target="go_bp", objective="multilabel"),
        probe=TransferProbe(
            neighbor_selection="knn",
            k=1,
            threshold=0.25,
            scoring="voting",
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={
            "go_bp:head:p1": [1.0, 0.0],
            "go_bp:head:p2": [0.0, 1.0],
            "go_bp:head:p3": [0.9, 0.1],
        },
    )

    assert result.metrics["go_propagated_f1"] == pytest.approx(1.0)
    assert result.metrics["go_propagated_precision"] == pytest.approx(1.0)
    assert result.metrics["go_propagated_recall"] == pytest.approx(1.0)
    assert result.metrics["go_propagated_threshold"] == pytest.approx(0.25)
    assert result.metrics["go_propagated_evaluated_protein_count"] == pytest.approx(1.0)


def test_run_task_on_layer_ignores_validation_only_go_classes_for_fmax(tmp_path: Path) -> None:
    _write_go_dataset_layout(tmp_path)
    metadata = {
        "source": "go",
        "labels_propagated": ["GO:0009987"],
        "go_obo_path": str(tmp_path / "go_main_head" / "go_ontology" / "go.obo"),
    }
    dataset = ProteinDataset(
        [
            ProteinExample("train", "AAAA", {"go_bp": ["GO:0009987"]}, "train", metadata=metadata),
            ProteinExample(
                "val",
                "CCCC",
                {"go_bp": ["GO:0008151"]},
                "val",
                metadata={**metadata, "labels_propagated": ["GO:0008151"]},
            ),
            ProteinExample("test", "DDDD", {"go_bp": ["GO:0009987"]}, "test", metadata=metadata),
        ]
    )
    task = Task(
        name="go_bp",
        dataset=dataset,
        prediction=PredictionSpec(target="go_bp", objective="multilabel"),
        probe=_FixedProbeBackend(
            ProbeBackendOutput(
                predictions={"test": ["GO:0009987"]},
                scores={"test": [0.9]},
            )
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={"train": [0.0], "val": [0.5], "test": [1.0]},
    )

    assert result.metrics["go_fmax"] == pytest.approx(1.0)


def test_go_protein_centric_metrics_exclude_root_terms(tmp_path: Path) -> None:
    _write_go_dataset_layout(tmp_path)
    ontology = load_go(str(tmp_path / "go_main_head" / "go_ontology" / "go.obo"))

    with pytest.raises(ValueError, match="non-root"):
        go_protein_centric_metrics(
            {"p1": [0.9]},
            class_names=["GO:0008150"],
            true_labels={"p1": ["GO:0008150"]},
            ontology=ontology,
        )


def test_go_weighted_protein_centric_metrics_use_information_accretion(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)
    ontology = load_go(str(tmp_path / "Train" / "go-basic.obo"))
    weights = read_information_accretion_weights(tmp_path / "IA.tsv")

    metrics = go_weighted_protein_centric_metrics(
        {
            "p1": [0.1, 0.9],
            "p2": [0.7, 0.8],
        },
        class_names=["GO:0008151", "GO:0009987"],
        true_labels={
            "p1": ["GO:0009987"],
            "p2": ["GO:0008151"],
        },
        ontology=ontology,
        term_weights=weights,
    )

    assert metrics["go_weighted_fmax"] == pytest.approx(0.8)
    assert metrics["go_weighted_macro_fmax"] == pytest.approx(0.8)
    assert metrics["go_weighted_micro_fmax"] == pytest.approx(0.8)
    assert metrics["go_weighted_fmax_threshold"] == pytest.approx(0.11)
    assert metrics["go_weighted_precision_at_fmax"] == pytest.approx(2.0 / 3.0)
    assert metrics["go_weighted_recall_at_fmax"] == pytest.approx(1.0)


def test_go_combined_protein_centric_metrics_matches_separate_metrics(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)
    ontology = load_go(str(tmp_path / "Train" / "go-basic.obo"))
    weights = read_information_accretion_weights(tmp_path / "IA.tsv")
    predicted_scores = {
        "p1": [0.1, 0.9],
        "p2": [0.7, 0.8],
    }
    class_names = ["GO:0008151", "GO:0009987"]
    true_labels = {
        "p1": ["GO:0009987"],
        "p2": ["GO:0008151"],
    }

    combined = go_combined_protein_centric_metrics(
        predicted_scores,
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        term_weights=weights,
        threshold_count=101,
    )
    unweighted = go_protein_centric_metrics(
        predicted_scores,
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        threshold_count=101,
    )
    weighted = go_weighted_protein_centric_metrics(
        predicted_scores,
        class_names=class_names,
        true_labels=true_labels,
        ontology=ontology,
        term_weights=weights,
        threshold_count=101,
    )

    for key, value in {**unweighted, **weighted}.items():
        assert combined[key] == pytest.approx(value)
    assert combined["go_macro_fmax"] == pytest.approx(combined["go_fmax"])
    assert combined["go_micro_fmax"] == pytest.approx(0.8)
    assert combined["go_weighted_macro_fmax"] == pytest.approx(combined["go_weighted_fmax"])
    assert combined["go_weighted_micro_fmax"] == pytest.approx(0.8)
    assert combined["go_weighted_smin"] == pytest.approx(0.5)
    assert combined["go_weighted_remaining_uncertainty_at_smin"] == pytest.approx(0.5)
    assert combined["go_weighted_misinformation_at_smin"] == pytest.approx(0.0)


def test_go_combined_protein_centric_metrics_excludes_known_partial_knowledge_terms(
    tmp_path: Path,
) -> None:
    _write_cafa6_dataset_layout(tmp_path)
    ontology = load_go(str(tmp_path / "Train" / "go-basic.obo"))
    weights = read_information_accretion_weights(tmp_path / "IA.tsv")

    metrics = go_combined_protein_centric_metrics(
        {"p1": [0.95, 0.9]},
        class_names=["GO:0008151", "GO:0009987"],
        true_labels={"p1": ["GO:0009987"]},
        ontology=ontology,
        term_weights=weights,
        known_labels={"p1": ["GO:0008151"]},
        terms_of_interest=["GO:0009987"],
        threshold_count=101,
    )

    assert metrics["go_fmax"] == pytest.approx(1.0)
    assert metrics["go_weighted_fmax"] == pytest.approx(1.0)
    assert metrics["go_weighted_smin"] == pytest.approx(0.0)


def test_go_fixed_threshold_protein_centric_metrics_report_propagated_f1(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)
    ontology = load_go(str(tmp_path / "Train" / "go-basic.obo"))

    metrics = go_fixed_threshold_protein_centric_metrics(
        {
            "p1": [0.1, 0.9],
            "p2": [0.7, 0.8],
        },
        class_names=["GO:0008151", "GO:0009987"],
        true_labels={
            "p1": ["GO:0009987"],
            "p2": ["GO:0008151"],
        },
        ontology=ontology,
        threshold=0.65,
    )

    assert metrics["go_propagated_f1"] == pytest.approx(6.0 / 7.0)
    assert metrics["go_propagated_precision"] == pytest.approx(0.75)
    assert metrics["go_propagated_recall"] == pytest.approx(1.0)
    assert metrics["go_propagated_threshold"] == pytest.approx(0.65)
    assert metrics["go_propagated_evaluated_protein_count"] == pytest.approx(2.0)


def test_run_task_on_layer_reports_cafa6_weighted_go_fmax(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)
    metadata = {
        "source": "cafa6",
        "labels_propagated": ["GO:0009987"],
        "go_obo_path": str(tmp_path / "Train" / "go-basic.obo"),
        "ia_path": str(tmp_path / "IA.tsv"),
    }
    dataset = ProteinDataset(
        [
            ProteinExample("train", "AAAA", {"go_bp": ["GO:0009987"]}, "train", metadata=metadata),
            ProteinExample("p1", "ACDE", {"go_bp": ["GO:0009987"]}, "test", metadata=metadata),
            ProteinExample(
                "p2",
                "FGHI",
                {"go_bp": ["GO:0008151"]},
                "test",
                metadata={**metadata, "labels_propagated": ["GO:0008151"]},
            ),
        ]
    )
    task = Task(
        name="cafa6_bp",
        dataset=dataset,
        prediction=PredictionSpec(target="go_bp", objective="multilabel"),
        probe=_FixedProbeBackend(
            ProbeBackendOutput(
                predictions={
                    "p1": ["GO:0009987"],
                    "p2": ["GO:0008151", "GO:0009987"],
                },
                scores={
                    "p1": [0.1, 0.9],
                    "p2": [0.7, 0.8],
                },
            )
        ),
    )

    result = run_task_on_layer(
        task=task,
        embeddings={"train": [0.0], "p1": [1.0], "p2": [2.0]},
    )

    assert result.metrics["go_weighted_fmax"] == pytest.approx(0.8)
    assert result.metrics["go_weighted_fmax_threshold"] == pytest.approx(0.101)
    assert result.metrics["go_weighted_precision_at_fmax"] == pytest.approx(2.0 / 3.0)


def test_cafa6_weighted_fmax_mean_averages_three_aspects() -> None:
    metrics = cafa6_weighted_fmax_mean(
        {
            "mf": {"go_weighted_fmax": 0.6},
            "bp": {"go_weighted_fmax": 0.9},
            "cc": {"go_weighted_fmax": 0.3},
        }
    )

    assert metrics["cafa6_weighted_fmax_mean"] == pytest.approx(0.6)
    assert metrics["cafa6_mf_weighted_fmax"] == pytest.approx(0.6)
    assert metrics["cafa6_bp_weighted_fmax"] == pytest.approx(0.9)
    assert metrics["cafa6_cc_weighted_fmax"] == pytest.approx(0.3)


def test_read_go_manifest_reads_generated_dataset_metadata(tmp_path: Path) -> None:
    _write_go_dataset_layout(tmp_path)

    manifest = read_go_manifest(tmp_path, name="go_bp_head")

    assert manifest == {"task": "go_bp", "track": "head"}


def test_load_go_dataset_raises_on_unsupported_target(tmp_path: Path) -> None:
    _write_go_dataset_layout(tmp_path)

    with pytest.raises(EmbeddingInputError, match="only supports target"):
        load_go_dataset(tmp_path, name="go_bp_head", target="wrong")


def test_load_cafa6_dataset_imports_kaggle_training_terms(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)

    dataset = load_cafa6_dataset(tmp_path, name="cafa6_bp")

    assert dataset.split_counts() == {"train": 8, "val": 1, "test": 1}
    assert dataset.target_values("go_bp")["P1"] == ["GO:0008151", "GO:0009987"]
    assert dataset.target_values("go_bp")["P2"] == ["GO:0009987"]
    assert dataset.examples[0].metadata is not None
    assert dataset.examples[0].metadata["source"] == "cafa6"
    assert dataset.examples[0].metadata["go_obo_path"].endswith("go-basic.obo")
    assert dataset.examples[0].metadata["ia_path"].endswith("IA.tsv")


def test_load_cafa5_dataset_imports_kaggle_training_terms(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)

    dataset = load_cafa5_dataset(tmp_path, name="cafa5_bp")

    assert dataset.split_counts() == {"train": 8, "val": 1, "test": 1}
    assert dataset.target_values("go_bp")["P1"] == ["GO:0008151", "GO:0009987"]
    assert dataset.examples[0].metadata is not None
    assert dataset.examples[0].metadata["source"] == "cafa5"


def test_load_cafa5_partial_subset_uses_released_targets_as_test_split(tmp_path: Path) -> None:
    _write_cafa5_target_subset_layout(tmp_path)

    dataset = load_cafa5_dataset(tmp_path, name="cafa5_partial_bp")

    assert dataset.split_counts() == {"train": 10, "val": 0, "test": 2}
    assert dataset.target_values("go_bp")["T1"] == ["GO:0009987"]
    assert dataset.target_values("go_bp")["T2"] == ["GO:0008151"]
    test_metadata = next(example.metadata for example in dataset.examples if example.id == "T1")
    assert test_metadata is not None
    assert test_metadata["evaluation_setting"] == "pk"
    assert test_metadata["subset"] == "partial"
    assert str(test_metadata["known_terms_path"]).endswith("known_t0.tsv")
    assert str(test_metadata["toi_path"]).endswith("toi_2025_03.tsv")


def test_load_cafa5_partial_subset_requires_known_terms_file(tmp_path: Path) -> None:
    _write_cafa5_target_subset_layout(tmp_path)
    (tmp_path / "Test (Targets)" / "known_t0.tsv").unlink()

    with pytest.raises(EmbeddingInputError, match="known terms"):
        load_cafa5_dataset(tmp_path, name="cafa5_partial_bp")


def test_load_cafa6_dataset_accepts_lowercase_aspects_and_header_variants(tmp_path: Path) -> None:
    _write_cafa_lowercase_aspect_layout(tmp_path)

    dataset = load_cafa6_dataset(tmp_path, name="cafa6_bp")

    assert dataset.split_counts() == {"train": 8, "val": 1, "test": 1}
    assert dataset.target_values("go_bp")["P1"] == ["GO:0009987"]


def test_load_cafa5_dataset_accepts_lowercase_aspects_and_header_variants(tmp_path: Path) -> None:
    _write_cafa_lowercase_aspect_layout(tmp_path)

    dataset = load_cafa5_dataset(tmp_path, name="cafa5_cc")

    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 0}
    assert dataset.target_values("go_cc") == {"P3": ["GO:0005575"]}


def test_load_dataset_routes_cafa5_through_cafa_collection(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)

    dataset = load_dataset("cafa:cafa5_cc", tmp_path, split="train")

    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 0}
    assert dataset.target_values("go_cc") == {"P3": ["GO:0005575"]}


def test_load_dataset_routes_cafa_collection_with_split_filter(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)

    dataset = load_dataset("cafa:cafa5_cc", tmp_path, split="train")

    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 0}
    assert dataset.target_values("go_cc") == {"P3": ["GO:0005575"]}


def test_load_dataset_does_not_route_cafa6_through_public_catalog(tmp_path: Path) -> None:
    _write_cafa6_dataset_layout(tmp_path)

    with pytest.raises(EmbeddingInputError, match="Unknown dataset catalog id"):
        load_dataset("cafa:cafa6_cc", tmp_path, split="train")


def test_download_cafa6_dataset_invokes_kaggle_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        env = kwargs["env"]
        assert isinstance(env, dict)
        environments.append(env)
        output_dir = Path(command[command.index("-p") + 1])
        _write_cafa6_dataset_zip(output_dir / "cafa-6-protein-function-prediction.zip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setenv("KAGGLE_API_TOKEN", "env-token")
    monkeypatch.setattr(cafa_download_module.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(cafa_download_module.subprocess, "run", fake_run)

    paths = download_cafa6_dataset(tmp_path)

    assert paths == [tmp_path / "cafa-6-protein-function-prediction"]
    assert commands == [
        [
            "/usr/bin/kaggle",
            "competitions",
            "download",
            "-c",
            "cafa-6-protein-function-prediction",
            "-p",
            str(tmp_path / "cafa-6-protein-function-prediction"),
        ]
    ]
    assert environments[0]["KAGGLE_API_TOKEN"] == "env-token"


def test_download_cafa6_dataset_reads_access_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environments: list[dict[str, str]] = []
    home = tmp_path / "home"
    token_dir = home / ".kaggle"
    token_dir.mkdir(parents=True)
    (token_dir / "access_token").write_text("file-token\n", encoding="utf-8")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        environments.append(env)
        output_dir = Path(command[command.index("-p") + 1])
        _write_cafa6_dataset_zip(output_dir / "cafa-6-protein-function-prediction.zip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.setattr(cafa_download_module.Path, "home", lambda: home)
    monkeypatch.setattr(cafa_download_module.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(cafa_download_module.subprocess, "run", fake_run)

    download_cafa6_dataset(tmp_path)

    assert environments[0]["KAGGLE_API_TOKEN"] == "file-token"


def test_load_dataset_downloads_cafa5_when_download_is_true(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        output_dir = Path(command[command.index("-p") + 1])
        _write_cafa6_dataset_zip(output_dir / "cafa-5-protein-function-prediction.zip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(cafa_download_module.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(cafa_download_module.subprocess, "run", fake_run)

    dataset = load_dataset("cafa:cafa5_bp", tmp_path, split="test", download=True)

    assert dataset.split_counts() == {"train": 0, "val": 0, "test": 1}
    assert "GO:0009987" in next(iter(dataset.target_values("go_bp").values()))


def test_download_cafa6_dataset_raises_when_kaggle_cli_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cafa_download_module.shutil, "which", lambda name: None)

    with pytest.raises(EmbeddingDependencyError, match="Kaggle CLI"):
        download_cafa6_dataset(tmp_path)


def test_download_cafa6_dataset_explains_forbidden_kaggle_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = command, kwargs
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="403 Client Error: Forbidden for url: https://api.kaggle.com/v1/competitions",
        )

    monkeypatch.setattr(cafa_download_module.shutil, "which", lambda name: "/usr/bin/kaggle")
    monkeypatch.setattr(cafa_download_module.subprocess, "run", fake_run)

    with pytest.raises(EmbeddingInputError, match="accept the competition rules"):
        download_cafa6_dataset(tmp_path)


def test_load_cafa6_dataset_raises_on_missing_kaggle_files(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingInputError, match="Train/train_terms.tsv"):
        load_cafa6_dataset(tmp_path, name="cafa6_bp")


def test_load_ec_dataset_imports_single_label_multiclass_targets(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    dataset = load_dataset("ec:single_ec_1_head", tmp_path, split="train", download=True)

    assert dataset.split_counts() == {"train": 4, "val": 0, "test": 0}
    assert dataset.target_values("single_ec_1") == {
        "ec_full:head:p1": "1",
        "ec_full:head:p5": "1",
        "ec_full:head:p7": "1",
        "ec_full:head:p8": "1",
    }


def test_load_ec_dataset_imports_ec3_subclass_holdout_targets(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    dataset = load_dataset("ec:ec_3_subclass_holdout_head", tmp_path, download=True)
    exact_splits: dict[str, set[str]] = {}
    for example in dataset.examples:
        assert example.metadata is not None
        exact_splits.setdefault(str(example.metadata["heldout_subclass"]), set()).add(example.split)

    assert dataset.split_counts() == {"train": 1, "val": 1, "test": 1}
    assert dataset.target_values("ec_3") == {
        "ec_full:head:p1": "1.1.1",
        "ec_full:head:p5": "1.1.1",
        "ec_full:head:p6": "1.1.1",
    }
    assert all(len(splits) == 1 for splits in exact_splits.values())


def test_load_ec_dataset_accumulates_subclass_holdouts_until_example_target(
    tmp_path: Path,
) -> None:
    _write_ec_holdout_budget_layout(tmp_path)

    dataset = load_dataset("ec:ec_3_subclass_holdout_head", tmp_path, download=True)
    heldout_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for example in dataset.examples:
        assert example.metadata is not None
        heldout_by_split[example.split].add(str(example.metadata["heldout_subclass"]))

    assert dataset.split_counts() == {"train": 20, "val": 3, "test": 3}
    assert heldout_by_split["test"] == {"1.1.1.1", "1.1.1.2"}
    assert heldout_by_split["val"] == {"1.1.1.3"}
    assert heldout_by_split["train"] == {"1.1.1.4"}


def test_load_ec_dataset_imports_ec2_subclass_holdout_targets(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    dataset = load_dataset("ec:ec_2_subclass_holdout_head", tmp_path, download=True)
    subclass_splits: dict[str, set[str]] = {}
    for example in dataset.examples:
        assert example.metadata is not None
        subclass_splits.setdefault(str(example.metadata["heldout_subclass"]), set()).add(example.split)

    assert dataset.split_counts() == {"train": 3, "val": 0, "test": 1}
    assert dataset.target_values("ec_2") == {
        "ec_full:head:p1": "1.1",
        "ec_full:head:p5": "1.1",
        "ec_full:head:p6": "1.1",
        "ec_full:head:p7": "1.1",
    }
    assert all(len(splits) == 1 for splits in subclass_splits.values())


def test_load_ec_dataset_imports_ec1_subclass_holdout_targets(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    dataset = load_dataset("ec:ec_1_subclass_holdout_head", tmp_path, download=True)
    subclass_splits: dict[str, set[str]] = {}
    for example in dataset.examples:
        assert example.metadata is not None
        subclass_splits.setdefault(str(example.metadata["heldout_subclass"]), set()).add(example.split)

    assert dataset.split_counts() == {"train": 4, "val": 0, "test": 1}
    assert dataset.target_values("ec_1") == {
        "ec_full:head:p1": "1",
        "ec_full:head:p5": "1",
        "ec_full:head:p6": "1",
        "ec_full:head:p7": "1",
        "ec_full:head:p8": "1",
    }
    assert all(len(splits) == 1 for splits in subclass_splits.values())


def test_read_ec_manifest_reads_generated_track_metadata(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    manifest = read_ec_manifest(tmp_path, track="head")

    assert manifest["task"] == "ec_full"
    assert manifest["track"] == "head"


def test_load_ec_dataset_raises_on_unsupported_target(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    with pytest.raises(EmbeddingInputError, match="only supports target"):
        load_ec_dataset(tmp_path, name="ec_2_head", target="wrong")


def test_load_ec_dataset_raises_on_unsupported_single_label_target(tmp_path: Path) -> None:
    _write_ec_dataset_layout(tmp_path)

    with pytest.raises(EmbeddingInputError, match="only supports target"):
        load_ec_dataset(tmp_path, name="single_ec_2_head", target="ec_2")


def test_load_dataset_raises_when_entry_is_catalog_only(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingInputError, match="catalog_only"):
        load_dataset("dtu:signalp", tmp_path)


def test_native_peer_registry_lists_sequence_importable_datasets() -> None:
    names = {spec.name for spec in list_peer_native_datasets()}

    assert {"fluorescence", "stability", "solubility", "fold", "secondary_structure"}.issubset(names)


def test_load_peer_dataset_imports_native_protein_lmdb(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    raw = tmp_path / "fluorescence" / "raw" / "fluorescence"
    raw.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (raw / f"fluorescence_{split}.lmdb").touch()

    def fake_read_lmdb_records(path, spec, *, limit=None):
        split_name = path.stem.rsplit("_", 1)[-1]
        rows = [
            {"sequence": "ACDE", "log_fluorescence": 1.5 if split_name != "test" else 2.5},
        ]
        return rows if limit is None else rows[:limit]

    monkeypatch.setattr(peer_module, "_read_lmdb_records", fake_read_lmdb_records)

    dataset = load_peer_dataset(tmp_path, name="Flu", split=["train", "valid", "test"])

    assert dataset.split_counts() == {"train": 1, "val": 1, "test": 1}
    assert dataset.target_values("log_fluorescence") == {
        "fluorescence:train:0": 1.5,
        "fluorescence:valid:0": 1.5,
        "fluorescence:test:0": 2.5,
    }


def test_load_peer_dataset_imports_native_residue_lmdb(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    raw = tmp_path / "secondary_structure" / "raw" / "secondary_structure"
    raw.mkdir(parents=True)
    for split in ("train", "cb513"):
        (raw / f"secondary_structure_{split}.lmdb").touch()

    def fake_read_lmdb_records(path, spec, *, limit=None):
        rows = [
            {"sequence": "ACDE", "ss3": [0, 1, 2, 2], "valid_mask": [1, 1, 0, 1]},
        ]
        return rows if limit is None else rows[:limit]

    monkeypatch.setattr(peer_module, "_read_lmdb_records", fake_read_lmdb_records)

    dataset = load_peer_dataset(tmp_path, name="SSP", split=["train", "cb513"])

    assert isinstance(dataset, ResidueDataset)
    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 1}
    assert dataset.target_values("ss3") == {
        "secondary_structure:train:0": [0, 1, 2, 2],
        "secondary_structure:cb513:0": [0, 1, 2, 2],
    }
    assert dataset.examples[0].mask == [True, True, False, True]


def test_load_peer_dataset_applies_split_limits_during_native_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    raw = tmp_path / "secondary_structure" / "raw" / "secondary_structure"
    raw.mkdir(parents=True)
    for split in ("train", "valid", "cb513"):
        (raw / f"secondary_structure_{split}.lmdb").touch()

    observed_limits = {}

    def fake_read_lmdb_records(path, spec, *, limit=None):
        split_name = path.stem.removeprefix("secondary_structure_")
        observed_limits[split_name] = limit
        rows = [
            {"sequence": "ACDE", "ss3": [0, 1, 2, 2], "valid_mask": [1, 1, 1, 1]},
            {"sequence": "WXYZ", "ss3": [2, 2, 1, 0], "valid_mask": [1, 1, 1, 1]},
        ]
        return rows if limit is None else rows[:limit]

    monkeypatch.setattr(peer_module, "_read_lmdb_records", fake_read_lmdb_records)

    dataset = load_peer_dataset(
        tmp_path,
        name="SSP",
        max_examples_per_split={"train": 1, "val": 1, "test": 1},
    )

    assert observed_limits == {"train": 1, "valid": 1, "cb513": 1}
    assert dataset.split_counts() == {"train": 1, "val": 1, "test": 1}


def test_load_peer_dataset_uses_benchmark_default_holdout_for_fold(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    raw = tmp_path / "fold" / "raw" / "remote_homology"
    raw.mkdir(parents=True)
    for split in ("train", "valid", "test_fold_holdout"):
        (raw / f"remote_homology_{split}.lmdb").touch()

    observed_paths = []

    def fake_read_lmdb_records(path, spec, *, limit=None):
        observed_paths.append(path.name)
        split_name = path.stem.removeprefix("remote_homology_")
        rows = [
            {"sequence": "ACDE", "fold_label": 1 if split_name == "test_fold_holdout" else 0},
        ]
        return rows if limit is None else rows[:limit]

    monkeypatch.setattr(peer_module, "_read_lmdb_records", fake_read_lmdb_records)

    dataset = load_peer_dataset(tmp_path, name="Fold")

    assert observed_paths == [
        "remote_homology_train.lmdb",
        "remote_homology_valid.lmdb",
        "remote_homology_test_fold_holdout.lmdb",
    ]
    assert dataset.split_counts() == {"train": 1, "val": 1, "test": 1}
    assert dataset.examples[-1].metadata is not None
    assert dataset.examples[-1].metadata["original_split"] == "test_fold_holdout"


def test_load_peer_dataset_rejects_pair_tasks_without_torchdrug(tmp_path) -> None:
    with pytest.raises(EmbeddingInputError, match="Native PEER import is not available"):
        load_peer_dataset(tmp_path, name="Yst")


def test_residue_source_registry_covers_requested_sources() -> None:
    names = {source.name for source in list_residue_sources()}

    assert {
        "dbptm",
        "musitedeep",
        "disprot",
        "biolip",
        "biolip_all",
        "biolip_dna",
        "biolip_rna",
        "biolip_pep",
        "biolip_other",
        "metalpdb",
        "scannet_binding",
        "netsurfp",
        "phosphoelm_all",
        "phosphoelm_ltp",
        "phosphoelm_htp",
    } == names
    assert get_residue_source("biolip").import_adapter == "load_biolip_dataset"
    assert get_residue_source("phosphoelm_all").import_adapter == "load_phosphoelm_dataset"
    assert get_residue_source("phosphoelm:ltp").name == "phosphoelm_ltp"
    assert get_residue_source("biolip:dna").name == "biolip_dna"


def test_dbptm_benchmark_registry_contains_requested_archive() -> None:
    spec = get_dbptm_benchmark("PhosphorylationByCDK")

    assert spec.archive_name == "PhosphorylationByCDK.tgz"
    assert spec.positive_sites == 1503
    assert spec.negative_sites == 29823
    assert spec.url.endswith("/benchmark/PhosphorylationByCDK.tgz")
    assert "phosphorylation_by_cdk" in {item.name for item in list_dbptm_benchmarks()}


def test_load_dbptm_benchmark_archive_labels_center_site(tmp_path) -> None:
    archive_path = tmp_path / "PhosphorylationByCDK.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for member_name, payload in {
            "PhosphorylationByCDK/CDK_pos.fasta": ">P1_11\nACDEFGHIKLMNPQRSTVWYA\n",
            "PhosphorylationByCDK/CDK_neg.fasta": ">P2_11\nACDEFGHIKLMNPQRSTVWYA\n",
        }.items():
            data = payload.encode("utf-8")
            info = tarfile.TarInfo(member_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))

    dataset = load_dbptm_benchmark_archive(archive_path, target="phosphorylation_by_cdk")

    assert dataset.target_values("phosphorylation_by_cdk") == {
        "CDK_pos:P1_11": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "CDK_neg:P2_11": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }


def test_load_dbptm_benchmark_dataset_can_download(tmp_path) -> None:
    archive_path = tmp_path / "dbptm" / "benchmark" / "PhosphorylationByCDK.tgz"
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        data = b">P1\nACD\n"
        info = tarfile.TarInfo("PhosphorylationByCDK/CDK_pos.fasta")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    dataset = load_dbptm_benchmark_dataset(tmp_path, name="phosphorylation_by_cdk", download=True)

    assert dataset.target_values("phosphorylation_by_cdk") == {"CDK_pos:P1": [0, 1, 0]}


def test_download_dbptm_benchmark_uses_direct_url(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen_urls: list[str] = []

    def fake_urlretrieve(url: str, filename):
        seen_urls.append(url)
        filename.write_bytes(b"archive")
        return filename, None

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    path = download_dbptm_benchmark(tmp_path, name="PhosphorylationByCDK")

    assert path.name == "PhosphorylationByCDK.tgz"
    assert seen_urls == ["https://biomics.lab.nycu.edu.tw/dbPTM/download/benchmark/PhosphorylationByCDK.tgz"]


def test_download_residue_source_rejects_sources_without_direct_urls(tmp_path) -> None:
    with pytest.raises(EmbeddingInputError, match="does not expose stable direct-download URLs"):
        download_residue_source(tmp_path, name="musitedeep")


def test_download_biolip_variant_uses_browser_user_agent(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen_requests = []

    class FakeResponse:
        def __enter__(self):
            return io.BytesIO(b"payload")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_urlopen(request):
        seen_requests.append(request)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    paths = download_residue_source(tmp_path, name="biolip_dna")

    assert [path.relative_to(tmp_path).as_posix() for path in paths] == [
        "biolip/BioLiP_nr.txt.gz",
        "biolip/protein_nr.fasta.gz",
    ]
    assert [path.read_bytes() for path in paths] == [b"payload", b"payload"]
    assert all(request.headers["User-agent"].startswith("Mozilla/5.0") for request in seen_requests)


def test_download_disprot_current_tsv_uses_current_release_url(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen_urls: list[str] = []

    def fake_urlretrieve(url: str, filename):
        seen_urls.append(url)
        filename.write_text("UniProt ACC\tDisProt ID\n", encoding="utf-8")
        return filename, None

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    path = download_disprot_current_tsv(tmp_path)

    assert path.name == "disprot_current.tsv"
    assert seen_urls == [
        "https://disprot.org/api/v2/download?format=tsv&release=current&term_ontology=IDPO&term_ontology=GO"
    ]


def test_download_disprot_current_json_uses_current_release_url(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen_urls: list[str] = []

    def fake_urlretrieve(url: str, filename):
        seen_urls.append(url)
        filename.write_text('{"data": []}', encoding="utf-8")
        return filename, None

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    path = download_disprot_current_json(tmp_path)

    assert path.name == "disprot_current.json"
    assert seen_urls == [
        "https://disprot.org/api/v2/download?format=json&release=current&term_ontology=IDPO&term_ontology=GO"
    ]
    assert download_residue_source(tmp_path, name="disprot") == [path]


def test_load_musitedeep_fasta_marks_hash_sites(tmp_path) -> None:
    fasta = tmp_path / "ptm.fasta"
    fasta.write_text(">P1\nS#TYK#\n>P2\nACDE\n", encoding="utf-8")

    dataset = load_musitedeep_fasta(fasta, target="phosphorylation")

    assert dataset.target_values("phosphorylation") == {"P1": [1, 0, 0, 1], "P2": [0, 0, 0, 0]}


def test_load_musitedeep_testdata_dataset_uses_local_fastas(tmp_path) -> None:
    data_dir = tmp_path / "musitedeep" / "testdata"
    data_dir.mkdir(parents=True)
    (data_dir / "train_sites.fasta").write_text(">P1\nS#TYK\n", encoding="utf-8")
    (data_dir / "testing_sites.fasta").write_text(">P2\nAC#DE\n", encoding="utf-8")

    dataset = load_musitedeep_testdata_dataset(
        tmp_path,
        target="phosphorylation",
        file_names=["train_sites.fasta", "testing_sites.fasta"],
    )

    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 1}
    assert dataset.target_values("phosphorylation") == {
        "train_sites:P1": [1, 0, 0, 0],
        "testing_sites:P2": [0, 1, 0, 0],
    }


def test_download_musitedeep_testdata_uses_github_file_list(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    records = [
            {
                "name": "ptm_sites.fasta",
                "type": "file",
                "download_url": "https://example.test/ptm_sites.fasta",
            },
            {
                "name": "README.md",
                "type": "file",
                "download_url": "https://example.test/README.md",
            },
        ]

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self) -> bytes:
            return json.dumps(records).encode("utf-8")

    def fake_urlopen(request):
        return _FakeResponse()

    def fake_urlretrieve(url: str, filename):
        path = filename
        path.write_text(">P1\nS#TYK\n", encoding="utf-8")
        return path, None

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    paths = download_musitedeep_testdata(tmp_path)

    assert [path.name for path in paths] == ["ptm_sites.fasta"]
    assert paths[0].read_text(encoding="utf-8") == ">P1\nS#TYK\n"


def test_load_interval_residue_tsv_merges_site_rows(tmp_path) -> None:
    path = tmp_path / "sites.tsv"
    path.write_text(
        "\n".join(
            [
                "id\tsequence\tstart\tend\tsplit",
                "P1\tACDEFG\t2\t2\ttrain",
                "P1\tACDEFG\t5\t6\ttrain",
                "P2\tSTYK\t1\t1\ttest",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_interval_residue_tsv(path, target="site")

    assert dataset.target_values("site") == {"P1": [0, 1, 0, 0, 1, 1], "P2": [1, 0, 0, 0]}
    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 1}


def test_load_residue_label_table_supports_compact_labels(tmp_path) -> None:
    path = tmp_path / "labels.tsv"
    path.write_text("id\tsequence\tlabels\tsplit\nP1\tACDE\tHECC\tval\n", encoding="utf-8")

    dataset = load_residue_label_table(path, target="ss")

    assert dataset.target_values("ss") == {"P1": ["H", "E", "C", "C"]}
    assert dataset.split_counts() == {"train": 0, "val": 1, "test": 0}


def test_load_disprot_json_expands_regions(tmp_path) -> None:
    path = tmp_path / "disprot.json"
    path.write_text(
        """
        {
          "data": [
            {
              "acc": "DP0001",
              "sequence": "ACDEFG",
              "regions": [{"start": 2, "end": 3}, {"start": 6, "end": 6}]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    dataset = load_disprot_json(path)

    assert dataset.target_values("disorder") == {"DP0001": [0, 1, 1, 0, 0, 1]}


def test_load_disprot_json_groups_current_disorder_rows_and_ignores_other_states(tmp_path) -> None:
    path = tmp_path / "disprot_current.json"
    path.write_text(
        """
        {
          "data": [
            {
              "acc": "P1",
              "disprot_id": "DP1",
              "sequence": "ABCDEFGH",
              "dataset": ["demo"],
              "regions": {
                "region_id": "DP1r1",
                "start": 2,
                "end": 4,
                "term_namespace": "Structural state",
                "term_name": "disorder"
              }
            },
            {
              "acc": "P1",
              "disprot_id": "DP1",
              "sequence": "ABCDEFGH",
              "dataset": ["demo", "extra"],
              "regions": {
                "region_id": "DP1r2",
                "start": 7,
                "end": 7,
                "term_namespace": "Structural state",
                "term_name": "disorder"
              }
            },
            {
              "acc": "P1",
              "disprot_id": "DP1",
              "sequence": "ABCDEFGH",
              "regions": {
                "region_id": "DP1r3",
                "start": 5,
                "end": 6,
                "term_namespace": "Structural state",
                "term_name": "order"
              }
            },
            {
              "acc": "P2",
              "disprot_id": "DP2",
              "sequence": "ABCDEFGH",
              "regions": {
                "region_id": "DP2r1",
                "start": 1,
                "end": 2,
                "term_namespace": "Function",
                "term_name": "disorder"
              }
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    dataset = load_disprot_json(path, split="train")

    assert dataset.target_values("disorder") == {"P1": [0, 1, 1, 1, 0, 0, 1, 0]}
    assert dataset.split_counts() == {"train": 1, "val": 0, "test": 0}
    metadata = dataset.examples[0].metadata
    assert metadata is not None
    assert metadata["dataset_tags"] == ("demo", "extra")
    assert metadata["disorder_interval_count"] == 2
    assert metadata["disorder_content_bin"] == "30-60%"


def test_load_disprot_json_assigns_stratified_splits_for_current_schema(tmp_path) -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "acc": f"P{index:02d}",
                "disprot_id": f"DP{index:02d}",
                "sequence": "A" * 20,
                "dataset": ["demo"],
                "regions": {
                    "region_id": f"DP{index:02d}r1",
                    "start": 1,
                    "end": 1,
                    "term_namespace": "Structural state",
                    "term_name": "disorder",
                },
            }
        )
    path = tmp_path / "disprot_current.json"
    path.write_text(json.dumps({"data": rows}), encoding="utf-8")

    dataset = load_disprot_json(path)

    assert dataset.split_counts() == {"train": 16, "val": 2, "test": 2}
    assert {example.metadata["dataset_stratum"] for example in dataset.examples if example.metadata} == {"demo"}


def test_load_disprot_json_rejects_conflicting_current_sequences(tmp_path) -> None:
    path = tmp_path / "disprot_current.json"
    path.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "acc": "P1",
                        "sequence": "AAAA",
                        "regions": {
                            "start": 1,
                            "end": 1,
                            "term_namespace": "Structural state",
                            "term_name": "disorder",
                        },
                    },
                    {
                        "acc": "P1",
                        "sequence": "CCCC",
                        "regions": {
                            "start": 2,
                            "end": 2,
                            "term_namespace": "Structural state",
                            "term_name": "disorder",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EmbeddingInputError, match="Conflicting sequences"):
        load_disprot_json(path)


def test_load_disprot_tsv_region_mode_uses_matching_disorder_rows(tmp_path) -> None:
    path = tmp_path / "disprot_current.tsv"
    path.write_text(
        "\n".join(
            [
                "UniProt ACC\tDisProt ID\tStart\tEnd\tRegion sequence\tTerm ID\tTerm name",
                "P1\tDP1\t2\t4\tCDE\tIDPO:0000002\tdisorder",
                "P1\tDP1\t6\t7\tFG\tGO:0008150\tbiological_process",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_disprot_tsv(path, region_mode=True)

    assert dataset.target_values("disorder") == {"DP1": [1, 1, 1]}


def test_load_disprot_tsv_expands_intervals_with_supplied_sequences(tmp_path) -> None:
    path = tmp_path / "disprot_current.tsv"
    path.write_text(
        "\n".join(
            [
                "UniProt ACC\tDisProt ID\tStart\tEnd\tRegion sequence\tTerm ID\tTerm name",
                "P1\tDP1\t2\t4\tCDE\tIDPO:0000002\tdisorder",
                "P1\tDP1\t7\t7\tG\tIDPO:0000002\tdisorder",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_disprot_tsv(path, protein_sequences={"P1": "ABCDEFG"})

    assert dataset.target_values("disorder") == {"P1": [0, 1, 1, 1, 0, 0, 1]}


def test_load_residue_source_dataset_imports_disprot_current_tsv(tmp_path) -> None:
    data_dir = tmp_path / "disprot"
    data_dir.mkdir()
    (data_dir / "disprot_current.tsv").write_text(
        "\n".join(
            [
                "UniProt ACC\tDisProt ID\tStart\tEnd\tRegion sequence\tTerm ID\tTerm name",
                "P1\tDP1\t2\t4\tCDE\tIDPO:0000002\tdisorder",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = load_residue_source_dataset(tmp_path, name="disprot")

    assert dataset.target_values("disorder") == {"DP1": [1, 1, 1]}


def test_load_biolip_dataset_uses_renumbered_binding_positions(tmp_path) -> None:
    annotation = tmp_path / "BioLiP_nr.txt"
    columns = [
        "1abc",
        "A",
        "1.0",
        "BS01",
        "LIG",
        "B",
        "1",
        "A100 C102",
        "A1 C3",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "P12345",
        "123",
        "1",
        "ACDE",
    ]
    annotation.write_text("\t".join(columns) + "\n", encoding="utf-8")

    dataset = load_biolip_dataset(annotation)

    assert dataset.target_values("ligand_binding_site") == {"1abcA": [1, 0, 1, 0]}


def test_load_biolip_dataset_filters_ligand_classes(tmp_path) -> None:
    annotation = tmp_path / "BioLiP_nr.txt"

    def row(pdb_id: str, ligand: str, renumbered_positions: str) -> str:
        columns = [
            pdb_id,
            "A",
            "1.0",
            "BS01",
            ligand,
            "B",
            "1",
            "-",
            renumbered_positions,
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "P12345",
            "123",
            "1",
            "ACDE",
        ]
        return "\t".join(columns)

    annotation.write_text(
        "\n".join(
            [
                row("1dna", "dna", "A1"),
                row("1rna", "rna", "C2"),
                row("1pep", "peptide", "D3"),
                row("1oth", "ZN", "E4"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dna = load_biolip_dataset(annotation, ligand_class="dna")
    rna = load_biolip_dataset(annotation, ligand_class="rna")
    peptide = load_biolip_dataset(annotation, ligand_class="pep")
    other = load_biolip_dataset(annotation, ligand_class="other")

    assert dna.target_values("dna_binding_site") == {"1dnaA": [1, 0, 0, 0]}
    assert rna.target_values("rna_binding_site") == {"1rnaA": [0, 1, 0, 0]}
    assert peptide.target_values("peptide_binding_site") == {"1pepA": [0, 0, 1, 0]}
    assert other.target_values("other_ligand_binding_site") == {"1othA": [0, 0, 0, 1]}


def test_load_biolip_dataset_can_assign_deterministic_splits(tmp_path) -> None:
    annotation = tmp_path / "BioLiP_nr.txt"
    rows = []
    for index in range(12):
        rows.append(
            "\t".join(
                [
                    f"{index:04d}",
                    "A",
                    "1.0",
                    "BS01",
                    "LIG",
                    "B",
                    "1",
                    "A100",
                    "A1",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    f"P{index:05d}",
                    "123",
                    "1",
                    "ACDE",
                ]
            )
        )
    annotation.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = load_biolip_dataset(annotation, split=None)

    assert dataset.split_counts() == {"train": 10, "val": 1, "test": 1}
    assert dataset.target_values("ligand_binding_site")["0000A"] == [1, 0, 0, 0]


def test_load_residue_source_dataset_imports_biolip_layout(tmp_path) -> None:
    data_dir = tmp_path / "biolip"
    data_dir.mkdir()
    columns = [
        "1abc",
        "A",
        "1.0",
        "BS01",
        "LIG",
        "B",
        "1",
        "A100 C102",
        "A1 C3",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "P12345",
        "123",
        "1",
        "ACDE",
    ]
    (data_dir / "BioLiP_nr.txt.gz").write_bytes(gzip.compress(("\t".join(columns) + "\n").encode("utf-8")))

    dataset = load_residue_source_dataset(tmp_path, name="biolip", split="test")

    assert dataset.split_counts() == {"train": 0, "val": 0, "test": 1}
    assert dataset.target_values("ligand_binding_site") == {"1abcA": [1, 0, 1, 0]}


def test_load_dataset_accepts_splitter_for_generated_residue_source_splits(tmp_path) -> None:
    data_dir = tmp_path / "biolip"
    data_dir.mkdir()
    rows = []
    for index in range(12):
        columns = [
            f"{index:04d}",
            "A",
            "1.0",
            "BS01",
            "LIG",
            "B",
            "1",
            "A100",
            "A1",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            f"P{index:05d}",
            "123",
            "1",
            "ACDE",
        ]
        rows.append("\t".join(columns))
    (data_dir / "BioLiP_nr.txt.gz").write_bytes(
        gzip.compress(("\n".join(rows) + "\n").encode("utf-8"))
    )

    dataset = load_dataset(
        "biolip:all",
        tmp_path,
        splitter=HashDatasetSplitter(
            salt="custom-biolip-split-v1",
            ratios={"train": 0.5, "val": 0.25, "test": 0.25},
        ),
    )

    assert dataset.split_counts() == {"train": 6, "val": 3, "test": 3}


def test_load_dataset_explicit_split_takes_precedence_over_splitter(tmp_path) -> None:
    data_dir = tmp_path / "biolip"
    data_dir.mkdir()
    columns = [
        "1abc",
        "A",
        "1.0",
        "BS01",
        "LIG",
        "B",
        "1",
        "A100 C102",
        "A1 C3",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "P12345",
        "123",
        "1",
        "ACDE",
    ]
    (data_dir / "BioLiP_nr.txt.gz").write_bytes(
        gzip.compress(("\t".join(columns) + "\n").encode("utf-8"))
    )

    dataset = load_dataset(
        "biolip:all",
        tmp_path,
        split="test",
        splitter=HashDatasetSplitter(salt="ignored-biolip-split-v1"),
    )

    assert dataset.split_counts() == {"train": 0, "val": 0, "test": 1}


def test_load_residue_source_dataset_imports_biolip_variant_from_shared_layout(tmp_path) -> None:
    data_dir = tmp_path / "biolip"
    data_dir.mkdir()
    rows = []
    for ligand, position in [("dna", "A1"), ("peptide", "C3")]:
        columns = [
            f"1{ligand[:2]}",
            "A",
            "1.0",
            "BS01",
            ligand,
            "B",
            "1",
            "-",
            position,
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "P12345",
            "123",
            "1",
            "ACDE",
        ]
        rows.append("\t".join(columns))
    (data_dir / "BioLiP_nr.txt.gz").write_bytes(gzip.compress(("\n".join(rows) + "\n").encode("utf-8")))

    dataset = load_residue_source_dataset(tmp_path, name="biolip_pep")

    assert dataset.ids() == ["1peA"]
    assert dataset.target_values("peptide_binding_site") == {"1peA": [0, 0, 1, 0]}


def test_load_phosphoelm_dataset_collapses_full_protein_sites_and_filters_evidence(tmp_path) -> None:
    dump = tmp_path / "phosphoELM_all_2015-04.dump"
    dump.write_text(
        "\n".join(
            [
                "acc\tsequence\tposition\tcode\tpmids\tkinases\tsource\tspecies\tentry_date",
                "P1\tMSTYAS\t2\tS\t1\t\tLTP\tHomo sapiens\t2015-01-01",
                "P1\tMSTYAS\t3\tT\t2\t\tHTP\tHomo sapiens\t2015-01-02",
                "P1\tMSTYAS\t3\tT\t3\t\tHTP\tHomo sapiens\t2015-01-03",
                "P2\tAAAY\t4\tY\t4\tSRC\tHTP\tMus musculus\t2015-01-04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    all_dataset = load_phosphoelm_dataset(dump, split="train")
    ltp_dataset = load_phosphoelm_dataset(dump, split="train", source_filter="LTP")
    htp_dataset = load_phosphoelm_dataset(dump, split="train", source_filter="HTP")
    serine_dataset = load_phosphoelm_dataset(dump, split="train", residue_codes=("S",))

    assert all_dataset.target_values("phosphorylation_site") == {
        "P1": [0, 1, 1, 0, 0, 0],
        "P2": [0, 0, 0, 1],
    }
    assert all_dataset.examples[0].mask == [False, True, True, True, False, True]
    assert ltp_dataset.target_values("phosphorylation_site") == {"P1": [0, 1, 0, 0, 0, 0]}
    assert htp_dataset.target_values("phosphorylation_site") == {
        "P1": [0, 0, 1, 0, 0, 0],
        "P2": [0, 0, 0, 1],
    }
    assert serine_dataset.target_values("phosphorylation_site") == {"P1": [0, 1, 0, 0, 0, 0]}
    assert serine_dataset.examples[0].mask == [False, True, False, False, False, True]
    assert htp_dataset.examples[0].metadata is not None
    assert htp_dataset.examples[0].metadata["positive_site_count"] == 1
    assert htp_dataset.examples[0].metadata["positive_residue_counts"] == {"S": 0, "T": 1, "Y": 0}
    assert htp_dataset.examples[0].metadata["positive_residue_stratum"] == "T"


def test_load_phosphoelm_dataset_stratifies_default_splits_by_species_and_residue(tmp_path) -> None:
    dump = tmp_path / "phosphoELM_all_2015-04.dump"
    rows = ["acc\tsequence\tposition\tcode\tpmids\tkinases\tsource\tspecies\tentry_date"]
    for index in range(10):
        rows.append(f"HS{index}\tMSTYAS\t2\tS\t1\t\tLTP\tHomo sapiens\t2015-01-01")
    for index in range(10):
        rows.append(f"MM{index}\tMSTYAS\t3\tT\t1\t\tLTP\tMus musculus\t2015-01-01")
    dump.write_text("\n".join(rows) + "\n", encoding="utf-8")

    dataset = load_phosphoelm_dataset(dump)
    species_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    residues_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for example in dataset.examples:
        assert example.metadata is not None
        species_by_split[example.split].add(str(example.metadata["species"]))
        residues_by_split[example.split].add(str(example.metadata["positive_residue_stratum"]))

    assert dataset.split_counts() == {"train": 16, "val": 2, "test": 2}
    assert species_by_split == {
        "train": {"Homo sapiens", "Mus musculus"},
        "val": {"Homo sapiens", "Mus musculus"},
        "test": {"Homo sapiens", "Mus musculus"},
    }
    assert residues_by_split == {"train": {"S", "T"}, "val": {"S", "T"}, "test": {"S", "T"}}


def test_load_residue_source_dataset_imports_phosphoelm_variants_from_shared_layout(tmp_path) -> None:
    data_dir = tmp_path / "phosphoelm"
    data_dir.mkdir()
    payload = (
        "acc\tsequence\tposition\tcode\tpmids\tkinases\tsource\tspecies\tentry_date\n"
        "P1\tMSTYAS\t2\tS\t1\t\tLTP\tHomo sapiens\t2015-01-01\n"
        "P1\tMSTYAS\t3\tT\t2\t\tHTP\tHomo sapiens\t2015-01-02\n"
    ).encode("utf-8")
    archive_path = data_dir / "phosphoELM_all_latest.dump.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("phosphoELM_all_2015-04.dump")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    all_dataset = load_residue_source_dataset(tmp_path, name="phosphoelm:all", split="test")
    ltp_dataset = load_residue_source_dataset(tmp_path, name="phosphoelm:ltp", split="test")
    htp_dataset = load_residue_source_dataset(tmp_path, name="phosphoelm:htp", split="test")

    assert all_dataset.split_counts() == {"train": 0, "val": 0, "test": 1}
    assert all_dataset.target_values("phosphorylation_site") == {"P1": [0, 1, 1, 0, 0, 0]}
    assert ltp_dataset.target_values("phosphorylation_site") == {"P1": [0, 1, 0, 0, 0, 0]}
    assert htp_dataset.target_values("phosphorylation_site") == {"P1": [0, 0, 1, 0, 0, 0]}


def test_load_residue_source_dataset_finds_phosphoelm_archive_from_nested_data_root(tmp_path) -> None:
    data_root = tmp_path / "data" / "probing"
    (data_root / "phosphoelm").mkdir(parents=True)
    payload = (
        "acc\tsequence\tposition\tcode\tpmids\tkinases\tsource\tspecies\tentry_date\n"
        "P1\tMSTYAS\t2\tS\t1\t\tLTP\tHomo sapiens\t2015-01-01\n"
    ).encode("utf-8")
    archive_path = tmp_path / "phosphoELM_all_latest.dump.tgz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("phosphoELM_all_2015-04.dump")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    dataset = load_residue_source_dataset(data_root, name="phosphoelm_all", split="train")

    assert dataset.target_values("phosphorylation_site") == {"P1": [0, 1, 0, 0, 0, 0]}
