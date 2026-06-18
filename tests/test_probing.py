from __future__ import annotations

import io
import gzip
import json
import tarfile

import pytest

from CBBIO import (
    PredictionSpec,
    ProbeSpec,
    ProteinDataset,
    ProteinExample,
    ResidueDataset,
    ResidueExample,
    Task,
    get_dataset_catalog_entry,
    download_dbptm_benchmark,
    download_disprot_current_json,
    download_disprot_current_tsv,
    download_musitedeep_testdata,
    download_residue_source,
    get_dbptm_benchmark,
    get_dtu_service,
    list_dbptm_benchmarks,
    list_dataset_catalog,
    list_dtu_services,
    get_peer_task,
    get_residue_source,
    list_peer_native_datasets,
    list_peer_tasks,
    list_residue_dataset_catalog,
    list_residue_sources,
    load_biolip_dataset,
    load_dbptm_benchmark_archive,
    load_dbptm_benchmark_dataset,
    load_disprot_json,
    load_disprot_tsv,
    load_flip_csv,
    load_flip_dataset,
    load_interval_residue_tsv,
    load_musitedeep_fasta,
    load_musitedeep_testdata_dataset,
    load_peer_dataset,
    load_phosphoelm_dataset,
    load_residue_label_table,
    load_residue_label_csv,
    load_residue_source_dataset,
    run_task_on_layer,
    search_dataset_catalog,
    train_and_evaluate_residue_probe,
)
import CBBIO.probing.peer as peer_module
from CBBIO.probing.metrics import binary_metrics
from CBBIO.probing.metrics import spearmanr
import CBBIO.probing.residue_sources as residue_sources_module
from CBBIO.embeddings import EmbeddingInputError


torch = pytest.importorskip("torch")


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
    assert result.predictions == {"nt": 0, "pt": 1}


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
    search_results = search_dataset_catalog("phosphorylation cdk residue", status="ready")

    assert get_dataset_catalog_entry("dbptm:phosphorylation_by_cdk").loader == "load_dbptm_benchmark_dataset"
    assert get_dataset_catalog_entry("source:disprot").status == "ready"
    assert get_dataset_catalog_entry("source:disprot").loader == "load_residue_source_dataset"
    assert get_dataset_catalog_entry("secondary_structure").id == "peer:secondary_structure"
    assert "dbptm:phosphorylation_by_cdk" in {entry.id for entry in ready_residue_ptm}
    assert search_results[0].id == "dbptm:phosphorylation_by_cdk"
    assert "peer:secondary_structure" in {entry.id for entry in search_dataset_catalog("secondary structure")}


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


def test_load_dbptm_benchmark_dataset_can_download(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    archive_path = tmp_path / "dbptm" / "benchmark" / "PhosphorylationByCDK.tgz"
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        data = b">P1\nACD\n"
        info = tarfile.TarInfo("PhosphorylationByCDK/CDK_pos.fasta")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    def fake_download(root, *, name: str = "phosphorylation_by_cdk", force: bool = False):
        return archive_path

    monkeypatch.setattr(residue_sources_module, "download_dbptm_benchmark", fake_download)

    dataset = load_dbptm_benchmark_dataset(tmp_path, name="phosphorylation_by_cdk", download=True)

    assert dataset.target_values("phosphorylation_by_cdk") == {"CDK_pos:P1": [0, 1, 0]}


def test_download_dbptm_benchmark_uses_direct_url(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen_urls: list[str] = []

    def fake_urlretrieve(url: str, filename):
        seen_urls.append(url)
        filename.write_bytes(b"archive")
        return filename, None

    monkeypatch.setattr(residue_sources_module, "urlretrieve", fake_urlretrieve)

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

    monkeypatch.setattr(residue_sources_module, "urlopen", fake_urlopen)

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

    monkeypatch.setattr(residue_sources_module, "urlretrieve", fake_urlretrieve)

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

    monkeypatch.setattr(residue_sources_module, "urlretrieve", fake_urlretrieve)

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
    def fake_github_contents(api_url: str) -> list[object]:
        return [
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

    def fake_urlretrieve(url: str, filename):
        path = filename
        path.write_text(">P1\nS#TYK\n", encoding="utf-8")
        return path, None

    monkeypatch.setattr(residue_sources_module, "_github_contents", fake_github_contents)
    monkeypatch.setattr(residue_sources_module, "urlretrieve", fake_urlretrieve)

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
