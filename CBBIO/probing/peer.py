"""PEER benchmark metadata and native dataset importers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import pickle
import tarfile
from typing import Any, Dict, List, Literal, Sequence, Tuple, cast
from urllib.parse import urlparse
from urllib.request import urlretrieve
import zipfile

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError

from .datasets import ObjectiveName, ProteinDataset, ProteinExample, ResidueDataset, ResidueExample, SplitName
from .flip import load_flip_dataset


PeerTaskLevel = Literal["protein", "residue", "residue_pair", "protein_pair", "protein_ligand"]

PEER_CITATION = (
    "Xu, Minghao and Zhang, Zuobai and Lu, Jiarui and Zhu, Zhaocheng and Zhang, "
    "Yangtian and Ma, Chang and Liu, Runcheng and Tang, Jian. "
    "PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding. "
    "arXiv preprint arXiv:2206.02096, 2022."
)


@dataclass(frozen=True)
class PeerTaskMetadata:
    """Metadata for a PEER benchmark task."""

    name: str
    short_name: str
    display_name: str
    level: PeerTaskLevel
    objective: ObjectiveName
    metric: str
    category: str
    dataset_class: str
    split_counts: Tuple[int, int, int] | None = None
    target: str = "target"
    source: str = "torchdrug"
    current_loader: str | None = None


@dataclass(frozen=True)
class PeerNativeDatasetSpec:
    """Native download/import metadata for TorchDrug-format PEER LMDB archives."""

    name: str
    url: str
    md5: str
    archive_subdir: str
    lmdb_prefix: str
    splits: Tuple[str, ...]
    target_fields: Tuple[str, ...]
    benchmark_splits: Tuple[str, ...] | None = None
    sequence_field: str = "primary"
    number_field: str = "num_examples"


PEER_TASKS: Dict[str, PeerTaskMetadata] = {
    "gb1": PeerTaskMetadata(
        name="gb1",
        short_name="GB1",
        display_name="GB1 fitness prediction",
        level="protein",
        objective="regression",
        metric="spearmanr",
        category="function_prediction",
        dataset_class="peer.flip.GB1",
        split_counts=(381, 43, 8309),
        source="flip",
        current_loader="load_flip_dataset",
    ),
    "aav": PeerTaskMetadata(
        name="aav",
        short_name="AAV",
        display_name="AAV fitness prediction",
        level="protein",
        objective="regression",
        metric="spearmanr",
        category="function_prediction",
        dataset_class="peer.flip.AAV",
        split_counts=(28626, 3181, 50776),
        source="flip",
        current_loader="load_flip_dataset",
    ),
    "thermostability": PeerTaskMetadata(
        name="thermostability",
        short_name="Thermo",
        display_name="Thermostability prediction",
        level="protein",
        objective="regression",
        metric="spearmanr",
        category="function_prediction",
        dataset_class="peer.flip.Thermostability",
        split_counts=(5149, 643, 1366),
        source="flip",
        current_loader="load_flip_dataset",
    ),
    "fluorescence": PeerTaskMetadata(
        name="fluorescence",
        short_name="Flu",
        display_name="Fluorescence prediction",
        level="protein",
        objective="regression",
        metric="spearmanr",
        category="function_prediction",
        dataset_class="torchdrug.datasets.Fluorescence",
        split_counts=(21446, 5362, 27217),
        target="log_fluorescence",
    ),
    "stability": PeerTaskMetadata(
        name="stability",
        short_name="Sta",
        display_name="Stability prediction",
        level="protein",
        objective="regression",
        metric="spearmanr",
        category="function_prediction",
        dataset_class="torchdrug.datasets.Stability",
        split_counts=(53571, 2512, 12851),
        target="stability_score",
    ),
    "beta_lactamase": PeerTaskMetadata(
        name="beta_lactamase",
        short_name="beta-lac",
        display_name="Beta-lactamase activity prediction",
        level="protein",
        objective="regression",
        metric="spearmanr",
        category="function_prediction",
        dataset_class="torchdrug.datasets.BetaLactamase",
        split_counts=(4158, 520, 520),
        target="scaled_effect1",
    ),
    "solubility": PeerTaskMetadata(
        name="solubility",
        short_name="Sol",
        display_name="Solubility prediction",
        level="protein",
        objective="binary",
        metric="accuracy",
        category="function_prediction",
        dataset_class="torchdrug.datasets.Solubility",
        split_counts=(62478, 6942, 1999),
        target="solubility",
    ),
    "subcellular_localization": PeerTaskMetadata(
        name="subcellular_localization",
        short_name="Sub",
        display_name="Subcellular localization prediction",
        level="protein",
        objective="multiclass",
        metric="accuracy",
        category="localization_prediction",
        dataset_class="torchdrug.datasets.SubcellularLocalization",
        split_counts=(8945, 2248, 2768),
        target="localization",
    ),
    "binary_localization": PeerTaskMetadata(
        name="binary_localization",
        short_name="Bin",
        display_name="Binary localization prediction",
        level="protein",
        objective="binary",
        metric="accuracy",
        category="localization_prediction",
        dataset_class="torchdrug.datasets.BinaryLocalization",
        split_counts=(5161, 1727, 1746),
        target="localization",
    ),
    "contact": PeerTaskMetadata(
        name="contact",
        short_name="Cont",
        display_name="Contact prediction",
        level="residue_pair",
        objective="binary",
        metric="l5_precision",
        category="structure_prediction",
        dataset_class="torchdrug.datasets.ProteinNet",
        split_counts=(25299, 224, 40),
    ),
    "fold": PeerTaskMetadata(
        name="fold",
        short_name="Fold",
        display_name="Fold classification",
        level="protein",
        objective="multiclass",
        metric="accuracy",
        category="structure_prediction",
        dataset_class="torchdrug.datasets.Fold",
        split_counts=(12312, 736, 718),
        target="fold_label",
    ),
    "secondary_structure": PeerTaskMetadata(
        name="secondary_structure",
        short_name="SSP",
        display_name="Secondary structure prediction",
        level="residue",
        objective="multiclass",
        metric="accuracy",
        category="structure_prediction",
        dataset_class="torchdrug.datasets.SecondaryStructure",
        split_counts=(8678, 2170, 513),
        target="ss3",
    ),
    "yeast_ppi": PeerTaskMetadata(
        name="yeast_ppi",
        short_name="Yst",
        display_name="Yeast PPI prediction",
        level="protein_pair",
        objective="binary",
        metric="accuracy",
        category="protein_protein_interaction_prediction",
        dataset_class="torchdrug.datasets.YeastPPI",
        split_counts=(1668, 131, 373),
    ),
    "human_ppi": PeerTaskMetadata(
        name="human_ppi",
        short_name="Hum",
        display_name="Human PPI prediction",
        level="protein_pair",
        objective="binary",
        metric="accuracy",
        category="protein_protein_interaction_prediction",
        dataset_class="torchdrug.datasets.HumanPPI",
        split_counts=(6844, 277, 227),
    ),
    "ppi_affinity": PeerTaskMetadata(
        name="ppi_affinity",
        short_name="Aff",
        display_name="PPI affinity prediction",
        level="protein_pair",
        objective="regression",
        metric="rmse",
        category="protein_protein_interaction_prediction",
        dataset_class="torchdrug.datasets.PPIAffinity",
        split_counts=(2127, 212, 343),
    ),
    "pdbbind": PeerTaskMetadata(
        name="pdbbind",
        short_name="PDB",
        display_name="Affinity prediction on PDBbind",
        level="protein_ligand",
        objective="regression",
        metric="rmse",
        category="protein_ligand_interaction_prediction",
        dataset_class="torchdrug.datasets.PDBBind",
        split_counts=(16436, 937, 285),
    ),
    "bindingdb": PeerTaskMetadata(
        name="bindingdb",
        short_name="BDB",
        display_name="Affinity prediction on BindingDB",
        level="protein_ligand",
        objective="regression",
        metric="rmse",
        category="protein_ligand_interaction_prediction",
        dataset_class="torchdrug.datasets.BindingDB",
        split_counts=(7900, 878, 5230),
    ),
}

PEER_TASK_ALIASES: Dict[str, str] = {task.short_name.lower(): key for key, task in PEER_TASKS.items()}
PEER_TASK_ALIASES.update(
    {
        "thermo": "thermostability",
        "beta-lac": "beta_lactamase",
        "bet-lac": "beta_lactamase",
        "sub": "subcellular_localization",
        "bin": "binary_localization",
        "cont": "contact",
        "ssp": "secondary_structure",
        "yst": "yeast_ppi",
        "hum": "human_ppi",
        "aff": "ppi_affinity",
        "pdb": "pdbbind",
        "bdb": "bindingdb",
    }
)


PEER_NATIVE_DATASETS: Dict[str, PeerNativeDatasetSpec] = {
    "fluorescence": PeerNativeDatasetSpec(
        name="fluorescence",
        url="http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz",
        md5="d63d1d51ec8c20ff0d981e4cbd67457a",
        archive_subdir="fluorescence",
        lmdb_prefix="fluorescence",
        splits=("train", "valid", "test"),
        target_fields=("log_fluorescence",),
    ),
    "stability": PeerNativeDatasetSpec(
        name="stability",
        url="http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz",
        md5="aa1e06eb5a59e0ecdae581e9ea029675",
        archive_subdir="stability",
        lmdb_prefix="stability",
        splits=("train", "valid", "test"),
        target_fields=("stability_score",),
    ),
    "beta_lactamase": PeerNativeDatasetSpec(
        name="beta_lactamase",
        url="https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/beta_lactamase.tar.gz",
        md5="65766a3969cc0e94b101d4063d204ba4",
        archive_subdir="beta_lactamase",
        lmdb_prefix="beta_lactamase",
        splits=("train", "valid", "test"),
        target_fields=("scaled_effect1",),
    ),
    "solubility": PeerNativeDatasetSpec(
        name="solubility",
        url="https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/solubility.tar.gz",
        md5="8a8612b7bfa2ed80375db6e465ccf77e",
        archive_subdir="solubility",
        lmdb_prefix="solubility",
        splits=("train", "valid", "test"),
        target_fields=("solubility",),
    ),
    "subcellular_localization": PeerNativeDatasetSpec(
        name="subcellular_localization",
        url="https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization.tar.gz",
        md5="37cb6138b8d4603512530458b7c8a77d",
        archive_subdir="subcellular_localization",
        lmdb_prefix="subcellular_localization",
        splits=("train", "valid", "test"),
        target_fields=("localization",),
    ),
    "binary_localization": PeerNativeDatasetSpec(
        name="binary_localization",
        url="https://miladeepgraphlearningproteindata.s3.us-east-2.amazonaws.com/peerdata/subcellular_localization_2.tar.gz",
        md5="5d2309bf1c0c2aed450102578e434f4e",
        archive_subdir="subcellular_localization_2",
        lmdb_prefix="subcellular_localization_2",
        splits=("train", "valid", "test"),
        target_fields=("localization",),
    ),
    "fold": PeerNativeDatasetSpec(
        name="fold",
        url="http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/remote_homology.tar.gz",
        md5="1d687bdeb9e3866f77504d6079eed00a",
        archive_subdir="remote_homology",
        lmdb_prefix="remote_homology",
        splits=("train", "valid", "test_fold_holdout", "test_family_holdout", "test_superfamily_holdout"),
        target_fields=("fold_label",),
        benchmark_splits=("train", "valid", "test_fold_holdout"),
    ),
    "secondary_structure": PeerNativeDatasetSpec(
        name="secondary_structure",
        url="http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/secondary_structure.tar.gz",
        md5="2f61e8e09c215c032ef5bc8b910c8e97",
        archive_subdir="secondary_structure",
        lmdb_prefix="secondary_structure",
        splits=("train", "valid", "casp12", "ts115", "cb513"),
        target_fields=("ss3", "valid_mask"),
        benchmark_splits=("train", "valid", "cb513"),
    ),
}


def list_peer_tasks(*, level: PeerTaskLevel | None = None) -> List[PeerTaskMetadata]:
    """Return PEER task metadata, optionally filtered by task level."""

    values = list(PEER_TASKS.values())
    if level is not None:
        values = [task for task in values if task.level == level]
    return values


def get_peer_task(name: str) -> PeerTaskMetadata:
    """Return metadata for a PEER task by canonical name or table short name."""

    key = str(name).strip().lower()
    resolved = PEER_TASK_ALIASES.get(key, key)
    task = PEER_TASKS.get(resolved)
    if task is None:
        supported = ", ".join(sorted(PEER_TASKS))
        raise EmbeddingInputError(f"Unknown PEER task {name!r}. Supported values: {supported}.")
    return task


def load_peer_dataset(
    root: str | Path,
    *,
    name: str,
    split: str | Sequence[str] | None = None,
    download: bool = False,
) -> ProteinDataset | ResidueDataset:
    """Load a PEER dataset without depending on TorchDrug.

    Native LMDB import is available for PEER sequence datasets represented by
    ``ProteinDataset`` or ``ResidueDataset``. FLIP tasks dispatch to the native
    FLIP CSV importer and require an explicit FLIP split protocol.
    """

    task = get_peer_task(name)
    if task.source == "flip":
        if split is None or not isinstance(split, str):
            raise EmbeddingInputError(f"FLIP PEER task {task.name!r} requires one split protocol string.")
        return load_flip_dataset(root, name=cast(Any, task.name), split=str(split), download=download)

    if task.name not in PEER_NATIVE_DATASETS:
        raise EmbeddingInputError(
            f"Native PEER import is not available for task {task.name!r} with level {task.level!r}. "
            "Current native importers support protein-level and residue-level sequence LMDB datasets."
        )

    spec = PEER_NATIVE_DATASETS[task.name]
    if download:
        download_peer_dataset(root, name=task.name)
    split_names = _resolve_peer_splits(spec, split)
    lmdb_paths = _peer_lmdb_paths(root, spec, split_names)
    rows_by_split = [(split_name, _read_lmdb_records(path, spec)) for split_name, path in zip(split_names, lmdb_paths)]
    if task.level == "protein":
        return _protein_rows_to_dataset(task, rows_by_split)
    if task.level == "residue":
        return _residue_rows_to_dataset(task, rows_by_split)
    raise EmbeddingInputError(f"Native PEER importer does not support task level {task.level!r}.")


def download_peer_dataset(root: str | Path, *, name: str, force: bool = False) -> Path:
    """Download and extract a native PEER LMDB archive.

    Returns the directory containing the extracted dataset subdirectory.
    """

    task = get_peer_task(name)
    spec = PEER_NATIVE_DATASETS.get(task.name)
    if spec is None:
        raise EmbeddingInputError(f"PEER task {task.name!r} does not have a native downloadable LMDB spec.")

    dataset_root = Path(root).expanduser() / task.name
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / _archive_filename(spec.url)
    if force or not archive_path.exists():
        urlretrieve(spec.url, archive_path)
    _verify_md5(archive_path, spec.md5)
    extract_root = dataset_root / "raw"
    if force or not (extract_root / spec.archive_subdir).exists():
        _extract_archive(archive_path, extract_root)
    return extract_root


def list_peer_native_datasets() -> List[PeerNativeDatasetSpec]:
    """Return native PEER LMDB dataset specs supported without TorchDrug."""

    return list(PEER_NATIVE_DATASETS.values())


@dataclass(frozen=True)
class ResidueDatasetMetadata:
    """Metadata for residue-level datasets worth adapting later."""

    name: str
    category: str
    level: PeerTaskLevel
    objective: ObjectiveName
    target: str
    source: str
    url: str
    notes: str


RESIDUE_DATASET_CATALOG: Dict[str, ResidueDatasetMetadata] = {
    "dbptm": ResidueDatasetMetadata(
        name="dbptm",
        category="ptm",
        level="residue",
        objective="binary",
        target="ptm_site",
        source="dbPTM",
        url="https://biomics.lab.nycu.edu.tw/dbPTM/",
        notes="Experimentally verified and curated PTM sites across many PTM types; useful for per-PTM residue binary tasks.",
    ),
    "musitedeep": ResidueDatasetMetadata(
        name="musitedeep",
        category="ptm",
        level="residue",
        objective="binary",
        target="ptm_site",
        source="MusiteDeep",
        url="https://www.musite.net/",
        notes="PTM site prediction resource covering multiple PTM types; useful as a benchmark/reference source.",
    ),
    "disprot": ResidueDatasetMetadata(
        name="disprot",
        category="disorder",
        level="residue",
        objective="binary",
        target="disorder",
        source="DisProt",
        url="https://disprot.org/download",
        notes="Manually curated intrinsically disordered regions; intervals can be expanded to per-residue labels.",
    ),
    "biolip": ResidueDatasetMetadata(
        name="biolip",
        category="binding",
        level="residue",
        objective="binary",
        target="ligand_binding_site",
        source="BioLiP",
        url="https://zhanggroup.org/BioLiP/",
        notes="Structure-derived biologically relevant ligand-protein interactions; binding residues can become residue labels.",
    ),
    "metalpdb": ResidueDatasetMetadata(
        name="metalpdb",
        category="binding",
        level="residue",
        objective="binary",
        target="metal_binding_site",
        source="MetalPDB",
        url="https://metalpdb.cerm.unifi.it/",
        notes="Metal-binding sites in biological macromolecular structures; useful for residue metal-binding labels.",
    ),
    "scannet_binding": ResidueDatasetMetadata(
        name="scannet_binding",
        category="binding",
        level="residue",
        objective="binary",
        target="binding_site",
        source="ScanNet",
        url="https://github.com/jertubiana/ScanNet",
        notes="Published residue-level protein binding site prediction benchmark/code; useful reference for binding-site splits.",
    ),
    "netsurfp": ResidueDatasetMetadata(
        name="netsurfp",
        category="structure",
        level="residue",
        objective="multiclass",
        target="secondary_structure",
        source="NetSurfP",
        url="https://services.healthtech.dtu.dk/services/NetSurfP-3.0/",
        notes="Secondary structure / solvent accessibility style residue labels; PEER uses NetSurfP-2.0 plus CB513 for SSP.",
    ),
}


def list_residue_dataset_catalog(*, category: str | None = None) -> List[ResidueDatasetMetadata]:
    """Return curated residue-level dataset metadata."""

    values = list(RESIDUE_DATASET_CATALOG.values())
    if category is not None:
        normalized = str(category).strip().lower()
        values = [item for item in values if item.category == normalized]
    return values


def _resolve_peer_splits(spec: PeerNativeDatasetSpec, split: str | Sequence[str] | None) -> Tuple[str, ...]:
    if split is None:
        return spec.benchmark_splits or spec.splits
    values = [split] if isinstance(split, str) else list(split)
    normalized = tuple(str(value).strip() for value in values if str(value).strip())
    unsupported = [value for value in normalized if value not in spec.splits]
    if unsupported:
        supported = ", ".join(spec.splits)
        raise EmbeddingInputError(f"Unsupported PEER split(s): {', '.join(unsupported)}. Supported values: {supported}.")
    return normalized


def _peer_lmdb_paths(root: str | Path, spec: PeerNativeDatasetSpec, splits: Sequence[str]) -> List[Path]:
    base = Path(root).expanduser() / spec.name / "raw" / spec.archive_subdir
    paths = [base / f"{spec.lmdb_prefix}_{split}.lmdb" for split in splits]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise EmbeddingInputError(
            "Missing PEER LMDB files. Run download_peer_dataset(...), pass download=True, "
            f"or provide extracted data. Missing: {', '.join(missing[:3])}."
        )
    return paths


def _read_lmdb_records(path: Path, spec: PeerNativeDatasetSpec) -> List[Dict[str, Any]]:
    try:
        lmdb = importlib.import_module("lmdb")
    except Exception as exc:
        raise EmbeddingDependencyError("Native PEER LMDB import requires the optional 'lmdb' package.") from exc

    records: List[Dict[str, Any]] = []
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    try:
        with env.begin(write=False) as txn:
            count_raw = txn.get(spec.number_field.encode())
            if count_raw is None:
                raise EmbeddingInputError(f"LMDB file {path} is missing count field {spec.number_field!r}.")
            count = int(pickle.loads(count_raw))
            for index in range(count):
                payload = txn.get(str(index).encode())
                if payload is None:
                    raise EmbeddingInputError(f"LMDB file {path} is missing row {index}.")
                item = pickle.loads(payload)
                if not isinstance(item, dict):
                    raise EmbeddingInputError(f"LMDB row {index} in {path} is not a mapping.")
                item_map = cast(Dict[str, Any], item)
                row: Dict[str, Any] = {"sequence": item_map[spec.sequence_field]}
                for field in spec.target_fields:
                    row[field] = item_map[field]
                records.append(row)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return records


def _protein_rows_to_dataset(
    task: PeerTaskMetadata,
    rows_by_split: Sequence[Tuple[str, Sequence[Dict[str, Any]]]],
) -> ProteinDataset:
    examples: List[ProteinExample] = []
    for split_name, rows in rows_by_split:
        split = _normalize_peer_split(split_name)
        for row_index, row in enumerate(rows):
            examples.append(
                ProteinExample(
                    id=f"{task.name}:{split_name}:{row_index}",
                    sequence=str(row["sequence"]),
                    labels={task.target: _to_python_value(row[task.target])},
                    split=split,
                    metadata={"source": "peer", "task": task.name, "original_split": split_name, "row_index": row_index},
                )
            )
    return ProteinDataset(examples)


def _residue_rows_to_dataset(
    task: PeerTaskMetadata,
    rows_by_split: Sequence[Tuple[str, Sequence[Dict[str, Any]]]],
) -> ResidueDataset:
    examples: List[ResidueExample] = []
    for split_name, rows in rows_by_split:
        split = _normalize_peer_split(split_name)
        for row_index, row in enumerate(rows):
            labels = [_to_python_value(value) for value in _as_sequence(row[task.target])]
            mask = None
            if "valid_mask" in row:
                mask = [bool(_to_python_value(value)) for value in _as_sequence(row["valid_mask"])]
            examples.append(
                ResidueExample(
                    id=f"{task.name}:{split_name}:{row_index}",
                    sequence=str(row["sequence"]),
                    labels={task.target: labels},
                    split=split,
                    mask=mask,
                    metadata={"source": "peer", "task": task.name, "original_split": split_name, "row_index": row_index},
                )
            )
    return ResidueDataset(examples)


def _normalize_peer_split(split_name: str) -> SplitName:
    normalized = str(split_name).strip().lower()
    if normalized == "train":
        return "train"
    if normalized in {"valid", "validation", "val"}:
        return "val"
    return "test"


def _to_python_value(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return value


def _as_sequence(value: object) -> Sequence[object]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise EmbeddingInputError("Residue-level target values must be sequences.")
    return cast(Sequence[object], value)


def _archive_filename(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    if not name:
        raise EmbeddingInputError(f"Could not determine archive filename from URL {url!r}.")
    return name


def _verify_md5(path: Path, expected: str) -> None:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise EmbeddingInputError(f"MD5 mismatch for {path}: expected {expected}, got {digest.hexdigest()}.")


def _extract_archive(path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            archive.extractall(output_dir, filter="data")
        return
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(output_dir)
        return
    raise EmbeddingInputError(f"Unsupported PEER archive format: {path}.")


__all__ = [
    "PEER_CITATION",
    "PEER_TASKS",
    "PEER_TASK_ALIASES",
    "PeerTaskLevel",
    "PeerTaskMetadata",
    "PEER_NATIVE_DATASETS",
    "PeerNativeDatasetSpec",
    "RESIDUE_DATASET_CATALOG",
    "ResidueDatasetMetadata",
    "download_peer_dataset",
    "get_peer_task",
    "list_peer_tasks",
    "list_peer_native_datasets",
    "list_residue_dataset_catalog",
    "load_peer_dataset",
]
