"""Download and load dbPTM residue-level benchmark datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import tarfile
import urllib.request

from CBBIO.embeddings import EmbeddingInputError

from .collection_types import CollectionMetadata, DatasetMetadata, residue_dataset_metadata
from .datasets import ResidueDataset, ResidueExample, SplitName


@dataclass(frozen=True)
class DbptmBenchmarkMetadata:
    """Describe one published dbPTM benchmark archive."""

    name: str
    display_name: str
    target: str
    archive_name: str
    protein_count: int
    positive_sites: int
    negative_sites: int
    url: str


DbptmBenchmarkSpec = DbptmBenchmarkMetadata

DBPTM_BENCHMARK_BASE_URL = "https://biomics.lab.nycu.edu.tw/dbPTM/download/benchmark"


def _benchmark(
    name: str,
    display_name: str,
    archive_name: str,
    protein_count: int,
    positive_sites: int,
    negative_sites: int,
) -> DbptmBenchmarkMetadata:
    return DbptmBenchmarkMetadata(
        name=name,
        display_name=display_name,
        target=name,
        archive_name=archive_name,
        protein_count=protein_count,
        positive_sites=positive_sites,
        negative_sites=negative_sites,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/{archive_name}",
    )


DBPTM_BENCHMARKS: dict[str, DbptmBenchmarkMetadata] = {
    item.name: item
    for item in (
        _benchmark(
            "phosphorylation_by_cdk",
            "Phosphorylation by CDK",
            "PhosphorylationByCDK.tgz",
            1020,
            1503,
            29823,
        ),
        _benchmark(
            "phosphorylation_by_mapk",
            "Phosphorylation by MAPK",
            "PhosphorylationByMAPK.tgz",
            857,
            1270,
            22436,
        ),
        _benchmark(
            "phosphorylation_by_pka",
            "Phosphorylation by PKA",
            "PhosphorylationByPKA.tgz",
            905,
            1209,
            29813,
        ),
        _benchmark(
            "phosphorylation_by_pkc",
            "Phosphorylation by PKC",
            "PhosphorylationByPKC.tgz",
            691,
            943,
            24207,
        ),
        _benchmark(
            "phosphorylation_by_ck2",
            "Phosphorylation by CK2",
            "PhosphorylationByCK2.tgz",
            511,
            819,
            15387,
        ),
        _benchmark("acetylation", "Acetylation", "Acetylation.tgz", 5646, 14407, 8704),
        _benchmark("methylation", "Methylation", "Methylation.tgz", 5438, 14686, 36501),
        _benchmark(
            "n_linked_glycosylation",
            "N-linked Glycosylation",
            "N-linkedGlycosylation.tgz",
            1969,
            2517,
            8330,
        ),
        _benchmark(
            "o_linked_glycosylation",
            "O-linked Glycosylation",
            "O-linkedGlycosylation.tgz",
            1298,
            4470,
            37969,
        ),
        _benchmark(
            "s_nitrosylation",
            "S-nitrosylation",
            "S-nitrosylation.tgz",
            1434,
            3592,
            5803,
        ),
        _benchmark("sumoylation", "Sumoylation", "Sumoylation.tgz", 1432, 5191, 16066),
        _benchmark(
            "ubiquitination",
            "Ubiquitination",
            "Ubiquitination.tgz",
            4453,
            9767,
            8579,
        ),
    )
}

DBPTM_COLLECTION_METADATA = CollectionMetadata(
    id="dbptm",
    display_name="dbPTM",
    description="Curated post-translational modification datasets and benchmarks.",
    homepage="https://biomics.lab.nycu.edu.tw/dbPTM/",
    tags=("ptm", "residue", "benchmark"),
)
DBPTM_DATASETS: tuple[DatasetMetadata, ...] = (
    residue_dataset_metadata(
        dataset_id="dbptm:all",
        name="dbptm",
        display_name="dbPTM",
        source="dbPTM",
        category="ptm",
        objective="binary",
        target="ptm_site",
        status="adapter",
        homepage="https://biomics.lab.nycu.edu.tw/dbPTM/",
        description=(
            "Source: dbPTM. Class: ptms. Split system: adapter-defined until exported "
            "site annotations are imported."
        ),
        import_adapter="load_interval_residue_tsv",
        tags=("residue", "ptm", "dbptm"),
    ),
    *tuple(
        residue_dataset_metadata(
            dataset_id=f"dbptm:{benchmark.name}",
            name=benchmark.name,
            display_name=benchmark.display_name,
            source="dbPTM",
            category="ptm",
            objective="binary",
            target=benchmark.target,
            status="ready",
            homepage="https://biomics.lab.nycu.edu.tw/dbPTM/download.php",
            description=(
                f"Source: dbPTM benchmark archive {benchmark.archive_name}. Class: ptms. "
                "Split system: archive-provided positive and negative windows."
            ),
            download_url=benchmark.url,
            download_adapter="download_dbptm_benchmark",
            import_adapter="load_dbptm_benchmark_archive",
            loader="load_dbptm_benchmark_dataset",
            tags=("ptm", "dbptm", "benchmark"),
            notes=f"{benchmark.protein_count} proteins.",
        )
        for benchmark in DBPTM_BENCHMARKS.values()
    ),
)


def get_dbptm_benchmark(name: str) -> DbptmBenchmarkMetadata:
    """Return one registered dbPTM benchmark by name or archive-style alias."""
    key = _normalize_benchmark_name(name)
    metadata = DBPTM_BENCHMARKS.get(key)
    if metadata is None:
        supported = ", ".join(sorted(DBPTM_BENCHMARKS))
        raise EmbeddingInputError(
            f"Unknown dbPTM benchmark {name!r}. Supported values: {supported}."
        )
    return metadata


def list_dbptm_benchmarks() -> list[DbptmBenchmarkMetadata]:
    """Return registered dbPTM benchmark metadata."""
    return list(DBPTM_BENCHMARKS.values())


def download_dbptm_benchmark(
    root: str | Path,
    *,
    name: str = "phosphorylation_by_cdk",
    force: bool = False,
) -> Path:
    """Download one dbPTM benchmark archive."""
    metadata = get_dbptm_benchmark(name)
    output_dir = Path(root).expanduser() / "dbptm" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / metadata.archive_name
    if force or not path.exists():
        urllib.request.urlretrieve(metadata.url, path)
    return path


def load_dbptm_benchmark_dataset(
    root: str | Path,
    *,
    name: str = "phosphorylation_by_cdk",
    target: str | None = None,
    split: SplitName = "train",
    download: bool = False,
) -> ResidueDataset:
    """Load one dbPTM benchmark from its standard local path."""
    metadata = get_dbptm_benchmark(name)
    archive_path = (
        download_dbptm_benchmark(root, name=metadata.name)
        if download
        else Path(root).expanduser() / "dbptm" / "benchmark" / metadata.archive_name
    )
    return load_dbptm_benchmark_archive(
        archive_path,
        target=target or metadata.target,
        split=split,
    )


def load_dbptm_benchmark_archive(
    archive_path: str | Path,
    *,
    target: str = "ptm_site",
    split: SplitName = "train",
) -> ResidueDataset:
    """Load positive and negative FASTA windows from a dbPTM archive."""
    path = Path(archive_path).expanduser()
    if not path.exists():
        raise EmbeddingInputError(f"dbPTM benchmark archive does not exist: {path}.")
    examples: list[ResidueExample] = []
    seen_ids: dict[str, int] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith(
                (".fa", ".fasta", ".faa")
            ):
                continue
            label = _member_label(member.name)
            fasta_handle = archive.extractfile(member)
            if fasta_handle is None:
                continue
            text = fasta_handle.read().decode("utf-8")
            for record_id, sequence in _iter_fasta_text(text):
                labels = [0] * len(sequence)
                if label == 1:
                    labels[_center_index(sequence)] = 1
                base_id = f"{Path(member.name).stem}:{record_id}"
                examples.append(
                    ResidueExample(
                        id=_unique_id(base_id, seen_ids),
                        sequence=sequence,
                        labels={target: labels},
                        split=split,
                        metadata={
                            "source": "dbptm_benchmark",
                            "file": member.name,
                            "site_label": label,
                        },
                    )
                )
    if not examples:
        raise EmbeddingInputError(
            f"No dbPTM benchmark FASTA records found in archive {path}."
        )
    return ResidueDataset(examples)


def _unique_id(base_id: str, seen_ids: dict[str, int]) -> str:
    count = seen_ids.get(base_id, 0)
    seen_ids[base_id] = count + 1
    return base_id if count == 0 else f"{base_id}__dup{count + 1}"


def _normalize_benchmark_name(name: str) -> str:
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text.startswith("phosphorylation_by_"):
        return text
    if text.startswith("phosphorylationby"):
        return f"phosphorylation_by_{text.removeprefix('phosphorylationby')}"
    return text


def _member_label(member_name: str) -> int:
    stem = Path(member_name).stem.lower()
    if stem.endswith("_pos") or "positive" in stem:
        return 1
    if stem.endswith("_neg") or "negative" in stem:
        return 0
    raise EmbeddingInputError(
        f"Could not infer positive/negative label from dbPTM FASTA member {member_name!r}."
    )


def _center_index(sequence: str) -> int:
    if not sequence:
        raise EmbeddingInputError("dbPTM benchmark sequence windows cannot be empty.")
    return len(sequence) // 2


def _iter_fasta_text(text: str) -> Iterable[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id: str | None = None
    chunks: list[str] = []
    for raw_line in text.splitlines():
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


__all__ = [
    "DBPTM_BENCHMARKS",
    "DBPTM_BENCHMARK_BASE_URL",
    "DBPTM_COLLECTION_METADATA",
    "DBPTM_DATASETS",
    "DbptmBenchmarkMetadata",
    "DbptmBenchmarkSpec",
    "download_dbptm_benchmark",
    "get_dbptm_benchmark",
    "list_dbptm_benchmarks",
    "load_dbptm_benchmark_archive",
    "load_dbptm_benchmark_dataset",
]
