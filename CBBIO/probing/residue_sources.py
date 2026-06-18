"""Download and import adapters for residue-level probing datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import re
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve

from CBBIO.embeddings import EmbeddingInputError

from .datasets import ObjectiveName, ResidueDataset, ResidueExample, SplitName


ResidueSourceName = Literal[
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
]

BioLipLigandClass = Literal["all", "dna", "rna", "pep", "other"]
PhosphoElmSourceFilter = Literal["all", "LTP", "HTP"]


@dataclass(frozen=True)
class ResidueSourceSpec:
    """Metadata for a residue-level source adapter."""

    name: ResidueSourceName
    category: str
    objective: ObjectiveName
    target: str
    source: str
    homepage: str
    import_adapter: str
    download_urls: Tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True)
class DbptmBenchmarkSpec:
    """Metadata for one dbPTM benchmark PTM dataset."""

    name: str
    display_name: str
    target: str
    archive_name: str
    protein_count: int
    positive_sites: int
    negative_sites: int
    url: str


DBPTM_BENCHMARK_BASE_URL = "https://biomics.lab.nycu.edu.tw/dbPTM/download/benchmark"
DISPROT_CURRENT_TSV_URL = (
    "https://disprot.org/api/v2/download?format=tsv&release=current&term_ontology=IDPO&term_ontology=GO"
)
DISPROT_CURRENT_JSON_URL = (
    "https://disprot.org/api/v2/download?format=json&release=current&term_ontology=IDPO&term_ontology=GO"
)
DISPROT_SPLIT_RATIOS: Tuple[Tuple[SplitName, float], ...] = (("train", 0.8), ("val", 0.1), ("test", 0.1))
BIOLIP_DOWNLOAD_URLS = (
    "https://zhanggroup.org/BioLiP/download/BioLiP_nr.txt.gz",
    "https://zhanggroup.org/BioLiP/data/protein_nr.fasta.gz",
)

DBPTM_BENCHMARKS: Dict[str, DbptmBenchmarkSpec] = {
    "phosphorylation_by_cdk": DbptmBenchmarkSpec(
        name="phosphorylation_by_cdk",
        display_name="Phosphorylation by CDK",
        target="phosphorylation_by_cdk",
        archive_name="PhosphorylationByCDK.tgz",
        protein_count=1020,
        positive_sites=1503,
        negative_sites=29823,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/PhosphorylationByCDK.tgz",
    ),
    "phosphorylation_by_mapk": DbptmBenchmarkSpec(
        name="phosphorylation_by_mapk",
        display_name="Phosphorylation by MAPK",
        target="phosphorylation_by_mapk",
        archive_name="PhosphorylationByMAPK.tgz",
        protein_count=857,
        positive_sites=1270,
        negative_sites=22436,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/PhosphorylationByMAPK.tgz",
    ),
    "phosphorylation_by_pka": DbptmBenchmarkSpec(
        name="phosphorylation_by_pka",
        display_name="Phosphorylation by PKA",
        target="phosphorylation_by_pka",
        archive_name="PhosphorylationByPKA.tgz",
        protein_count=905,
        positive_sites=1209,
        negative_sites=29813,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/PhosphorylationByPKA.tgz",
    ),
    "phosphorylation_by_pkc": DbptmBenchmarkSpec(
        name="phosphorylation_by_pkc",
        display_name="Phosphorylation by PKC",
        target="phosphorylation_by_pkc",
        archive_name="PhosphorylationByPKC.tgz",
        protein_count=691,
        positive_sites=943,
        negative_sites=24207,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/PhosphorylationByPKC.tgz",
    ),
    "phosphorylation_by_ck2": DbptmBenchmarkSpec(
        name="phosphorylation_by_ck2",
        display_name="Phosphorylation by CK2",
        target="phosphorylation_by_ck2",
        archive_name="PhosphorylationByCK2.tgz",
        protein_count=511,
        positive_sites=819,
        negative_sites=15387,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/PhosphorylationByCK2.tgz",
    ),
    "acetylation": DbptmBenchmarkSpec(
        name="acetylation",
        display_name="Acetylation",
        target="acetylation",
        archive_name="Acetylation.tgz",
        protein_count=5646,
        positive_sites=14407,
        negative_sites=8704,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/Acetylation.tgz",
    ),
    "methylation": DbptmBenchmarkSpec(
        name="methylation",
        display_name="Methylation",
        target="methylation",
        archive_name="Methylation.tgz",
        protein_count=5438,
        positive_sites=14686,
        negative_sites=36501,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/Methylation.tgz",
    ),
    "n_linked_glycosylation": DbptmBenchmarkSpec(
        name="n_linked_glycosylation",
        display_name="N-linked Glycosylation",
        target="n_linked_glycosylation",
        archive_name="N-linkedGlycosylation.tgz",
        protein_count=1969,
        positive_sites=2517,
        negative_sites=8330,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/N-linkedGlycosylation.tgz",
    ),
    "o_linked_glycosylation": DbptmBenchmarkSpec(
        name="o_linked_glycosylation",
        display_name="O-linked Glycosylation",
        target="o_linked_glycosylation",
        archive_name="O-linkedGlycosylation.tgz",
        protein_count=1298,
        positive_sites=4470,
        negative_sites=37969,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/O-linkedGlycosylation.tgz",
    ),
    "s_nitrosylation": DbptmBenchmarkSpec(
        name="s_nitrosylation",
        display_name="S-nitrosylation",
        target="s_nitrosylation",
        archive_name="S-nitrosylation.tgz",
        protein_count=1434,
        positive_sites=3592,
        negative_sites=5803,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/S-nitrosylation.tgz",
    ),
    "sumoylation": DbptmBenchmarkSpec(
        name="sumoylation",
        display_name="Sumoylation",
        target="sumoylation",
        archive_name="Sumoylation.tgz",
        protein_count=1432,
        positive_sites=5191,
        negative_sites=16066,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/Sumoylation.tgz",
    ),
    "ubiquitination": DbptmBenchmarkSpec(
        name="ubiquitination",
        display_name="Ubiquitination",
        target="ubiquitination",
        archive_name="Ubiquitination.tgz",
        protein_count=4453,
        positive_sites=9767,
        negative_sites=8579,
        url=f"{DBPTM_BENCHMARK_BASE_URL}/Ubiquitination.tgz",
    ),
}


RESIDUE_SOURCE_SPECS: Dict[str, ResidueSourceSpec] = {
    "dbptm": ResidueSourceSpec(
        name="dbptm",
        category="ptm",
        objective="binary",
        target="ptm_site",
        source="dbPTM",
        homepage="https://biomics.lab.nycu.edu.tw/dbPTM/",
        import_adapter="load_interval_residue_tsv",
        notes="Use exported/dbPTM tabular site annotations and map intervals/sites to labels.",
    ),
    "musitedeep": ResidueSourceSpec(
        name="musitedeep",
        category="ptm",
        objective="binary",
        target="ptm_site",
        source="MusiteDeep",
        homepage="https://www.musite.net/",
        import_adapter="load_musitedeep_fasta",
        notes="MusiteDeep FASTA marks annotated residues by appending '#' after the amino acid.",
    ),
    "disprot": ResidueSourceSpec(
        name="disprot",
        category="disorder",
        objective="binary",
        target="disorder",
        source="DisProt",
        homepage="https://disprot.org/download",
        import_adapter="load_disprot_tsv",
        notes="Download the current TSV release and import disorder intervals; whole-protein labels require external sequences.",
    ),
    "biolip": ResidueSourceSpec(
        name="biolip",
        category="binding",
        objective="binary",
        target="ligand_binding_site",
        source="BioLiP",
        homepage="https://zhanggroup.org/BioLiP/download.html",
        import_adapter="load_biolip_dataset",
        download_urls=BIOLIP_DOWNLOAD_URLS,
        notes="BioLiP annotation column 9 provides binding residues renumbered from 1; column 21 contains receptor sequence.",
    ),
    "biolip_all": ResidueSourceSpec(
        name="biolip_all",
        category="binding",
        objective="binary",
        target="ligand_binding_site",
        source="BioLiP",
        homepage="https://zhanggroup.org/BioLiP/download.html",
        import_adapter="load_biolip_dataset",
        download_urls=BIOLIP_DOWNLOAD_URLS,
        notes="All BioLiP ligand classes merged into one residue-level binding-site target.",
    ),
    "biolip_dna": ResidueSourceSpec(
        name="biolip_dna",
        category="binding",
        objective="binary",
        target="dna_binding_site",
        source="BioLiP",
        homepage="https://zhanggroup.org/BioLiP/download.html",
        import_adapter="load_biolip_dataset",
        download_urls=BIOLIP_DOWNLOAD_URLS,
        notes="BioLiP rows whose ligand type is DNA, converted to residue-level binding-site labels.",
    ),
    "biolip_rna": ResidueSourceSpec(
        name="biolip_rna",
        category="binding",
        objective="binary",
        target="rna_binding_site",
        source="BioLiP",
        homepage="https://zhanggroup.org/BioLiP/download.html",
        import_adapter="load_biolip_dataset",
        download_urls=BIOLIP_DOWNLOAD_URLS,
        notes="BioLiP rows whose ligand type is RNA, converted to residue-level binding-site labels.",
    ),
    "biolip_pep": ResidueSourceSpec(
        name="biolip_pep",
        category="binding",
        objective="binary",
        target="peptide_binding_site",
        source="BioLiP",
        homepage="https://zhanggroup.org/BioLiP/download.html",
        import_adapter="load_biolip_dataset",
        download_urls=BIOLIP_DOWNLOAD_URLS,
        notes="BioLiP rows whose ligand type is peptide, converted to residue-level binding-site labels.",
    ),
    "biolip_other": ResidueSourceSpec(
        name="biolip_other",
        category="binding",
        objective="binary",
        target="other_ligand_binding_site",
        source="BioLiP",
        homepage="https://zhanggroup.org/BioLiP/download.html",
        import_adapter="load_biolip_dataset",
        download_urls=BIOLIP_DOWNLOAD_URLS,
        notes="BioLiP ligand rows excluding DNA, RNA, and peptide classes.",
    ),
    "metalpdb": ResidueSourceSpec(
        name="metalpdb",
        category="binding",
        objective="binary",
        target="metal_binding_site",
        source="MetalPDB",
        homepage="https://metalpdb.cerm.unifi.it/",
        import_adapter="load_interval_residue_tsv",
        notes="Use exported metal-binding site residues/intervals in a tabular format.",
    ),
    "scannet_binding": ResidueSourceSpec(
        name="scannet_binding",
        category="binding",
        objective="binary",
        target="binding_site",
        source="ScanNet",
        homepage="https://github.com/jertubiana/ScanNet",
        import_adapter="load_residue_label_table",
        notes="Use ScanNet-style residue label tables after generating or exporting sequence-level labels.",
    ),
    "netsurfp": ResidueSourceSpec(
        name="netsurfp",
        category="structure",
        objective="multiclass",
        target="secondary_structure",
        source="NetSurfP",
        homepage="https://services.healthtech.dtu.dk/services/NetSurfP-3.0/",
        import_adapter="load_residue_label_table",
        notes="Use NetSurfP tabular predictions/annotations with one label per residue.",
    ),
    "phosphoelm_all": ResidueSourceSpec(
        name="phosphoelm_all",
        category="ptm",
        objective="binary",
        target="phosphorylation_site",
        source="Phospho.ELM",
        homepage="http://phospho.elm.eu.org/",
        import_adapter="load_phosphoelm_dataset",
        notes="Full-protein phosphorylation labels from Phospho.ELM HTP and LTP evidence; unannotated S/T/Y residues are treated as negatives.",
    ),
    "phosphoelm_ltp": ResidueSourceSpec(
        name="phosphoelm_ltp",
        category="ptm",
        objective="binary",
        target="phosphorylation_site",
        source="Phospho.ELM",
        homepage="http://phospho.elm.eu.org/",
        import_adapter="load_phosphoelm_dataset",
        notes="Full-protein phosphorylation labels from Phospho.ELM low-throughput evidence only.",
    ),
    "phosphoelm_htp": ResidueSourceSpec(
        name="phosphoelm_htp",
        category="ptm",
        objective="binary",
        target="phosphorylation_site",
        source="Phospho.ELM",
        homepage="http://phospho.elm.eu.org/",
        import_adapter="load_phosphoelm_dataset",
        notes="Full-protein phosphorylation labels from Phospho.ELM high-throughput evidence only.",
    ),
}

MUSITEDEEP_TESTDATA_API_URL = (
    "https://api.github.com/repos/duolinwang/MusiteDeep/contents/testdata?ref=master"
)


def get_residue_source(name: str) -> ResidueSourceSpec:
    key = _normalize_residue_source_name(name)
    spec = RESIDUE_SOURCE_SPECS.get(key)
    if spec is None:
        supported = ", ".join(sorted(RESIDUE_SOURCE_SPECS))
        raise EmbeddingInputError(f"Unknown residue source {name!r}. Supported values: {supported}.")
    return spec


def _normalize_residue_source_name(name: str) -> str:
    text = str(name).strip().lower().replace("-", "_")
    aliases = {
        "dbptm:all": "dbptm",
        "musitedeep:all": "musitedeep",
        "disprot:all": "disprot",
        "biolip:all": "biolip_all",
        "biolip:dna": "biolip_dna",
        "biolip:rna": "biolip_rna",
        "biolip:pep": "biolip_pep",
        "biolip:peptide": "biolip_pep",
        "biolip:other": "biolip_other",
        "metalpdb:all": "metalpdb",
        "scannet:binding": "scannet_binding",
        "netsurfp:secondary_structure": "netsurfp",
        "phosphoelm:all": "phosphoelm_all",
        "phosphoelm:ltp": "phosphoelm_ltp",
        "phosphoelm:htp": "phosphoelm_htp",
    }
    return aliases.get(text, text)


def list_residue_sources(*, category: str | None = None) -> List[ResidueSourceSpec]:
    values = list(RESIDUE_SOURCE_SPECS.values())
    if category is not None:
        normalized = str(category).strip().lower()
        values = [value for value in values if value.category == normalized]
    return values


def get_dbptm_benchmark(name: str) -> DbptmBenchmarkSpec:
    key = _normalize_dbptm_benchmark_name(name)
    spec = DBPTM_BENCHMARKS.get(key)
    if spec is None:
        supported = ", ".join(sorted(DBPTM_BENCHMARKS))
        raise EmbeddingInputError(f"Unknown dbPTM benchmark {name!r}. Supported values: {supported}.")
    return spec


def list_dbptm_benchmarks() -> List[DbptmBenchmarkSpec]:
    return list(DBPTM_BENCHMARKS.values())


def download_residue_source(root: str | Path, *, name: str, force: bool = False) -> List[Path]:
    """Download source files for sources with stable direct-download URLs."""

    spec = get_residue_source(name)
    if spec.name == "disprot":
        return [download_disprot_current_json(root, force=force)]
    if not spec.download_urls:
        raise EmbeddingInputError(
            f"Residue source {spec.name!r} does not expose stable direct-download URLs in this adapter. "
            f"Download from {spec.homepage} and use {spec.import_adapter}."
        )
    output_dir = Path(root).expanduser() / _residue_source_storage_name(spec.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for url in spec.download_urls:
        path = output_dir / _download_filename(url)
        if force or not path.exists():
            _download_url(url, path)
        paths.append(path)
    return paths


def download_disprot_current_tsv(root: str | Path, *, force: bool = False) -> Path:
    """Download the current DisProt TSV export with IDPO and GO terms."""

    output_dir = Path(root).expanduser() / "disprot"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "disprot_current.tsv"
    if force or not path.exists():
        urlretrieve(DISPROT_CURRENT_TSV_URL, path)
    return path


def download_disprot_current_json(root: str | Path, *, force: bool = False) -> Path:
    """Download the current DisProt JSON export with IDPO and GO terms."""

    output_dir = Path(root).expanduser() / "disprot"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "disprot_current.json"
    if force or not path.exists():
        urlretrieve(DISPROT_CURRENT_JSON_URL, path)
    return path


def download_dbptm_benchmark(
    root: str | Path,
    *,
    name: str = "phosphorylation_by_cdk",
    force: bool = False,
) -> Path:
    """Download one dbPTM benchmark ``.tgz`` archive."""

    spec = get_dbptm_benchmark(name)
    output_dir = Path(root).expanduser() / "dbptm" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / spec.archive_name
    if force or not path.exists():
        urlretrieve(spec.url, path)
    return path


def load_dbptm_benchmark_dataset(
    root: str | Path,
    *,
    name: str = "phosphorylation_by_cdk",
    target: str | None = None,
    split: SplitName = "train",
    download: bool = False,
) -> ResidueDataset:
    """Load a dbPTM benchmark archive from ``root/dbptm/benchmark``."""

    spec = get_dbptm_benchmark(name)
    archive_path = download_dbptm_benchmark(root, name=spec.name) if download else Path(root).expanduser() / "dbptm" / "benchmark" / spec.archive_name
    return load_dbptm_benchmark_archive(archive_path, target=target or spec.target, split=split)


def load_dbptm_benchmark_archive(
    archive_path: str | Path,
    *,
    target: str = "ptm_site",
    split: SplitName = "train",
) -> ResidueDataset:
    """Load dbPTM benchmark positive/negative FASTA windows from a ``.tgz`` archive."""

    path = Path(archive_path).expanduser()
    if not path.exists():
        raise EmbeddingInputError(f"dbPTM benchmark archive does not exist: {path}.")
    examples: List[ResidueExample] = []
    seen_ids: dict[str, int] = {}

    def _unique_dbptm_id(base_id: str) -> str:
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        if count == 0:
            return base_id
        return f"{base_id}__dup{count + 1}"

    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.lower().endswith((".fa", ".fasta", ".faa")):
                continue
            label = _dbptm_member_label(member.name)
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
                        id=_unique_dbptm_id(base_id),
                        sequence=sequence,
                        labels={target: labels},
                        split=split,
                        metadata={"source": "dbptm_benchmark", "file": member.name, "site_label": label},
                    )
                )
    if not examples:
        raise EmbeddingInputError(f"No dbPTM benchmark FASTA records found in archive {path}.")
    return ResidueDataset(examples)


def download_musitedeep_testdata(
    root: str | Path,
    *,
    file_names: Sequence[str] | None = None,
    force: bool = False,
) -> List[Path]:
    """Download FASTA files from ``duolinwang/MusiteDeep/testdata``."""

    output_dir = Path(root).expanduser() / "musitedeep" / "testdata"
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {str(name) for name in file_names} if file_names is not None else None
    records = _github_contents(MUSITEDEEP_TESTDATA_API_URL)
    paths: List[Path] = []
    for record in records:
        item = cast(Mapping[str, object], record)
        name = item.get("name")
        download_url = item.get("download_url")
        item_type = item.get("type")
        if not isinstance(name, str) or not name.endswith(".fasta"):
            continue
        if wanted is not None and name not in wanted:
            continue
        if item_type != "file" or not isinstance(download_url, str):
            continue
        path = output_dir / name
        if force or not path.exists():
            urlretrieve(download_url, path)
        paths.append(path)
    if wanted is not None:
        missing = sorted(wanted - {path.name for path in paths})
        if missing:
            raise EmbeddingInputError(f"MusiteDeep testdata files not found on GitHub: {', '.join(missing)}.")
    return paths


def load_musitedeep_testdata_dataset(
    root: str | Path,
    *,
    target: str = "phosphorylation",
    file_names: Sequence[str] | None = None,
    download: bool = False,
) -> ResidueDataset:
    """Load MusiteDeep GitHub testdata FASTA files as one residue dataset.

    Files whose names contain ``test`` are assigned to the test split; all
    other files are assigned to train.
    """

    data_dir = Path(root).expanduser() / "musitedeep" / "testdata"
    paths = download_musitedeep_testdata(root, file_names=file_names) if download else _local_musitedeep_fastas(data_dir, file_names=file_names)
    examples: List[ResidueExample] = []
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


def load_residue_source_dataset(
    root: str | Path,
    *,
    name: str,
    target: str | None = None,
    split: SplitName | None = None,
    download: bool = False,
) -> ResidueDataset:
    """Load a configured residue source when the adapter has enough native metadata."""

    spec = get_residue_source(name)
    resolved_target = target or spec.target
    base = Path(root).expanduser() / _residue_source_storage_name(spec.name)
    biolip_ligand_class = _biolip_ligand_class_for_source(spec.name)
    if biolip_ligand_class is not None:
        if download:
            download_residue_source(root, name=spec.name)
        annotation = base / "BioLiP_nr.txt.gz"
        fasta = base / "protein_nr.fasta.gz"
        return load_biolip_dataset(
            annotation,
            protein_fasta=fasta if fasta.exists() else None,
            target=resolved_target,
            split=split,
            ligand_class=biolip_ligand_class,
        )
    if spec.name == "disprot":
        json_path = download_disprot_current_json(root) if download else base / "disprot_current.json"
        if json_path.exists():
            return load_disprot_json(json_path, target=resolved_target, split=split)
        tsv_path = base / "disprot_current.tsv"
        return load_disprot_tsv(tsv_path, target=resolved_target, split=split or "train", region_mode=True)
    phosphoelm_filter = _phosphoelm_source_filter_for_source(spec.name)
    if phosphoelm_filter is not None:
        return load_phosphoelm_dataset(
            _local_phosphoelm_dump(base),
            target=resolved_target,
            split=split,
            source_filter=phosphoelm_filter,
        )
    raise EmbeddingInputError(
        f"Residue source {spec.name!r} requires explicit input files. Use adapter {spec.import_adapter}."
    )


def load_phosphoelm_dataset(
    path: str | Path,
    *,
    target: str = "phosphorylation_site",
    split: SplitName | None = None,
    source_filter: PhosphoElmSourceFilter = "all",
    residue_codes: Sequence[str] = ("S", "T", "Y"),
) -> ResidueDataset:
    """Load a Phospho.ELM dump as full-protein phosphorylation labels.

    Positive labels are experimentally annotated sites in the dump. Other S/T/Y
    residues are left as 0 and included in the mask; non-S/T/Y residues are
    masked out because they are not phosphorylation-site candidates for NetPhos.
    """

    resolved_filter = _normalize_phosphoelm_source_filter(source_filter)
    candidate_codes = _normalize_phosphoelm_residue_codes(residue_codes)
    grouped: Dict[str, Dict[str, Any]] = {}
    evidence_counts: Dict[str, Dict[str, int]] = {}
    for row_index, row in enumerate(_iter_phosphoelm_rows(path)):
        row_source = str(row.get("source", "")).strip().upper()
        if resolved_filter != "all" and row_source != resolved_filter:
            continue
        acc = str(row.get("acc", "")).strip()
        sequence = str(row.get("sequence", "")).strip()
        code = str(row.get("code", "")).strip().upper()
        if code not in candidate_codes:
            continue
        if not acc or not sequence:
            raise EmbeddingInputError(f"Phospho.ELM row {row_index} requires non-empty acc and sequence fields.")
        try:
            position = int(str(row.get("position", "")).strip())
        except ValueError as exc:
            raise EmbeddingInputError(f"Phospho.ELM row {row_index} has invalid position {row.get('position')!r}.") from exc
        if position < 1 or position > len(sequence):
            raise EmbeddingInputError(
                f"Phospho.ELM row {row_index} position {position} is outside sequence length {len(sequence)}."
            )
        observed = sequence[position - 1].upper()
        if code != observed:
            raise EmbeddingInputError(
                f"Phospho.ELM row {row_index} code {code!r} does not match sequence residue {observed!r}."
            )
        entry = grouped.setdefault(
            acc,
            {
                "sequence": sequence,
                "labels": [0] * len(sequence),
                "mask": [aa.upper() in candidate_codes for aa in sequence],
                "metadata": {
                    "source": "phosphoelm",
                    "evidence_filter": resolved_filter,
                    "residue_codes": tuple(candidate_codes),
                    "species": row.get("species", ""),
                },
            },
        )
        if entry["sequence"] != sequence:
            raise EmbeddingInputError(f"Conflicting Phospho.ELM sequences for accession {acc!r}.")
        cast(List[int], entry["labels"])[position - 1] = 1
        counts = evidence_counts.setdefault(acc, {"HTP": 0, "LTP": 0, "other": 0})
        counts[row_source if row_source in {"HTP", "LTP"} else "other"] += 1

    if not grouped:
        raise EmbeddingInputError(f"No Phospho.ELM rows matched source_filter={resolved_filter!r}.")

    examples: List[ResidueExample] = []
    for acc, entry in grouped.items():
        labels = cast(List[int], entry["labels"])
        mask = cast(List[bool], entry["mask"])
        metadata = dict(cast(Dict[str, Any], entry["metadata"]))
        metadata["positive_site_count"] = sum(labels)
        metadata["positive_residue_counts"] = _phosphoelm_positive_residue_counts(str(entry["sequence"]), labels)
        metadata["positive_residue_stratum"] = _phosphoelm_positive_residue_stratum(
            cast(Mapping[str, int], metadata["positive_residue_counts"])
        )
        metadata["candidate_site_count"] = sum(mask)
        metadata["evidence_row_counts"] = evidence_counts[acc]
        examples.append(
            ResidueExample(
                id=acc,
                sequence=str(entry["sequence"]),
                labels={target: labels},
                split=split or "train",
                mask=mask,
                metadata=metadata,
            )
        )

    if split is None:
        examples = _split_phosphoelm_examples(examples)
    return ResidueDataset(examples)


def load_musitedeep_fasta(
    path: str | Path,
    *,
    target: str = "ptm_site",
    split: SplitName = "train",
) -> ResidueDataset:
    """Load MusiteDeep annotated FASTA where ``#`` marks the preceding residue."""

    examples: List[ResidueExample] = []
    for record_id, raw_sequence in _iter_fasta(path):
        sequence, labels = _parse_hash_marked_sequence(raw_sequence)
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


def load_interval_residue_tsv(
    path: str | Path,
    *,
    target: str = "target",
    id_field: str = "id",
    sequence_field: str = "sequence",
    start_field: str = "start",
    end_field: str = "end",
    split_field: str | None = "split",
    one_based: bool = True,
    delimiter: str = "\t",
    default_split: SplitName = "train",
) -> ResidueDataset:
    """Load residue labels from interval/site rows.

    Multiple rows with the same id are merged. Coordinates are inclusive.
    """

    grouped: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise EmbeddingInputError(f"Interval file {path} does not contain a header row.")
        for row in reader:
            record_id = str(row.get(id_field, "")).strip()
            sequence = str(row.get(sequence_field, "")).strip()
            if not record_id or not sequence:
                raise EmbeddingInputError("Interval rows require non-empty id and sequence fields.")
            entry = grouped.setdefault(
                record_id,
                {
                    "sequence": sequence,
                    "labels": [0] * len(sequence),
                    "split": _row_split(row, split_field=split_field, default_split=default_split),
                },
            )
            if entry["sequence"] != sequence:
                raise EmbeddingInputError(f"Conflicting sequences for interval id {record_id!r}.")
            start = int(str(row.get(start_field, "")).strip())
            end = int(str(row.get(end_field, start)).strip())
            _mark_interval(cast(List[int], entry["labels"]), start=start, end=end, one_based=one_based)

    return ResidueDataset(
        ResidueExample(
            id=record_id,
            sequence=str(entry["sequence"]),
            labels={target: cast(List[int], entry["labels"])},
            split=cast(SplitName, entry["split"]),
            metadata={"source": "interval_tsv"},
        )
        for record_id, entry in grouped.items()
    )


def load_residue_label_table(
    path: str | Path,
    *,
    target: str = "target",
    id_field: str = "id",
    sequence_field: str = "sequence",
    labels_field: str = "labels",
    split_field: str | None = "split",
    delimiter: str = "\t",
    default_split: SplitName = "train",
) -> ResidueDataset:
    """Load rows with one compact per-residue label string/list per sequence."""

    examples: List[ResidueExample] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise EmbeddingInputError(f"Residue label table {path} does not contain a header row.")
        for row_index, row in enumerate(reader):
            record_id = str(row.get(id_field) or row_index)
            sequence = str(row.get(sequence_field, "")).strip()
            labels = _parse_label_sequence(row.get(labels_field, ""), expected_length=len(sequence))
            examples.append(
                ResidueExample(
                    id=record_id,
                    sequence=sequence,
                    labels={target: labels},
                    split=_row_split(row, split_field=split_field, default_split=default_split),
                    metadata={"source": "residue_label_table", "row_index": row_index},
                )
            )
    return ResidueDataset(examples)


def load_disprot_json(
    path: str | Path,
    *,
    target: str = "disorder",
    split: SplitName | None = None,
) -> ResidueDataset:
    """Load DisProt-style JSON entries by expanding regions to residue labels."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = _disprot_entries(payload)
    if _has_current_disprot_region_schema(entries):
        return _load_current_disprot_json_entries(entries, target=target, split=split)

    examples: List[ResidueExample] = []
    for index, entry in enumerate(entries):
        entry_map = cast(Mapping[str, Any], entry)
        sequence = _entry_sequence(entry_map)
        labels = [0] * len(sequence)
        for region in _entry_regions(entry_map):
            start, end = _region_bounds(cast(Mapping[str, Any], region))
            _mark_interval(labels, start=start, end=end, one_based=True)
        record_id = str(entry_map.get("acc") or entry_map.get("disprot_id") or entry_map.get("id") or index)
        examples.append(
            ResidueExample(
                id=record_id,
                sequence=sequence,
                labels={target: labels},
                split=split or "train",
                metadata={"source": "disprot"},
            )
        )
    return ResidueDataset(examples)


def load_disprot_tsv(
    path: str | Path,
    *,
    target: str = "disorder",
    split: SplitName = "train",
    protein_sequences: Mapping[str, str] | None = None,
    region_mode: bool = False,
    term_ids: Sequence[str] = ("IDPO:0000002",),
    term_names: Sequence[str] = ("disorder",),
) -> ResidueDataset:
    """Load the DisProt TSV export.

    The current TSV export contains region sequences and coordinates but not
    full protein sequences. Pass ``protein_sequences`` to expand intervals
    over full proteins, or set ``region_mode=True`` to create one positive
    region-window example per matching TSV row.
    """

    rows = _matching_disprot_tsv_rows(path, term_ids=term_ids, term_names=term_names)
    if region_mode:
        examples: List[ResidueExample] = []
        for row_index, row in enumerate(rows):
            sequence = str(row.get("Region sequence", "")).strip()
            if not sequence:
                raise EmbeddingInputError("DisProt TSV region-mode rows require 'Region sequence'.")
            record_id = str(row.get("Region ID") or row.get("DisProt ID") or row_index)
            examples.append(
                ResidueExample(
                    id=record_id,
                    sequence=sequence,
                    labels={target: [1] * len(sequence)},
                    split=split,
                    metadata={
                        "source": "disprot",
                        "uniprot_acc": row.get("UniProt ACC", ""),
                        "disprot_id": row.get("DisProt ID", ""),
                        "start": row.get("Start", ""),
                        "end": row.get("End", ""),
                        "term_id": row.get("Term ID", ""),
                        "term_name": row.get("Term name", ""),
                    },
                )
            )
        return ResidueDataset(examples)

    if protein_sequences is None:
        raise EmbeddingInputError(
            "DisProt TSV does not include full protein sequences. Pass protein_sequences or set region_mode=True."
        )

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        acc = str(row.get("UniProt ACC", "")).strip()
        disprot_id = str(row.get("DisProt ID", "")).strip()
        record_id = acc or disprot_id
        sequence = protein_sequences.get(acc) or protein_sequences.get(disprot_id)
        if not record_id or sequence is None:
            raise EmbeddingInputError(
                f"Missing protein sequence for DisProt TSV row with UniProt ACC {acc!r} / DisProt ID {disprot_id!r}."
            )
        entry = grouped.setdefault(
            record_id,
            {
                "sequence": sequence,
                "labels": [0] * len(sequence),
                "metadata": {"source": "disprot", "uniprot_acc": acc, "disprot_id": disprot_id},
            },
        )
        if entry["sequence"] != sequence:
            raise EmbeddingInputError(f"Conflicting sequences for DisProt record {record_id!r}.")
        start = int(str(row.get("Start", "")).strip())
        end = int(str(row.get("End", start)).strip())
        _mark_interval(cast(List[int], entry["labels"]), start=start, end=end, one_based=True)

    return ResidueDataset(
        ResidueExample(
            id=record_id,
            sequence=str(entry["sequence"]),
            labels={target: cast(List[int], entry["labels"])},
            split=split,
            metadata=cast(Dict[str, Any], entry["metadata"]),
        )
        for record_id, entry in grouped.items()
    )


def load_biolip_dataset(
    annotation_path: str | Path,
    *,
    protein_fasta: str | Path | None = None,
    target: str | None = None,
    split: SplitName | None = "train",
    ligand_class: BioLipLigandClass = "all",
) -> ResidueDataset:
    """Load BioLiP annotation rows into residue-level ligand-binding labels."""

    resolved_ligand_class = _normalize_biolip_ligand_class(ligand_class)
    resolved_target = target or _biolip_target_for_ligand_class(resolved_ligand_class)
    fasta_sequences = dict(_iter_fasta(protein_fasta)) if protein_fasta is not None and Path(protein_fasta).exists() else {}
    grouped: Dict[str, Dict[str, Any]] = {}
    opener = gzip.open if str(annotation_path).endswith(".gz") else open
    with opener(annotation_path, "rt", encoding="utf-8") as handle:
        for line_index, raw_line in enumerate(handle):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t") if "\t" in line else line.split()
            if len(columns) < 21:
                raise EmbeddingInputError(f"BioLiP row {line_index} has {len(columns)} columns; expected at least 21.")
            if not _biolip_ligand_matches(columns[4], resolved_ligand_class):
                continue
            record_id = f"{columns[0]}{columns[1]}"
            sequence = fasta_sequences.get(record_id, columns[20])
            entry = grouped.setdefault(record_id, {"sequence": sequence, "labels": [0] * len(sequence)})
            if entry["sequence"] != sequence:
                raise EmbeddingInputError(f"Conflicting BioLiP sequences for receptor {record_id!r}.")
            for position in _biolip_positions(columns[8]):
                _mark_interval(cast(List[int], entry["labels"]), start=position, end=position, one_based=True)
    examples = [
        ResidueExample(
            id=record_id,
            sequence=str(entry["sequence"]),
            labels={resolved_target: cast(List[int], entry["labels"])},
            split=split or "train",
            metadata={"source": "biolip", "ligand_class": resolved_ligand_class},
        )
        for record_id, entry in grouped.items()
    ]
    if split is None:
        examples = _split_biolip_examples(examples)
    return ResidueDataset(examples)


def _iter_fasta(path: str | Path | None) -> Iterable[Tuple[str, str]]:
    if path is None:
        return []
    opener = gzip.open if str(path).endswith(".gz") else open
    records: List[Tuple[str, str]] = []
    current_id: str | None = None
    chunks: List[str] = []
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


def _iter_fasta_text(text: str) -> Iterable[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    current_id: str | None = None
    chunks: List[str] = []
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


def _parse_hash_marked_sequence(raw_sequence: str) -> Tuple[str, List[int]]:
    residues: List[str] = []
    labels: List[int] = []
    for char in raw_sequence:
        if char == "#":
            if not labels:
                raise EmbeddingInputError("MusiteDeep marker '#' appears before any residue.")
            labels[-1] = 1
            continue
        residues.append(char)
        labels.append(0)
    return "".join(residues), labels


def _row_split(row: Mapping[str, str], *, split_field: str | None, default_split: SplitName) -> SplitName:
    if split_field is None:
        return default_split
    value = str(row.get(split_field, "")).strip().lower()
    if not value:
        return default_split
    if value in {"train", "training"}:
        return "train"
    if value in {"val", "valid", "validation"}:
        return "val"
    if value == "test":
        return "test"
    raise EmbeddingInputError("Split must be one of: train, val, test.")


def _mark_interval(labels: List[int], *, start: int, end: int, one_based: bool) -> None:
    start_index = start - 1 if one_based else start
    end_index = end - 1 if one_based else end
    if start_index < 0 or end_index >= len(labels) or end_index < start_index:
        raise EmbeddingInputError(
            f"Invalid residue interval start={start} end={end} for sequence length {len(labels)}."
        )
    for index in range(start_index, end_index + 1):
        labels[index] = 1


def _parse_label_sequence(value: str | None, *, expected_length: int) -> List[Any]:
    text = "" if value is None else str(value).strip()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise EmbeddingInputError("Label JSON must be an array.")
        labels: List[Any] = list(cast(List[object], parsed))
    elif "," in text:
        labels = [_coerce_label(part.strip()) for part in text.split(",")]
    elif " " in text:
        labels = [_coerce_label(part.strip()) for part in text.split() if part.strip()]
    else:
        labels = [_coerce_label(char) for char in text]
    if len(labels) != expected_length:
        raise EmbeddingInputError(f"Label length {len(labels)} does not match sequence length {expected_length}.")
    return labels


def _coerce_label(value: object) -> Any:
    if isinstance(value, (int, float, bool)):
        return int(value) if isinstance(value, bool) else value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _disprot_entries(payload: object) -> Sequence[object]:
    if isinstance(payload, list):
        return cast(List[object], payload)
    if isinstance(payload, dict):
        payload_map = cast(Mapping[str, object], payload)
        for key in ("data", "entries", "results"):
            value = payload_map.get(key)
            if isinstance(value, list):
                return cast(List[object], value)
    raise EmbeddingInputError("DisProt JSON must be a list or contain data/entries/results list.")


def _has_current_disprot_region_schema(entries: Sequence[object]) -> bool:
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        raw_regions = cast(Mapping[str, object], entry).get("regions")
        if isinstance(raw_regions, Mapping) and (
            "term_namespace" in raw_regions or "term_name" in raw_regions or "term_id" in raw_regions
        ):
            return True
    return False


def _load_current_disprot_json_entries(
    entries: Sequence[object],
    *,
    target: str,
    split: SplitName | None,
) -> ResidueDataset:
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_map = cast(Mapping[str, Any], entry)
        raw_region = entry_map.get("regions")
        if not isinstance(raw_region, Mapping):
            continue
        region = cast(Mapping[str, Any], raw_region)
        if not _is_disprot_disorder_region(region):
            continue

        sequence = _entry_sequence(entry_map)
        record_id = str(entry_map.get("acc") or entry_map.get("disprot_id") or entry_map.get("id") or "").strip()
        if not record_id:
            raise EmbeddingInputError("Current DisProt JSON disorder rows require acc, disprot_id, or id.")
        disprot_id = str(entry_map.get("disprot_id") or "").strip()
        dataset_tags = tuple(str(tag) for tag in entry_map.get("dataset", ()) if str(tag).strip())
        group = grouped.setdefault(
            record_id,
            {
                "sequence": sequence,
                "labels": [0] * len(sequence),
                "metadata": {
                    "source": "disprot",
                    "uniprot_acc": str(entry_map.get("acc") or ""),
                    "disprot_id": disprot_id,
                    "dataset_tags": dataset_tags,
                    "disorder_interval_count": 0,
                },
            },
        )
        if group["sequence"] != sequence:
            raise EmbeddingInputError(f"Conflicting sequences for DisProt record {record_id!r}.")
        metadata = cast(Dict[str, Any], group["metadata"])
        metadata["dataset_tags"] = tuple(sorted(set(cast(Tuple[str, ...], metadata["dataset_tags"]) + dataset_tags)))
        if disprot_id and not metadata.get("disprot_id"):
            metadata["disprot_id"] = disprot_id

        start, end = _region_bounds(region)
        _mark_interval(cast(List[int], group["labels"]), start=start, end=end, one_based=True)
        metadata["disorder_interval_count"] = int(metadata["disorder_interval_count"]) + 1

    if not grouped:
        raise EmbeddingInputError("No Structural state disorder rows found in DisProt JSON.")

    examples: List[ResidueExample] = []
    for record_id, group in grouped.items():
        labels = cast(List[int], group["labels"])
        sequence = str(group["sequence"])
        metadata = dict(cast(Dict[str, Any], group["metadata"]))
        metadata["disorder_fraction"] = sum(labels) / len(labels)
        metadata["disorder_content_bin"] = _disprot_disorder_content_bin(float(metadata["disorder_fraction"]))
        metadata["dataset_stratum"] = _disprot_dataset_stratum(cast(Sequence[str], metadata["dataset_tags"]))
        examples.append(
            ResidueExample(
                id=record_id,
                sequence=sequence,
                labels={target: labels},
                split=split or "train",
                metadata=metadata,
            )
        )

    if split is None:
        examples = _assign_disprot_stratified_splits(examples)
    return ResidueDataset(examples)


def _is_disprot_disorder_region(region: Mapping[str, Any]) -> bool:
    namespace = str(region.get("term_namespace") or "").strip().lower()
    term_name = str(region.get("term_name") or "").strip().lower()
    return namespace == "structural state" and term_name == "disorder"


def _assign_disprot_stratified_splits(examples: Sequence[ResidueExample]) -> List[ResidueExample]:
    primary_groups: Dict[Tuple[str, str], List[ResidueExample]] = {}
    for example in examples:
        metadata = example.metadata or {}
        key = (
            str(metadata.get("dataset_stratum") or "none"),
            str(metadata.get("disorder_content_bin") or "unknown"),
        )
        primary_groups.setdefault(key, []).append(example)

    assigned: List[ResidueExample] = []
    rare_by_content: Dict[str, List[ResidueExample]] = {}
    for key, group in primary_groups.items():
        if len(group) >= 10:
            assigned.extend(_split_disprot_group(group))
        else:
            rare_by_content.setdefault(key[1], []).extend(group)

    rare_global: List[ResidueExample] = []
    for group in rare_by_content.values():
        if len(group) >= 10:
            assigned.extend(_split_disprot_group(group))
        else:
            rare_global.extend(group)
    if rare_global:
        assigned.extend(_split_disprot_group(rare_global))

    return sorted(assigned, key=lambda example: example.id)


def _split_disprot_group(group: Sequence[ResidueExample]) -> List[ResidueExample]:
    ordered = sorted(group, key=lambda example: _stable_disprot_hash(example.id))
    counts = _disprot_split_counts(len(ordered))
    split_names: List[SplitName] = [split_name for split_name, count in counts for _ in range(count)]
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


def _disprot_split_counts(total: int) -> List[Tuple[SplitName, int]]:
    if total <= 0:
        return [("train", 0), ("val", 0), ("test", 0)]
    val = int(round(total * 0.1))
    test = int(round(total * 0.1))
    if total >= 10:
        val = max(1, val)
        test = max(1, test)
    elif total >= 2:
        test = max(1, test)
    train = total - val - test
    while train < 1 and val > 0:
        val -= 1
        train += 1
    while train < 1 and test > 0:
        test -= 1
        train += 1
    return [("train", train), ("val", val), ("test", test)]


def _stable_disprot_hash(value: str) -> str:
    return hashlib.sha256(f"disprot-split-v1:{value}".encode("utf-8")).hexdigest()


def _disprot_disorder_content_bin(fraction: float) -> str:
    if fraction < 0.1:
        return "0-10%"
    if fraction < 0.3:
        return "10-30%"
    if fraction < 0.6:
        return "30-60%"
    return "60-100%"


def _disprot_dataset_stratum(dataset_tags: Sequence[str]) -> str:
    values = sorted({str(tag).strip() for tag in dataset_tags if str(tag).strip()})
    return "+".join(values) if values else "none"


def _split_biolip_examples(examples: Sequence[ResidueExample]) -> List[ResidueExample]:
    ordered = sorted(examples, key=lambda example: _stable_biolip_hash(example.id))
    counts = _disprot_split_counts(len(ordered))
    split_names: List[SplitName] = [split_name for split_name, count in counts for _ in range(count)]
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


def _stable_biolip_hash(value: str) -> str:
    return hashlib.sha256(f"biolip-split-v1:{value}".encode("utf-8")).hexdigest()


def _residue_source_storage_name(name: str) -> str:
    normalized = str(name)
    if normalized.startswith("biolip"):
        return "biolip"
    if normalized.startswith("phosphoelm"):
        return "phosphoelm"
    return normalized


def _phosphoelm_source_filter_for_source(name: str) -> PhosphoElmSourceFilter | None:
    normalized = str(name).strip().lower()
    if normalized == "phosphoelm_all":
        return "all"
    if normalized == "phosphoelm_ltp":
        return "LTP"
    if normalized == "phosphoelm_htp":
        return "HTP"
    return None


def _normalize_phosphoelm_source_filter(value: str) -> PhosphoElmSourceFilter:
    normalized = str(value).strip().upper()
    if normalized in {"", "ALL"}:
        return "all"
    if normalized in {"LTP", "HTP"}:
        return cast(PhosphoElmSourceFilter, normalized)
    raise EmbeddingInputError("Phospho.ELM source_filter must be one of: all, LTP, HTP.")


def _normalize_phosphoelm_residue_codes(values: Sequence[str]) -> Tuple[str, ...]:
    codes = tuple(dict.fromkeys(str(value).strip().upper() for value in values if str(value).strip()))
    if not codes:
        raise EmbeddingInputError("Phospho.ELM residue_codes must contain at least one residue code.")
    invalid = [code for code in codes if code not in {"S", "T", "Y"}]
    if invalid:
        raise EmbeddingInputError("Phospho.ELM residue_codes must be drawn from: S, T, Y.")
    return codes


def _local_phosphoelm_dump(root: Path) -> Path:
    candidates: List[Path] = []
    for directory in (root, *root.parents):
        candidates.extend(
            [
                directory / "phosphoELM_all_latest.dump.tgz",
                directory / "phosphoELM_all_2015-04.dump",
            ]
        )
    for path in candidates:
        if path.exists():
            return path
    raise EmbeddingInputError(
        f"No Phospho.ELM dump found under {root}. Expected phosphoELM_all_latest.dump.tgz "
        "or phosphoELM_all_2015-04.dump."
    )


def _iter_phosphoelm_rows(path: str | Path) -> Iterable[Dict[str, str]]:
    source = Path(path).expanduser()
    if not source.exists():
        raise EmbeddingInputError(f"Phospho.ELM dump does not exist: {source}.")
    if source.suffixes[-2:] == [".dump", ".tgz"] or source.name.endswith(".tgz"):
        with tarfile.open(source, "r:gz") as archive:
            members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".dump")]
            if not members:
                raise EmbeddingInputError(f"No .dump member found in Phospho.ELM archive {source}.")
            handle = archive.extractfile(members[0])
            if handle is None:
                raise EmbeddingInputError(f"Could not read Phospho.ELM archive member {members[0].name}.")
            with handle:
                text = (line.decode("utf-8") for line in handle)
                yield from csv.DictReader(text, delimiter="\t")
        return
    with source.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _split_phosphoelm_examples(examples: Sequence[ResidueExample]) -> List[ResidueExample]:
    primary_groups: Dict[Tuple[str, str], List[ResidueExample]] = {}
    for example in examples:
        metadata = example.metadata or {}
        key = (
            str(metadata.get("species") or "unknown"),
            str(metadata.get("positive_residue_stratum") or "none"),
        )
        primary_groups.setdefault(key, []).append(example)

    assigned: List[ResidueExample] = []
    rare_by_species: Dict[str, List[ResidueExample]] = {}
    for key, group in primary_groups.items():
        if len(group) >= 10:
            assigned.extend(_split_phosphoelm_group(group))
        else:
            rare_by_species.setdefault(key[0], []).extend(group)

    rare_global: List[ResidueExample] = []
    for group in rare_by_species.values():
        if len(group) >= 10:
            assigned.extend(_split_phosphoelm_group(group))
        else:
            rare_global.extend(group)
    if rare_global:
        assigned.extend(_split_phosphoelm_group(rare_global))

    return sorted(assigned, key=lambda example: example.id)


def _split_phosphoelm_group(group: Sequence[ResidueExample]) -> List[ResidueExample]:
    ordered = sorted(group, key=lambda example: _stable_phosphoelm_hash(example.id))
    counts = _disprot_split_counts(len(ordered))
    split_names: List[SplitName] = [split_name for split_name, count in counts for _ in range(count)]
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


def _stable_phosphoelm_hash(value: str) -> str:
    return hashlib.sha256(f"phosphoelm-split-v1:{value}".encode("utf-8")).hexdigest()


def _phosphoelm_positive_residue_counts(sequence: str, labels: Sequence[int]) -> Dict[str, int]:
    counts = {"S": 0, "T": 0, "Y": 0}
    for residue, label in zip(sequence, labels):
        code = residue.upper()
        if label and code in counts:
            counts[code] += 1
    return counts


def _phosphoelm_positive_residue_stratum(counts: Mapping[str, int]) -> str:
    codes = [code for code in ("S", "T", "Y") if int(counts.get(code, 0)) > 0]
    return "+".join(codes) if codes else "none"


def _biolip_ligand_class_for_source(name: str) -> BioLipLigandClass | None:
    normalized = str(name).strip().lower()
    if normalized in {"biolip", "biolip_all"}:
        return "all"
    if normalized == "biolip_dna":
        return "dna"
    if normalized == "biolip_rna":
        return "rna"
    if normalized == "biolip_pep":
        return "pep"
    if normalized == "biolip_other":
        return "other"
    return None


def _normalize_biolip_ligand_class(value: str) -> BioLipLigandClass:
    normalized = str(value).strip().lower()
    if normalized in {"all", "dna", "rna", "other"}:
        return cast(BioLipLigandClass, normalized)
    if normalized in {"pep", "peptide"}:
        return "pep"
    raise EmbeddingInputError("BioLiP ligand_class must be one of: all, dna, rna, pep, other.")


def _biolip_target_for_ligand_class(ligand_class: BioLipLigandClass) -> str:
    if ligand_class == "dna":
        return "dna_binding_site"
    if ligand_class == "rna":
        return "rna_binding_site"
    if ligand_class == "pep":
        return "peptide_binding_site"
    if ligand_class == "other":
        return "other_ligand_binding_site"
    return "ligand_binding_site"


def _biolip_ligand_matches(value: str, ligand_class: BioLipLigandClass) -> bool:
    ligand = str(value).strip().lower()
    if ligand_class == "all":
        return True
    if ligand_class == "pep":
        return ligand == "peptide"
    if ligand_class == "other":
        return ligand not in {"dna", "rna", "peptide"}
    return ligand == ligand_class


def _entry_sequence(entry: Mapping[str, Any]) -> str:
    sequence = entry.get("sequence")
    if isinstance(sequence, str):
        return sequence
    if isinstance(sequence, Mapping):
        sequence_map = cast(Mapping[str, object], sequence)
        for key in ("sequence", "value"):
            value = sequence_map.get(key)
            if isinstance(value, str):
                return value
    raise EmbeddingInputError("DisProt entry is missing a sequence string.")


def _entry_regions(entry: Mapping[str, Any]) -> Sequence[object]:
    raw_regions = entry.get("regions")
    if raw_regions is None:
        raw_regions = entry.get("annotations")
    regions: object = raw_regions if raw_regions is not None else []
    if not isinstance(regions, list):
        raise EmbeddingInputError("DisProt entry regions must be a list.")
    return cast(List[object], regions)


def _matching_disprot_tsv_rows(
    path: str | Path,
    *,
    term_ids: Sequence[str],
    term_names: Sequence[str],
) -> List[Mapping[str, str]]:
    wanted_ids = {term_id.strip().lower() for term_id in term_ids}
    wanted_names = {term_name.strip().lower() for term_name in term_names}
    rows: List[Mapping[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise EmbeddingInputError(f"DisProt TSV {path} does not contain a header row.")
        required = {"UniProt ACC", "DisProt ID", "Start", "End", "Region sequence", "Term ID", "Term name"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise EmbeddingInputError(f"DisProt TSV is missing required columns: {', '.join(missing)}.")
        for row in reader:
            term_id = str(row.get("Term ID", "")).strip().lower()
            term_name = str(row.get("Term name", "")).strip().lower()
            if term_id in wanted_ids or term_name in wanted_names:
                rows.append(row)
    if not rows:
        raise EmbeddingInputError("No matching DisProt TSV rows found for the requested term filters.")
    return rows


def _region_bounds(region: Mapping[str, Any]) -> Tuple[int, int]:
    start = region.get("start") or region.get("start_position") or region.get("begin")
    end = region.get("end") or region.get("end_position") or region.get("stop") or start
    if start is None or end is None:
        raise EmbeddingInputError("DisProt region is missing start/end coordinates.")
    return int(start), int(end)


def _biolip_positions(value: str) -> List[int]:
    positions: List[int] = []
    for token in value.replace(";", " ").split():
        match = re.search(r"(-?\d+)$", token)
        if match is not None:
            positions.append(int(match.group(1)))
    return positions


def _download_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise EmbeddingInputError(f"Could not determine filename for URL {url!r}.")
    return name


def _download_url(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 CBBIO/0.1"})
    with urlopen(request) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _normalize_dbptm_benchmark_name(name: str) -> str:
    text = str(name).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if text.startswith("phosphorylation_by_"):
        return text
    if text.startswith("phosphorylationby"):
        return f"phosphorylation_by_{text.removeprefix('phosphorylationby')}"
    return text


def _dbptm_member_label(member_name: str) -> int:
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


def _github_contents(api_url: str) -> Sequence[object]:
    path, _headers = urlretrieve(api_url)
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    finally:
        Path(path).unlink(missing_ok=True)
    if not isinstance(payload, list):
        raise EmbeddingInputError("GitHub contents API did not return a file list.")
    return cast(List[object], payload)


def _local_musitedeep_fastas(root: Path, *, file_names: Sequence[str] | None) -> List[Path]:
    if file_names is not None:
        paths = [root / name for name in file_names]
    else:
        paths = sorted(root.glob("*.fasta"))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise EmbeddingInputError(
            "Missing MusiteDeep testdata FASTA files. Pass download=True or download them first. "
            f"Missing: {', '.join(missing[:5])}."
        )
    if not paths:
        raise EmbeddingInputError(f"No MusiteDeep FASTA files found under {root}.")
    return paths


__all__ = [
    "DBPTM_BENCHMARKS",
    "BioLipLigandClass",
    "DISPROT_CURRENT_JSON_URL",
    "DISPROT_CURRENT_TSV_URL",
    "DbptmBenchmarkSpec",
    "PhosphoElmSourceFilter",
    "RESIDUE_SOURCE_SPECS",
    "ResidueSourceName",
    "ResidueSourceSpec",
    "download_dbptm_benchmark",
    "download_disprot_current_json",
    "download_disprot_current_tsv",
    "download_residue_source",
    "download_musitedeep_testdata",
    "get_dbptm_benchmark",
    "get_residue_source",
    "list_dbptm_benchmarks",
    "list_residue_sources",
    "load_biolip_dataset",
    "load_dbptm_benchmark_archive",
    "load_dbptm_benchmark_dataset",
    "load_disprot_json",
    "load_disprot_tsv",
    "load_interval_residue_tsv",
    "load_musitedeep_fasta",
    "load_musitedeep_testdata_dataset",
    "load_phosphoelm_dataset",
    "load_residue_label_table",
    "load_residue_source_dataset",
]
