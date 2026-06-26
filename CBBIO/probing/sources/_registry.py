"""Register residue dataset sources and dispatch provider adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Literal
from urllib.parse import urlparse
import urllib.request

from CBBIO.embeddings import EmbeddingInputError

from .biolip import BIOLIP_DOWNLOAD_URLS, BioLipLigandClass, load_biolip_dataset
from ..collection_types import CollectionMetadata, DatasetMetadata, residue_dataset_metadata
from ..datasets import ObjectiveName, ResidueDataset, SplitName
from ..splitters import DatasetSplitter
from .dbptm import (
    DBPTM_BENCHMARKS,
    DbptmBenchmarkSpec,
    download_dbptm_benchmark,
    get_dbptm_benchmark,
    list_dbptm_benchmarks,
    load_dbptm_benchmark_archive,
    load_dbptm_benchmark_dataset,
)
from .disprot import (
    DISPROT_CURRENT_JSON_URL,
    DISPROT_CURRENT_TSV_URL,
    download_disprot_current_json,
    download_disprot_current_tsv,
    load_disprot_json,
    load_disprot_tsv,
)
from .musitedeep import (
    download_musitedeep_testdata,
    load_musitedeep_fasta,
    load_musitedeep_testdata_dataset,
)
from .phosphoelm import PhosphoElmSourceFilter, load_phosphoelm_dataset
from ._tables import load_interval_residue_tsv, load_residue_label_table


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
    download_urls: tuple[str, ...] = ()
    notes: str = ""



RESIDUE_SOURCE_SPECS: dict[str, ResidueSourceSpec] = {
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
        notes=(
            "Download the current TSV release and import disorder intervals; "
            "whole-protein labels require external sequences."
        ),
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
        notes=(
            "BioLiP annotation column 9 provides binding residues renumbered from 1; "
            "column 21 contains receptor sequence."
        ),
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
        notes=(
            "BioLiP rows whose ligand type is DNA, converted to residue-level "
            "binding-site labels."
        ),
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
        notes=(
            "BioLiP rows whose ligand type is RNA, converted to residue-level "
            "binding-site labels."
        ),
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
        notes=(
            "BioLiP rows whose ligand type is peptide, converted to residue-level "
            "binding-site labels."
        ),
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
        notes=(
            "Use ScanNet-style residue label tables after generating or exporting "
            "sequence-level labels."
        ),
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
        notes=(
            "Full-protein phosphorylation labels from Phospho.ELM HTP and LTP evidence; "
            "unannotated S/T/Y residues are treated as negatives."
        ),
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

ADAPTER_COLLECTION_METADATA: tuple[CollectionMetadata, ...] = (
    CollectionMetadata("metalpdb", "MetalPDB", "Metal-binding residue datasets."),
    CollectionMetadata("scannet", "ScanNet", "Protein binding-site datasets."),
    CollectionMetadata("netsurfp", "NetSurfP", "Protein structure annotation datasets."),
)
ADAPTER_DATASETS: tuple[DatasetMetadata, ...] = tuple(
    residue_dataset_metadata(
        dataset_id=dataset_id,
        name=name,
        display_name=RESIDUE_SOURCE_SPECS[name].source,
        source=RESIDUE_SOURCE_SPECS[name].source,
        category=RESIDUE_SOURCE_SPECS[name].category,
        objective=RESIDUE_SOURCE_SPECS[name].objective,
        target=RESIDUE_SOURCE_SPECS[name].target,
        status="adapter",
        homepage=RESIDUE_SOURCE_SPECS[name].homepage,
        description=(
            f"Source: {RESIDUE_SOURCE_SPECS[name].source}. "
            f"Class: {RESIDUE_SOURCE_SPECS[name].category}. "
            "Split system: adapter-defined until imported."
        ),
        import_adapter=RESIDUE_SOURCE_SPECS[name].import_adapter,
        tags=("residue", RESIDUE_SOURCE_SPECS[name].category, name),
        notes=RESIDUE_SOURCE_SPECS[name].notes,
    )
    for dataset_id, name in (
        ("metalpdb:all", "metalpdb"),
        ("scannet:binding", "scannet_binding"),
        ("netsurfp:secondary_structure", "netsurfp"),
    )
)

def get_residue_source(name: str) -> ResidueSourceSpec:
    """Return one registered residue source specification."""
    key = _normalize_residue_source_name(name)
    spec = RESIDUE_SOURCE_SPECS.get(key)
    if spec is None:
        supported = ", ".join(sorted(RESIDUE_SOURCE_SPECS))
        raise EmbeddingInputError(
            f"Unknown residue source {name!r}. Supported values: {supported}."
        )
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


def list_residue_sources(*, category: str | None = None) -> list[ResidueSourceSpec]:
    """Return registered residue source specifications."""
    values = list(RESIDUE_SOURCE_SPECS.values())
    if category is not None:
        normalized = str(category).strip().lower()
        values = [value for value in values if value.category == normalized]
    return values


def download_residue_source(
    root: str | Path,
    *,
    name: str,
    force: bool = False,
) -> list[Path]:
    """Download source files for sources with stable direct-download URLs."""

    spec = get_residue_source(name)
    if spec.name == "disprot":
        return [download_disprot_current_json(root, force=force)]
    if not spec.download_urls:
        raise EmbeddingInputError(
            f"Residue source {spec.name!r} does not expose stable direct-download URLs "
            "in this adapter. "
            f"Download from {spec.homepage} and use {spec.import_adapter}."
        )
    output_dir = Path(root).expanduser() / _residue_source_storage_name(spec.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for url in spec.download_urls:
        path = output_dir / _download_filename(url)
        if force or not path.exists():
            _download_url(url, path)
        paths.append(path)
    return paths


def load_residue_source_dataset(
    root: str | Path,
    *,
    name: str,
    target: str | None = None,
    split: SplitName | None = None,
    download: bool = False,
    splitter: DatasetSplitter | None = None,
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
            splitter=splitter,
        )
    if spec.name == "disprot":
        json_path = (
            download_disprot_current_json(root)
            if download
            else base / "disprot_current.json"
        )
        if json_path.exists():
            return load_disprot_json(
                json_path,
                target=resolved_target,
                split=split,
                splitter=splitter,
            )
        if splitter is not None:
            raise EmbeddingInputError("DisProt TSV region-mode loading does not accept splitter.")
        tsv_path = base / "disprot_current.tsv"
        return load_disprot_tsv(
            tsv_path,
            target=resolved_target,
            split=split or "train",
            region_mode=True,
        )
    phosphoelm_filter = _phosphoelm_source_filter_for_source(spec.name)
    if phosphoelm_filter is not None:
        return load_phosphoelm_dataset(
            _local_phosphoelm_dump(base),
            target=resolved_target,
            split=split,
            source_filter=phosphoelm_filter,
            splitter=splitter,
        )
    raise EmbeddingInputError(
        f"Residue source {spec.name!r} requires explicit input files. "
        f"Use adapter {spec.import_adapter}."
    )



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


def _local_phosphoelm_dump(root: Path) -> Path:
    candidates: list[Path] = []
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


def _download_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise EmbeddingInputError(f"Could not determine filename for URL {url!r}.")
    return name


def _download_url(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 CBBIO/0.1"})
    with urllib.request.urlopen(request) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


__all__ = [
    "ADAPTER_COLLECTION_METADATA",
    "ADAPTER_DATASETS",
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
