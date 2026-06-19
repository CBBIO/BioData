"""Download and import adapters for residue-level probing datasets."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple, cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from CBBIO.embeddings import EmbeddingInputError

from .biolip import BIOLIP_DOWNLOAD_URLS, BioLipLigandClass, load_biolip_dataset
from .datasets import ObjectiveName, ResidueDataset, ResidueExample, SplitName
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
    download_urls: Tuple[str, ...] = ()
    notes: str = ""



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

def get_residue_source(name: str) -> ResidueSourceSpec:
    """Return one registered residue source specification."""
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
    """Return registered residue source specifications."""
    values = list(RESIDUE_SOURCE_SPECS.values())
    if category is not None:
        normalized = str(category).strip().lower()
        values = [value for value in values if value.category == normalized]
    return values


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
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 CBBIO/0.1"})
    with urlopen(request) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


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
