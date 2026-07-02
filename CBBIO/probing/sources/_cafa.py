"""Shared CAFA Kaggle dataset loading helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
from typing import Literal, cast
import zipfile

from CBBIO.embeddings import EmbeddingDependencyError, EmbeddingInputError

from ..collection_types import CollectionMetadata, DatasetMetadata
from ..datasets import ProteinDataset, ProteinExample, SplitName
from ..splitters import DatasetSplitter


CafaAspect = Literal["bp", "cc", "mf"]

CAFA_COLLECTION_METADATA = CollectionMetadata(
    id="cafa",
    display_name="CAFA",
    description="Kaggle CAFA protein-function prediction training datasets.",
    homepage="https://www.kaggle.com/competitions",
    tags=("protein", "function", "gene_ontology", "go", "cafa", "kaggle"),
)

CAFA_ASPECTS: tuple[CafaAspect, ...] = ("bp", "cc", "mf")
CAFA_ASPECT_NAMES: Mapping[CafaAspect, str] = {
    "bp": "biological process",
    "cc": "cellular component",
    "mf": "molecular function",
}
_CAFA_ASPECT_CODES: Mapping[str, CafaAspect] = {
    "BPO": "bp",
    "BP": "bp",
    "P": "bp",
    "PROCESS": "bp",
    "BIOLOGICAL_PROCESS": "bp",
    "BIOLOGICALPROCESS": "bp",
    "GO:0008150": "bp",
    "CCO": "cc",
    "CC": "cc",
    "C": "cc",
    "CELLULAR_COMPONENT": "cc",
    "CELLULARCOMPONENT": "cc",
    "GO:0005575": "cc",
    "MFO": "mf",
    "MF": "mf",
    "F": "mf",
    "FUNCTION": "mf",
    "MOLECULAR_FUNCTION": "mf",
    "MOLECULARFUNCTION": "mf",
    "GO:0003674": "mf",
}


@dataclass(frozen=True)
class CafaChallengeSpec:
    """Configuration for one CAFA Kaggle challenge."""

    version: str
    competition_slug: str
    root_alias: str
    default_splitter: DatasetSplitter

    @property
    def display_name(self) -> str:
        """Return the uppercase challenge name."""
        return self.version.upper()

    @property
    def zip_name(self) -> str:
        """Return the expected Kaggle archive name."""
        return f"{self.competition_slug}.zip"


def build_cafa_datasets(spec: CafaChallengeSpec) -> tuple[DatasetMetadata, ...]:
    """Build metadata entries for one CAFA challenge."""
    return tuple(
        DatasetMetadata(
            id=f"cafa:{spec.version}_{aspect}",
            name=f"{spec.version}_{aspect}",
            display_name=f"{spec.display_name} {CAFA_ASPECT_NAMES[aspect]}",
            collection="cafa",
            source=f"{spec.display_name} Kaggle competition",
            category="function_prediction",
            task_class="function",
            preferred_metric="go_weighted_fmax",
            description=(
                f"Source: {spec.display_name} Kaggle competition. "
                "Class: function_prediction. "
                "Split system: deterministic hash split over training proteins."
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
            homepage=f"https://www.kaggle.com/competitions/{spec.competition_slug}/data",
            download_adapter=f"download_{spec.version}_dataset",
            import_adapter=f"load_{spec.version}_dataset",
            loader=f"load_{spec.version}_dataset",
            tags=("protein", "function", "gene_ontology", "go", "cafa", spec.version, aspect),
            notes=(
                f"Requires a local Kaggle {spec.display_name} download containing "
                "Train/train_terms.tsv and Train/train_sequences.fasta. Optional ontology "
                "and IA files such as go-basic.obo and IA.txt are recorded in metadata when present."
            ),
        )
        for aspect in CAFA_ASPECTS
    )


def load_cafa_dataset(
    root: str | Path,
    *,
    spec: CafaChallengeSpec,
    name: str,
    split: str | Sequence[str] | None = None,
    target: str | None = None,
    splitter: DatasetSplitter | None = None,
    go_obo_path: str | Path | None = None,
    ia_path: str | Path | None = None,
) -> ProteinDataset:
    """Load one CAFA aspect dataset from local Kaggle competition files."""
    aspect = parse_cafa_dataset_name(name, spec=spec)
    expected_target = f"go_{aspect}"
    resolved_target = target or expected_target
    if resolved_target != expected_target:
        raise EmbeddingInputError(
            f"{spec.display_name} dataset {name!r} only supports target {expected_target!r}."
        )
    root_path = resolve_cafa_root(Path(root).expanduser(), spec=spec)
    sequence_path = root_path / "Train" / "train_sequences.fasta"
    terms_path = root_path / "Train" / "train_terms.tsv"
    sequences = _read_fasta_sequences(sequence_path, spec=spec)
    labels_by_id = _read_train_terms(terms_path, aspect=aspect, spec=spec)
    ontology_path = _resolve_go_obo_path(root_path, go_obo_path=go_obo_path, spec=spec)
    information_accretion_path = _resolve_ia_path(root_path, ia_path=ia_path, spec=spec)

    examples = [
        ProteinExample(
            id=protein_id,
            sequence=sequence,
            labels={resolved_target: labels_by_id[protein_id]},
            split="train",
            metadata=_cafa_metadata(
                spec=spec,
                aspect=aspect,
                labels=labels_by_id[protein_id],
                terms_path=terms_path,
                sequence_path=sequence_path,
                ontology_path=ontology_path,
                ia_path=information_accretion_path,
            ),
        )
        for protein_id, sequence in sequences.items()
        if protein_id in labels_by_id
    ]
    if not examples:
        observed_aspects = _observed_aspects(terms_path)
        raise EmbeddingInputError(
            f"{spec.display_name} dataset {name!r} has no proteins with {aspect!r} labels "
            f"in {terms_path}. Observed aspect values: {observed_aspects}."
        )

    split_dataset = cast(
        ProteinDataset,
        (splitter or spec.default_splitter).split_dataset(ProteinDataset(examples)),
    )
    selected_splits = _resolve_splits(split, spec=spec)
    if selected_splits is None:
        return split_dataset
    return ProteinDataset(
        example for example in split_dataset.examples if example.split in selected_splits
    )


def download_cafa_dataset(root: str | Path, *, spec: CafaChallengeSpec, force: bool = False) -> list[Path]:
    """Download and unzip one CAFA competition with the Kaggle CLI."""
    root_path = Path(root).expanduser()
    output_dir = _download_output_dir(root_path, spec=spec)
    if not force:
        try:
            resolved = resolve_cafa_root(root_path, spec=spec)
        except EmbeddingInputError:
            resolved = None
        if resolved is not None:
            return [resolved]
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise EmbeddingDependencyError(
            f"Kaggle CLI is required to download {spec.display_name}. Install with "
            "`pip install kaggle`, configure Kaggle credentials, or download the competition "
            "files manually."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        kaggle,
        "competitions",
        "download",
        "-c",
        spec.competition_slug,
        "-p",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        env=_kaggle_download_env(),
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown Kaggle CLI error"
        raise EmbeddingInputError(_format_cafa_download_error(message, spec=spec))
    _extract_cafa_zip(output_dir, spec=spec)
    resolved = resolve_cafa_root(root_path, spec=spec)
    return [resolved]


def parse_cafa_dataset_name(name: str, *, spec: CafaChallengeSpec) -> CafaAspect:
    """Parse a CAFA dataset name into a GO aspect."""
    normalized = str(name).strip().lower().replace("-", "_")
    for prefix in (f"{spec.version}:", "cafa:"):
        if normalized.startswith(prefix):
            normalized = normalized.partition(":")[2]
    aliases = {
        "bp": "bp",
        "go_bp": "bp",
        f"{spec.version}_bp": "bp",
        "cc": "cc",
        "go_cc": "cc",
        f"{spec.version}_cc": "cc",
        "mf": "mf",
        "go_mf": "mf",
        f"{spec.version}_mf": "mf",
    }
    aspect = aliases.get(normalized)
    if aspect is None:
        raise EmbeddingInputError(
            f"{spec.display_name} dataset name must be one of: "
            f"{spec.version}_bp, {spec.version}_cc, {spec.version}_mf."
        )
    return cast(CafaAspect, aspect)


def resolve_cafa_root(root: Path, *, spec: CafaChallengeSpec) -> Path:
    """Resolve a directory containing a CAFA Kaggle training layout."""
    candidates = (
        root,
        root / spec.competition_slug,
        root / spec.root_alias,
    )
    for candidate in candidates:
        if (candidate / "Train" / "train_terms.tsv").exists() and (
            candidate / "Train" / "train_sequences.fasta"
        ).exists():
            return candidate
    raise EmbeddingInputError(
        f"No {spec.display_name} Kaggle layout found under {root}. Expected "
        "Train/train_terms.tsv and Train/train_sequences.fasta."
    )


def _download_output_dir(root: Path, *, spec: CafaChallengeSpec) -> Path:
    if root.name in {spec.competition_slug, spec.root_alias}:
        return root
    return root / spec.competition_slug


def _format_cafa_download_error(message: str, *, spec: CafaChallengeSpec) -> str:
    if "403" in message or "Forbidden" in message:
        return (
            f"{spec.display_name} Kaggle download failed with 403 Forbidden. "
            f"Open the {spec.display_name} competition page in the Kaggle account used by your token, "
            "accept the competition rules/join the competition, then retry. "
            f"Kaggle CLI output: {message}"
        )
    return f"{spec.display_name} Kaggle download failed: {message}"


def _extract_cafa_zip(output_dir: Path, *, spec: CafaChallengeSpec) -> None:
    try:
        resolve_cafa_root(output_dir, spec=spec)
        return
    except EmbeddingInputError:
        pass
    zip_path = output_dir / spec.zip_name
    if not zip_path.exists():
        zip_candidates = sorted(output_dir.glob("*.zip"))
        if len(zip_candidates) == 1:
            zip_path = zip_candidates[0]
        else:
            raise EmbeddingInputError(
                f"{spec.display_name} Kaggle download did not produce {spec.zip_name!r} in {output_dir}."
            )
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(output_dir)
    except zipfile.BadZipFile as exc:
        raise EmbeddingInputError(
            f"{spec.display_name} Kaggle archive is not a valid zip file: {zip_path}."
        ) from exc


def _kaggle_download_env() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("KAGGLE_API_TOKEN"):
        return env
    token_path = Path.home() / ".kaggle" / "access_token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            env["KAGGLE_API_TOKEN"] = token
    return env


def _read_fasta_sequences(path: Path, *, spec: CafaChallengeSpec) -> dict[str, str]:
    if not path.exists():
        raise EmbeddingInputError(f"{spec.display_name} sequence FASTA not found: {path}.")
    sequences: dict[str, str] = {}
    current_ids: tuple[str, ...] = ()
    current_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if current_ids:
                    _store_sequence_aliases(sequences, current_ids, current_parts, path, spec=spec)
                current_ids = _fasta_record_ids(text)
                current_parts = []
                if not current_ids:
                    raise EmbeddingInputError(
                        f"{spec.display_name} FASTA {path} line {line_number} has an empty id."
                    )
                continue
            if not current_ids:
                raise EmbeddingInputError(
                    f"{spec.display_name} FASTA {path} line {line_number} appears before a header."
                )
            current_parts.append(text)
    if current_ids:
        _store_sequence_aliases(sequences, current_ids, current_parts, path, spec=spec)
    if not sequences:
        raise EmbeddingInputError(f"{spec.display_name} FASTA contains no sequences: {path}.")
    return sequences


def _fasta_record_ids(header: str) -> tuple[str, ...]:
    tokens = header[1:].split()
    if not tokens:
        return ()
    identifiers = [tokens[0].strip()]
    for token in tokens[:2]:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1].strip():
            identifiers.append(parts[1].strip())
    return tuple(dict.fromkeys(identifier for identifier in identifiers if identifier))


def _store_sequence_aliases(
    sequences: dict[str, str],
    protein_ids: Sequence[str],
    parts: Sequence[str],
    path: Path,
    *,
    spec: CafaChallengeSpec,
) -> None:
    sequence = _finalize_sequence(protein_ids[0], parts, path, spec=spec)
    for protein_id in protein_ids:
        existing = sequences.get(protein_id)
        if existing is not None and existing != sequence:
            raise EmbeddingInputError(
                f"{spec.display_name} FASTA record id {protein_id!r} appears with conflicting sequences "
                f"in {path}."
            )
        sequences[protein_id] = sequence


def _finalize_sequence(
    protein_id: str,
    parts: Sequence[str],
    path: Path,
    *,
    spec: CafaChallengeSpec,
) -> str:
    sequence = "".join(parts).strip()
    if not sequence:
        raise EmbeddingInputError(
            f"{spec.display_name} FASTA record {protein_id!r} in {path} has an empty sequence."
        )
    return sequence


def _read_train_terms(path: Path, *, aspect: CafaAspect, spec: CafaChallengeSpec) -> dict[str, list[str]]:
    if not path.exists():
        raise EmbeddingInputError(f"{spec.display_name} training terms file not found: {path}.")
    labels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = _normalized_fieldnames(reader.fieldnames)
        id_field = fieldnames.get("entryid")
        term_field = fieldnames.get("term")
        aspect_field = fieldnames.get("aspect")
        if id_field is None or term_field is None or aspect_field is None:
            raise EmbeddingInputError(
                f"{spec.display_name} train_terms.tsv must contain EntryID, term, and aspect columns."
            )
        for line_number, row in enumerate(reader, start=2):
            protein_id = str(row.get(id_field, "")).strip()
            term = str(row.get(term_field, "")).strip()
            row_aspect = _normalize_aspect(
                row.get(aspect_field, ""),
                path=path,
                line_number=line_number,
                spec=spec,
            )
            if not protein_id or not term:
                raise EmbeddingInputError(
                    f"{spec.display_name} train_terms.tsv {path} line {line_number} "
                    "has an empty EntryID or term."
                )
            if row_aspect == aspect:
                labels.setdefault(protein_id, set()).add(term)
    return {protein_id: sorted(terms) for protein_id, terms in labels.items() if terms}


def _observed_aspects(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = _normalized_fieldnames(reader.fieldnames)
        aspect_field = fieldnames.get("aspect")
        if aspect_field is None:
            return []
        values = sorted(
            {
                str(row.get(aspect_field, "")).strip()
                for index, row in enumerate(reader)
                if index < 100000 and str(row.get(aspect_field, "")).strip()
            }
        )
    return values[:20]


def _normalized_fieldnames(fieldnames: Sequence[str] | None) -> dict[str, str]:
    return {field.strip().lower().replace("_", ""): field for field in fieldnames or []}


def _normalize_aspect(
    value: object,
    *,
    path: Path,
    line_number: int,
    spec: CafaChallengeSpec,
) -> CafaAspect:
    normalized = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    aspect = _CAFA_ASPECT_CODES.get(normalized) or _CAFA_ASPECT_CODES.get(
        normalized.replace("_", "")
    )
    if aspect is None:
        raise EmbeddingInputError(
            f"{spec.display_name} train_terms.tsv {path} line {line_number} "
            f"has unsupported aspect {value!r}."
        )
    return aspect


def _resolve_splits(
    split: str | Sequence[str] | None,
    *,
    spec: CafaChallengeSpec,
) -> set[SplitName] | None:
    if split is None:
        return None
    values = (split,) if isinstance(split, str) else tuple(split)
    resolved: set[SplitName] = set()
    for value in values:
        normalized = str(value).strip().lower()
        if normalized == "valid":
            normalized = "val"
        if normalized not in {"train", "val", "test"}:
            raise EmbeddingInputError(f"{spec.display_name} split must be one of: train, val, test.")
        resolved.add(cast(SplitName, normalized))
    return resolved


def _resolve_go_obo_path(
    root: Path,
    *,
    go_obo_path: str | Path | None,
    spec: CafaChallengeSpec,
) -> Path | None:
    if go_obo_path is not None:
        path = Path(go_obo_path).expanduser()
        if not path.exists():
            raise EmbeddingInputError(f"{spec.display_name} GO ontology file not found: {path}.")
        return path
    for directory in (root, root / "Train"):
        for name in ("go-basic.obo", "go.obo"):
            candidate = directory / name
            if candidate.exists() and candidate.suffix == ".obo":
                return candidate
    return None


def _resolve_ia_path(
    root: Path,
    *,
    ia_path: str | Path | None,
    spec: CafaChallengeSpec,
) -> Path | None:
    if ia_path is not None:
        path = Path(ia_path).expanduser()
        if not path.exists():
            raise EmbeddingInputError(
                f"{spec.display_name} information-accretion weight file not found: {path}."
            )
        return path
    for directory in (root, root / "Train"):
        for name in ("IA.txt", "IA.tsv", "ia.txt", "ia.tsv"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def _cafa_metadata(
    *,
    spec: CafaChallengeSpec,
    aspect: CafaAspect,
    labels: Sequence[str],
    terms_path: Path,
    sequence_path: Path,
    ontology_path: Path | None,
    ia_path: Path | None,
) -> dict[str, str | list[str]]:
    metadata: dict[str, str | list[str]] = {
        "source": spec.version,
        "aspect": aspect,
        "labels_propagated": list(labels),
        "train_terms_path": str(terms_path),
        "train_sequences_path": str(sequence_path),
    }
    if ontology_path is not None:
        metadata["go_obo_path"] = str(ontology_path)
    if ia_path is not None:
        metadata["ia_path"] = str(ia_path)
    return metadata


__all__ = [
    "CAFA_ASPECTS",
    "CAFA_ASPECT_NAMES",
    "CAFA_COLLECTION_METADATA",
    "CafaAspect",
    "CafaChallengeSpec",
    "build_cafa_datasets",
    "download_cafa_dataset",
    "load_cafa_dataset",
    "parse_cafa_dataset_name",
    "resolve_cafa_root",
]
