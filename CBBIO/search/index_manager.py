"""Persistent, streaming FAISS indexes for large BioData embedding collections."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from ..BioData import BioDataError
from ..types import DistanceMetric
from .utils import as_numpy_matrix, import_faiss

if TYPE_CHECKING:
    from ..BioData import BioDataClient


IndexState = Literal["missing", "current", "stale", "invalid"]

_MANIFEST_FORMAT_VERSION = 1
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_VALID_METRICS = {"l2", "cosine", "inner_product"}


class SearchIndexError(BioDataError):
    """Base exception for persistent local search indexes."""


class SearchIndexNotFoundError(SearchIndexError):
    """Raised when a requested persistent search index is absent."""


class SearchIndexStaleError(SearchIndexError):
    """Raised when an index does not match the requested source revision."""


@dataclass(frozen=True)
class IndexKey:
    """Identify one local index for a database embedding collection."""

    database_label: str
    embedding_type_id: int
    layer_index: int
    metric: DistanceMetric
    dimension: int

    def __post_init__(self) -> None:
        if not _SAFE_PATH_COMPONENT.fullmatch(self.database_label):
            raise ValueError("database_label must contain only letters, digits, '.', '_' or '-'.")
        if self.embedding_type_id < 0:
            raise ValueError("embedding_type_id must be non-negative.")
        if self.layer_index < 0:
            raise ValueError("layer_index must be non-negative.")
        if self.metric not in _VALID_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric!r}.")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")

    @classmethod
    def from_biodata(
        cls,
        client: BioDataClient,
        *,
        database_label: str,
        embedding_type_id: int,
        layer_index: int = 0,
        metric: DistanceMetric = "cosine",
    ) -> IndexKey:
        """Create a key using the stored dimension of a BioData embedding collection.

        Raises:
            NotFoundError: If the embedding type or layer is unavailable in BioData.
            ValueError: If the resulting key values are invalid.
        """
        return cls(
            database_label=database_label,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            metric=metric,
            dimension=client.embedding_index_dimension(embedding_type_id, layer_index),
        )


@dataclass(frozen=True)
class IndexBuildSpec:
    """Configure a streaming FAISS IVF-PQ build."""

    nlist: int
    subquantizers: int = 64
    bits_per_code: int = 8
    training_sample_size: int = 100_000
    training_sample_seed: int = 0
    nprobe: int = 64

    def __post_init__(self) -> None:
        if self.nlist <= 0:
            raise ValueError("nlist must be positive.")
        if self.subquantizers <= 0:
            raise ValueError("subquantizers must be positive.")
        if self.bits_per_code <= 0:
            raise ValueError("bits_per_code must be positive.")
        if self.training_sample_size <= 0:
            raise ValueError("training_sample_size must be positive.")
        if self.training_sample_seed < 0:
            raise ValueError("training_sample_seed must be non-negative.")
        if self.nprobe <= 0:
            raise ValueError("nprobe must be positive.")


@dataclass(frozen=True)
class IndexVectorBatch:
    """Provide database vector IDs and their embeddings for one streaming batch."""

    sequence_ids: Any
    vectors: Any


@dataclass(frozen=True)
class ExactVectorBatch:
    """Provide protein IDs and embeddings for one exact-store streaming batch."""

    sequence_ids: Any
    protein_ids: Any
    vectors: Any


@dataclass(frozen=True)
class IndexArtifact:
    """Name the files that make up one persistent local index."""

    directory: Path
    index_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class IndexManifest:
    """Record provenance and configuration for a persisted index."""

    key: IndexKey
    vector_count: int
    source_revision: str | None
    index_parameters: Mapping[str, int | float | str]
    created_at: str
    format_version: int = _MANIFEST_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.vector_count < 0:
            raise ValueError("vector_count must be non-negative.")
        if self.format_version != _MANIFEST_FORMAT_VERSION:
            raise ValueError(f"Unsupported manifest format: {self.format_version}.")
        object.__setattr__(self, "index_parameters", MappingProxyType(dict(self.index_parameters)))

    def to_dict(self) -> dict[str, Any]:
        """Serialize this manifest into JSON-compatible values."""
        return {
            "format_version": self.format_version,
            "key": {
                "database_label": self.key.database_label,
                "embedding_type_id": self.key.embedding_type_id,
                "layer_index": self.key.layer_index,
                "metric": self.key.metric,
                "dimension": self.key.dimension,
            },
            "vector_count": self.vector_count,
            "source_revision": self.source_revision,
            "index_parameters": dict(self.index_parameters),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IndexManifest:
        """Deserialize an index manifest from JSON-compatible values."""
        key_payload = payload.get("key")
        if not isinstance(key_payload, Mapping):
            raise SearchIndexError("Index manifest is missing a valid key.")
        key_values = cast(Mapping[str, Any], key_payload)
        try:
            key = IndexKey(
                database_label=str(key_values["database_label"]),
                embedding_type_id=int(key_values["embedding_type_id"]),
                layer_index=int(key_values["layer_index"]),
                metric=cast(DistanceMetric, str(key_values["metric"])),
                dimension=int(key_values["dimension"]),
            )
            parameters = payload.get("index_parameters", {})
            if not isinstance(parameters, Mapping):
                raise TypeError("index_parameters is not an object")
            parameter_values = cast(Mapping[str, int | float | str], parameters)
            return cls(
                key=key,
                vector_count=int(payload["vector_count"]),
                source_revision=_optional_string(payload.get("source_revision")),
                index_parameters=dict(parameter_values),
                created_at=str(payload["created_at"]),
                format_version=int(payload["format_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SearchIndexError("Index manifest has invalid contents.") from exc


@dataclass(frozen=True)
class IndexInspection:
    """Describe whether a persisted index can serve a requested collection."""

    state: IndexState
    artifact: IndexArtifact
    manifest: IndexManifest | None = None
    reason: str | None = None


@dataclass(frozen=True)
class IndexCandidate:
    """Identify one ANN candidate returned by a persistent local index."""

    sequence_id: int
    score: float


@dataclass(frozen=True)
class ExactStoreArtifact:
    """Name the files that make up a portable exact vector store."""

    directory: Path
    vectors_path: Path
    metadata_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class ExactStoreManifest:
    """Record provenance and layout for a portable exact vector store."""

    database_label: str
    embedding_type_id: int
    layer_index: int
    dimension: int
    vector_count: int
    source_revision: str | None
    created_at: str
    vector_dtype: str = "float16"
    format_version: int = _MANIFEST_FORMAT_VERSION
    source_read_seconds: float | None = None
    vector_write_seconds: float | None = None
    metadata_write_seconds: float | None = None

    def __post_init__(self) -> None:
        if not _SAFE_PATH_COMPONENT.fullmatch(self.database_label):
            raise ValueError("database_label must contain only letters, digits, '.', '_' or '-'.")
        if self.embedding_type_id < 0 or self.layer_index < 0:
            raise ValueError("embedding_type_id and layer_index must be non-negative.")
        if self.dimension <= 0:
            raise ValueError("dimension must be positive.")
        if self.vector_count < 0:
            raise ValueError("vector_count must be non-negative.")
        if self.vector_dtype != "float16":
            raise ValueError("Exact stores currently require float16 vectors.")
        if self.format_version != _MANIFEST_FORMAT_VERSION:
            raise ValueError(f"Unsupported manifest format: {self.format_version}.")
        for duration in (
            self.source_read_seconds,
            self.vector_write_seconds,
            self.metadata_write_seconds,
        ):
            if duration is not None and (not math.isfinite(duration) or duration < 0):
                raise ValueError("Exact-store timing values must be finite and non-negative.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize this manifest into JSON-compatible values."""
        return {
            "format_version": self.format_version,
            "database_label": self.database_label,
            "embedding_type_id": self.embedding_type_id,
            "layer_index": self.layer_index,
            "dimension": self.dimension,
            "vector_count": self.vector_count,
            "source_revision": self.source_revision,
            "vector_dtype": self.vector_dtype,
            "created_at": self.created_at,
            "source_read_seconds": self.source_read_seconds,
            "vector_write_seconds": self.vector_write_seconds,
            "metadata_write_seconds": self.metadata_write_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExactStoreManifest":
        """Deserialize an exact-store manifest from JSON-compatible values."""
        try:
            return cls(
                database_label=str(payload["database_label"]),
                embedding_type_id=int(payload["embedding_type_id"]),
                layer_index=int(payload["layer_index"]),
                dimension=int(payload["dimension"]),
                vector_count=int(payload["vector_count"]),
                source_revision=_optional_string(payload.get("source_revision")),
                vector_dtype=str(payload["vector_dtype"]),
                created_at=str(payload["created_at"]),
                format_version=int(payload["format_version"]),
                source_read_seconds=_optional_duration(payload.get("source_read_seconds")),
                vector_write_seconds=_optional_duration(payload.get("vector_write_seconds")),
                metadata_write_seconds=_optional_duration(payload.get("metadata_write_seconds")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SearchIndexError("Exact-store manifest has invalid contents.") from exc


@dataclass(frozen=True)
class ExactStoreInspection:
    """Describe whether a portable exact vector store can serve a collection."""

    state: IndexState
    artifact: ExactStoreArtifact
    manifest: ExactStoreManifest | None = None
    reason: str | None = None


class IndexManager:
    """Build, inspect, and query persistent IVF-PQ indexes and exact vector stores."""

    def __init__(self, root: str | Path, *, search_nprobe: int | None = None) -> None:
        """Create an index manager rooted at a local artifact directory.

        Args:
            root: Directory containing persistent index artifacts.
            search_nprobe: Optional IVF partitions to probe at query time. This overrides
                the value persisted when the index was built without changing its files.
        """
        if search_nprobe is not None and search_nprobe <= 0:
            raise ValueError("search_nprobe must be positive when provided.")
        self.root = Path(root)
        self._search_nprobe = search_nprobe

    def artifact_for(self, key: IndexKey) -> IndexArtifact:
        """Return the persistent file locations for an index key."""
        directory = (
            self.root
            / key.database_label
            / f"embedding-type-{key.embedding_type_id}"
            / f"layer-{key.layer_index}"
            / key.metric
        )
        current_directory = directory / "current"
        return IndexArtifact(
            directory=directory,
            index_path=current_directory / "index.faiss",
            manifest_path=current_directory / "manifest.json",
        )

    def exact_store_artifact_for(
        self,
        *,
        database_label: str,
        embedding_type_id: int,
        layer_index: int = 0,
    ) -> ExactStoreArtifact:
        """Return persistent file locations for an exact vector store."""
        _validate_exact_store_identity(
            database_label=database_label,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
        )
        directory = (
            self.root
            / database_label
            / f"embedding-type-{embedding_type_id}"
            / f"layer-{layer_index}"
            / "exact"
        )
        current_directory = directory / "current"
        return ExactStoreArtifact(
            directory=directory,
            vectors_path=current_directory / "vectors.f16",
            metadata_path=current_directory / "metadata.sqlite",
            manifest_path=current_directory / "manifest.json",
        )

    def inspect_exact_store(
        self,
        *,
        database_label: str,
        embedding_type_id: int,
        layer_index: int = 0,
        source_revision: str | None = None,
    ) -> ExactStoreInspection:
        """Report whether a portable exact store exists and matches its source revision."""
        artifact = self.exact_store_artifact_for(
            database_label=database_label,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
        )
        if not all(path.is_file() for path in (artifact.vectors_path, artifact.metadata_path, artifact.manifest_path)):
            return ExactStoreInspection("missing", artifact, reason="Exact-store files are absent.")
        artifact = _resolved_exact_store_artifact(artifact)
        try:
            manifest = self._read_exact_store_manifest(artifact.manifest_path)
        except SearchIndexError as exc:
            return ExactStoreInspection("invalid", artifact, reason=str(exc))
        if (
            manifest.database_label != database_label
            or manifest.embedding_type_id != int(embedding_type_id)
            or manifest.layer_index != int(layer_index)
        ):
            return ExactStoreInspection("invalid", artifact, manifest, "Exact-store manifest does not match collection.")
        expected_bytes = manifest.vector_count * manifest.dimension * 2
        if artifact.vectors_path.stat().st_size != expected_bytes:
            return ExactStoreInspection("invalid", artifact, manifest, "Exact-store vector file has an unexpected size.")
        if source_revision is not None and manifest.source_revision != source_revision:
            return ExactStoreInspection("stale", artifact, manifest, "Source revision differs from exact-store manifest.")
        return ExactStoreInspection("current", artifact, manifest)

    def build_exact_store(
        self,
        key: IndexKey,
        batch_source: Callable[[], Iterable[ExactVectorBatch]],
        *,
        source_revision: str | None,
        overwrite: bool = False,
    ) -> ExactStoreManifest:
        """Build a portable float16 exact store from one bounded-memory source pass.

        The stored matrix is metric-independent. FAISS CPU and cuVS GPU normalize vectors for
        cosine searches as they materialize their backend-specific exact states.
        """
        artifact = self.exact_store_artifact_for(
            database_label=key.database_label,
            embedding_type_id=key.embedding_type_id,
            layer_index=key.layer_index,
        )
        existing = self.inspect_exact_store(
            database_label=key.database_label,
            embedding_type_id=key.embedding_type_id,
            layer_index=key.layer_index,
        )
        if existing.state != "missing" and not overwrite:
            raise SearchIndexError(
                f"An exact store already exists for {key.database_label!r}, embedding type "
                f"{key.embedding_type_id}, layer {key.layer_index}."
            )

        np = _import_numpy()
        generation_name = uuid.uuid4().hex
        generation_root = artifact.directory / "generations"
        staging_directory = generation_root / f".{generation_name}.tmp"
        generation_root.mkdir(parents=True, exist_ok=True)
        staging_directory.mkdir()
        vectors_path = staging_directory / "vectors.f16"
        metadata_path = staging_directory / "metadata.sqlite"
        vector_count = 0
        source_read_seconds = 0.0
        vector_write_seconds = 0.0
        metadata_write_seconds = 0.0
        try:
            with vectors_path.open("wb") as vector_file, sqlite3.connect(metadata_path) as connection:
                connection.execute(
                    "CREATE TABLE vector_rows (row_index INTEGER PRIMARY KEY, sequence_id INTEGER NOT NULL, protein_id TEXT NOT NULL)"
                )
                batches = iter(batch_source())
                while True:
                    source_started_at = time.perf_counter()
                    try:
                        batch = next(batches)
                    except StopIteration:
                        source_read_seconds += time.perf_counter() - source_started_at
                        break
                    source_read_seconds += time.perf_counter() - source_started_at
                    sequence_ids, protein_ids, vectors = _validated_exact_batch(batch, key=key, np=np)
                    vector_write_started_at = time.perf_counter()
                    vectors.astype(np.float16, copy=False).tofile(vector_file)
                    vector_write_seconds += time.perf_counter() - vector_write_started_at
                    metadata_write_started_at = time.perf_counter()
                    connection.executemany(
                        "INSERT INTO vector_rows (row_index, sequence_id, protein_id) VALUES (?, ?, ?)",
                        [
                            (vector_count + offset, int(sequence_id), str(protein_id))
                            for offset, (sequence_id, protein_id) in enumerate(zip(sequence_ids, protein_ids, strict=True))
                        ],
                    )
                    metadata_write_seconds += time.perf_counter() - metadata_write_started_at
                    vector_count += int(vectors.shape[0])
                vector_write_started_at = time.perf_counter()
                vector_file.flush()
                vector_write_seconds += time.perf_counter() - vector_write_started_at
                metadata_write_started_at = time.perf_counter()
                connection.execute("CREATE INDEX vector_rows_protein_id ON vector_rows (protein_id)")
                connection.commit()
                metadata_write_seconds += time.perf_counter() - metadata_write_started_at
            if vector_count < 1:
                raise SearchIndexError("Cannot build an exact store without vectors.")
            manifest = ExactStoreManifest(
                database_label=key.database_label,
                embedding_type_id=key.embedding_type_id,
                layer_index=key.layer_index,
                dimension=key.dimension,
                vector_count=vector_count,
                source_revision=source_revision,
                created_at=datetime.now(timezone.utc).isoformat(),
                source_read_seconds=source_read_seconds,
                vector_write_seconds=vector_write_seconds,
                metadata_write_seconds=metadata_write_seconds,
            )
            self._write_exact_store_manifest(staging_directory / "manifest.json", manifest)
            _publish_generation(artifact.directory, staging_directory, generation_name)
            return manifest
        except Exception:
            raise

    def load_exact_store(
        self,
        *,
        database_label: str,
        embedding_type_id: int,
        layer_index: int = 0,
        source_revision: str | None = None,
    ) -> ExactStoreInspection:
        """Load metadata for a current portable exact store.

        Raises:
            SearchIndexNotFoundError: If the exact-store files are absent or invalid.
            SearchIndexStaleError: If the store source revision is stale.
        """
        inspection = self.inspect_exact_store(
            database_label=database_label,
            embedding_type_id=embedding_type_id,
            layer_index=layer_index,
            source_revision=source_revision,
        )
        if inspection.state in {"missing", "invalid"}:
            raise SearchIndexNotFoundError(inspection.reason or "Persistent exact store is unavailable.")
        if inspection.state == "stale":
            raise SearchIndexStaleError(inspection.reason or "Persistent exact store is stale.")
        return inspection

    def iter_exact_store_batches(
        self,
        inspection: ExactStoreInspection,
        *,
        batch_size: int = 10_000,
    ) -> Iterable[tuple[list[str], Any]]:
        """Yield float32 vector batches and protein IDs from a loaded exact store."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if inspection.state != "current" or inspection.manifest is None:
            raise SearchIndexError("A current exact-store inspection is required to read vectors.")
        np = _import_numpy()
        manifest = inspection.manifest
        matrix = np.memmap(
            inspection.artifact.vectors_path,
            dtype=np.float16,
            mode="r",
            shape=(manifest.vector_count, manifest.dimension),
        )
        with sqlite3.connect(inspection.artifact.metadata_path) as connection:
            cursor = connection.execute("SELECT protein_id FROM vector_rows ORDER BY row_index")
            offset = 0
            while rows := cursor.fetchmany(batch_size):
                next_offset = offset + len(rows)
                yield [str(row[0]) for row in rows], np.asarray(matrix[offset:next_offset], dtype=np.float32)
                offset = next_offset
        if offset != manifest.vector_count:
            raise SearchIndexError("Exact-store metadata row count does not match its manifest.")

    def _exact_store_candidate_distances(
        self,
        inspection: ExactStoreInspection,
        query_vector: Any,
        protein_ids: Iterable[str],
        *,
        metric: DistanceMetric,
    ) -> dict[str, float]:
        """Compute deterministic float64 distances for exact-store candidate protein IDs."""
        if inspection.state != "current" or inspection.manifest is None:
            raise SearchIndexError("A current exact-store inspection is required to score candidates.")
        candidate_ids = list(dict.fromkeys(str(protein_id) for protein_id in protein_ids))
        if not candidate_ids:
            return {}
        np = _import_numpy()
        manifest = inspection.manifest
        query = as_numpy_matrix([query_vector]).reshape(-1).astype(np.float16).astype(np.float64)
        if int(query.shape[0]) != manifest.dimension:
            raise SearchIndexError(
                f"Exact-store query dimension must be {manifest.dimension}, got {int(query.shape[0])}."
            )
        rows_by_protein_id: dict[str, int] = {}
        with sqlite3.connect(inspection.artifact.metadata_path) as connection:
            for start in range(0, len(candidate_ids), 900):
                identifiers = candidate_ids[start : start + 900]
                placeholders = ", ".join("?" for _ in identifiers)
                rows = connection.execute(
                    "SELECT protein_id, row_index FROM vector_rows "
                    f"WHERE protein_id IN ({placeholders})",
                    identifiers,
                )
                rows_by_protein_id.update({str(protein_id): int(row_index) for protein_id, row_index in rows})
        ordered_ids = [protein_id for protein_id in candidate_ids if protein_id in rows_by_protein_id]
        if not ordered_ids:
            return {}
        matrix = np.memmap(
            inspection.artifact.vectors_path,
            dtype=np.float16,
            mode="r",
            shape=(manifest.vector_count, manifest.dimension),
        )
        row_indices = np.asarray([rows_by_protein_id[protein_id] for protein_id in ordered_ids], dtype=np.int64)
        vectors = np.asarray(matrix[row_indices], dtype=np.float64)
        if metric == "cosine":
            query_norm = float(np.linalg.norm(query))
            vector_norms = np.linalg.norm(vectors, axis=1)
            denominators = vector_norms * query_norm
            similarities = np.divide(
                vectors @ query,
                denominators,
                out=np.zeros(len(ordered_ids), dtype=np.float64),
                where=denominators != 0.0,
            )
            distances = np.clip(1.0 - similarities, 0.0, 2.0)
        elif metric == "l2":
            distances = np.linalg.norm(vectors - query, axis=1)
        else:
            distances = -(vectors @ query)
        return {
            protein_id: float(distance)
            for protein_id, distance in zip(ordered_ids, distances, strict=True)
        }

    def inspect(self, key: IndexKey, *, source_revision: str | None = None) -> IndexInspection:
        """Report whether the requested index exists and matches its source revision."""
        artifact = self.artifact_for(key)
        if artifact.index_path.is_file() and artifact.manifest_path.is_file():
            artifact = _resolved_artifact(artifact)
        else:
            legacy_artifact = _legacy_artifact_for(artifact.directory)
            if legacy_artifact.index_path.is_file() and legacy_artifact.manifest_path.is_file():
                artifact = legacy_artifact
            else:
                return IndexInspection("missing", artifact, reason="Index files are absent.")
        try:
            manifest = self._read_manifest(artifact.manifest_path)
        except SearchIndexError as exc:
            return IndexInspection("invalid", artifact, reason=str(exc))
        if manifest.key != key:
            return IndexInspection("invalid", artifact, manifest, "Manifest key does not match index key.")
        if source_revision is not None and manifest.source_revision != source_revision:
            return IndexInspection("stale", artifact, manifest, "Source revision differs from index manifest.")
        return IndexInspection("current", artifact, manifest)

    def build_ivf_pq(
        self,
        key: IndexKey,
        batch_source: Callable[[], Iterable[IndexVectorBatch]],
        *,
        source_revision: str | None,
        spec: IndexBuildSpec,
        overwrite: bool = False,
    ) -> IndexManifest:
        """Build and persist an IVF-PQ index from two bounded-memory source passes.

        Args:
            key: Embedding collection to index.
            batch_source: Factory that yields a fresh stream for each required pass.
            source_revision: Database watermark used to detect a stale local index.
            spec: IVF-PQ parameters.
            overwrite: Replace an existing index only after a successful build.

        Returns:
            Provenance manifest for the completed index.

        Raises:
            SearchIndexError: If the source is invalid or an index already exists.
            DriverDependencyError: If FAISS or NumPy is unavailable.
        """
        artifact = self.artifact_for(key)
        existing = self.inspect(key)
        if existing.state != "missing" and not overwrite:
            raise SearchIndexError(
                f"An index already exists for {key.database_label!r}, embedding type "
                f"{key.embedding_type_id}, layer {key.layer_index}, metric {key.metric!r}."
            )

        np = _import_numpy()
        if key.dimension % spec.subquantizers != 0:
            raise SearchIndexError(
                f"dimension {key.dimension} must be divisible by subquantizers {spec.subquantizers}."
            )
        sample = self._collect_training_sample(
            batch_source(),
            key=key,
            sample_size=spec.training_sample_size,
            seed=spec.training_sample_seed,
        )
        if int(sample.shape[0]) < spec.nlist:
            raise SearchIndexError(
                f"Training sample has {int(sample.shape[0])} vectors but nlist is {spec.nlist}."
            )

        faiss = import_faiss()
        metric_type = faiss.METRIC_L2 if key.metric == "l2" else faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlatL2(key.dimension) if key.metric == "l2" else faiss.IndexFlatIP(key.dimension)
        index = faiss.IndexIVFPQ(
            quantizer,
            key.dimension,
            spec.nlist,
            spec.subquantizers,
            spec.bits_per_code,
            metric_type,
        )
        index.train(sample)
        if hasattr(index, "nprobe"):
            index.nprobe = min(spec.nprobe, spec.nlist)

        vector_count = 0
        for batch in batch_source():
            sequence_ids, vectors = _validated_batch(batch, key=key, np=np)
            index.add_with_ids(vectors, sequence_ids)
            vector_count += int(vectors.shape[0])

        manifest = IndexManifest(
            key=key,
            vector_count=vector_count,
            source_revision=source_revision,
            index_parameters={
                "backend": "faiss_ivfpq",
                "nlist": spec.nlist,
                "subquantizers": spec.subquantizers,
                "bits_per_code": spec.bits_per_code,
                "nprobe": min(spec.nprobe, spec.nlist),
                "training_sample_seed": spec.training_sample_seed,
                "training_sample_strategy": "smallest_splitmix64_sequence_id",
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._publish_index(artifact, index, manifest)
        return manifest

    def append_ivf_pq(
        self,
        key: IndexKey,
        batches: Iterable[IndexVectorBatch],
        *,
        source_revision: str | None,
    ) -> IndexManifest:
        """Append new, unique sequence IDs to a current IVF-PQ index.

        The caller must stream only IDs that are not already present. Deletions and modified
        embeddings require a full rebuild because IVF-PQ does not efficiently rewrite codes.
        """
        inspection = self.inspect(key)
        if inspection.state != "current" or inspection.manifest is None:
            raise SearchIndexNotFoundError(inspection.reason or "Persistent search index is unavailable.")

        np = _import_numpy()
        index = self.load(key)
        appended_count = 0
        for batch in batches:
            sequence_ids, vectors = _validated_batch(batch, key=key, np=np)
            index.add_with_ids(vectors, sequence_ids)
            appended_count += int(vectors.shape[0])

        manifest = IndexManifest(
            key=key,
            vector_count=inspection.manifest.vector_count + appended_count,
            source_revision=source_revision,
            index_parameters=inspection.manifest.index_parameters,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._publish_index(self.artifact_for(key), index, manifest)
        return manifest

    def load(self, key: IndexKey, *, source_revision: str | None = None) -> Any:
        """Load a current FAISS index for an embedding collection.

        Raises:
            SearchIndexNotFoundError: If the index files are absent or invalid.
            SearchIndexStaleError: If the index source revision is stale.
        """
        inspection = self.inspect(key, source_revision=source_revision)
        if inspection.state == "missing" or inspection.state == "invalid":
            raise SearchIndexNotFoundError(inspection.reason or "Persistent search index is unavailable.")
        if inspection.state == "stale":
            raise SearchIndexStaleError(inspection.reason or "Persistent search index is stale.")
        index = import_faiss().read_index(str(inspection.artifact.index_path))
        if self._search_nprobe is not None:
            nlist = int(getattr(index, "nlist", 0))
            if nlist <= 0 or not hasattr(index, "nprobe"):
                raise SearchIndexError("Persistent index does not support an IVF nprobe override.")
            index.nprobe = min(self._search_nprobe, nlist)
        return index

    def search(
        self,
        index: Any,
        query_vector: Any,
        *,
        key: IndexKey,
        candidate_count: int,
    ) -> list[IndexCandidate]:
        """Return ANN sequence IDs from an already-loaded persistent index."""
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        np = _import_numpy()
        query = as_numpy_matrix([query_vector])
        if int(query.shape[1]) != key.dimension:
            raise ValueError(f"Query dimension must be {key.dimension}, got {int(query.shape[1])}.")
        if key.metric == "cosine":
            query = _normalize_rows(query, np=np)
        scores, sequence_ids = index.search(query, candidate_count)
        return [
            IndexCandidate(sequence_id=int(sequence_id), score=float(score))
            for score, sequence_id in zip(scores[0], sequence_ids[0], strict=True)
            if int(sequence_id) >= 0
        ]

    def _collect_training_sample(
        self,
        batches: Iterable[IndexVectorBatch],
        *,
        key: IndexKey,
        sample_size: int,
        seed: int,
    ) -> Any:
        """Collect a bounded deterministic hash sample from a complete vector stream."""
        np = _import_numpy()
        sample = np.empty((sample_size, key.dimension), dtype=np.float32)
        sample_priorities = np.empty(sample_size, dtype=np.uint64)
        collected = 0
        for batch in batches:
            sequence_ids, vectors = _validated_batch(batch, key=key, np=np)
            priorities = _training_sample_priorities(sequence_ids, seed=seed, np=np)
            if collected < sample_size:
                remaining = sample_size - collected
                rows = min(remaining, int(vectors.shape[0]))
                sample[collected : collected + rows] = vectors[:rows]
                sample_priorities[collected : collected + rows] = priorities[:rows]
                collected += rows
                if rows == int(vectors.shape[0]):
                    continue
                vectors = vectors[rows:]
                priorities = priorities[rows:]

            candidate_priorities = np.concatenate((sample_priorities, priorities))
            candidate_vectors = np.vstack((sample, vectors))
            selected_rows = np.argpartition(candidate_priorities, sample_size - 1)[:sample_size]
            sample = candidate_vectors[selected_rows]
            sample_priorities = candidate_priorities[selected_rows]
        return sample[:collected]

    def _read_manifest(self, path: Path) -> IndexManifest:
        """Read and validate one JSON manifest."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SearchIndexError(f"Cannot read index manifest at {path}.") from exc
        if not isinstance(payload, Mapping):
            raise SearchIndexError(f"Index manifest at {path} is not an object.")
        return IndexManifest.from_dict(cast(Mapping[str, Any], payload))

    def _read_exact_store_manifest(self, path: Path) -> ExactStoreManifest:
        """Read and validate one exact-store JSON manifest."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SearchIndexError(f"Cannot read exact-store manifest at {path}.") from exc
        if not isinstance(payload, Mapping):
            raise SearchIndexError(f"Exact-store manifest at {path} is not an object.")
        return ExactStoreManifest.from_dict(cast(Mapping[str, Any], payload))

    def _publish_index(self, artifact: IndexArtifact, index: Any, manifest: IndexManifest) -> None:
        """Publish an index and manifest together through an atomic current-generation switch."""
        artifact.directory.mkdir(parents=True, exist_ok=True)
        generation_root = artifact.directory / "generations"
        generation_root.mkdir(exist_ok=True)
        generation_name = uuid.uuid4().hex
        staging_directory = generation_root / f".{generation_name}.tmp"
        generation_directory = generation_root / generation_name
        staging_directory.mkdir()
        self._write_index(staging_directory / "index.faiss", index)
        self._write_manifest(staging_directory / "manifest.json", manifest)
        os.replace(staging_directory, generation_directory)

        current_path = artifact.directory / "current"
        temporary_current_path = artifact.directory / f".current-{generation_name}.tmp"
        temporary_current_path.symlink_to(Path("generations") / generation_name, target_is_directory=True)
        os.replace(temporary_current_path, current_path)

    def _write_index(self, path: Path, index: Any) -> None:
        """Write one FAISS index inside an unpublished generation directory."""
        temporary_path = path.with_suffix(".faiss.tmp")
        import_faiss().write_index(index, str(temporary_path))
        os.replace(temporary_path, path)

    def _write_manifest(self, path: Path, manifest: IndexManifest) -> None:
        """Write one manifest inside an unpublished generation directory."""
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)

    def _write_exact_store_manifest(self, path: Path, manifest: ExactStoreManifest) -> None:
        """Write one exact-store manifest inside an unpublished generation directory."""
        temporary_path = path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)


def _validate_exact_store_identity(
    *,
    database_label: str,
    embedding_type_id: int,
    layer_index: int,
) -> None:
    """Validate the collection identity used by a metric-independent exact store."""
    if not _SAFE_PATH_COMPONENT.fullmatch(database_label):
        raise ValueError("database_label must contain only letters, digits, '.', '_' or '-'.")
    if embedding_type_id < 0:
        raise ValueError("embedding_type_id must be non-negative.")
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative.")


def _publish_generation(directory: Path, staging_directory: Path, generation_name: str) -> None:
    """Atomically select a complete staged generation as the current artifact."""
    generation_directory = staging_directory.parent / generation_name
    os.replace(staging_directory, generation_directory)
    current_path = directory / "current"
    temporary_current_path = directory / f".current-{generation_name}.tmp"
    temporary_current_path.symlink_to(Path("generations") / generation_name, target_is_directory=True)
    os.replace(temporary_current_path, current_path)


def _resolved_exact_store_artifact(artifact: ExactStoreArtifact) -> ExactStoreArtifact:
    """Bind an exact-store artifact to immutable resolved generation paths."""
    return ExactStoreArtifact(
        directory=artifact.directory,
        vectors_path=artifact.vectors_path.resolve(),
        metadata_path=artifact.metadata_path.resolve(),
        manifest_path=artifact.manifest_path.resolve(),
    )


def _legacy_artifact_for(directory: Path) -> IndexArtifact:
    """Return the artifact layout written before generation-based publication."""
    return IndexArtifact(
        directory=directory,
        index_path=directory / "index.faiss",
        manifest_path=directory / "manifest.json",
    )


def _resolved_artifact(artifact: IndexArtifact) -> IndexArtifact:
    """Bind a current-generation artifact to immutable resolved file paths."""
    return IndexArtifact(
        directory=artifact.directory,
        index_path=artifact.index_path.resolve(),
        manifest_path=artifact.manifest_path.resolve(),
    )


def _optional_string(value: Any) -> str | None:
    """Normalize an optional JSON string."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Expected a string or null.")
    return value


def _optional_duration(value: Any) -> float | None:
    """Normalize an optional finite non-negative duration."""
    if value is None:
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Expected a finite non-negative duration.")
    return duration


def _import_numpy() -> Any:
    """Import NumPy only when persistent index operations need it."""
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        from ..BioData import DriverDependencyError

        raise DriverDependencyError("NumPy is required for persistent local search indexes. Install with: pip install numpy") from exc
    return np


def _training_sample_priorities(sequence_ids: Any, *, seed: int, np: Any) -> Any:
    """Return deterministic pseudo-random priorities for a stream of sequence IDs."""
    values = np.asarray(sequence_ids, dtype=np.uint64) + np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _validated_exact_batch(batch: ExactVectorBatch, *, key: IndexKey, np: Any) -> tuple[Any, Any, Any]:
    """Convert one exact-store source batch and validate its IDs and vectors."""
    sequence_ids = np.asarray(batch.sequence_ids, dtype=np.int64).reshape(-1)
    protein_ids = [str(value) for value in batch.protein_ids]
    vectors = np.asarray(batch.vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise SearchIndexError("Exact-store vectors must be a two-dimensional matrix.")
    if int(vectors.shape[0]) != int(sequence_ids.shape[0]) or int(vectors.shape[0]) != len(protein_ids):
        raise SearchIndexError("Each exact-store vector batch must have one sequence and protein ID per vector.")
    if int(vectors.shape[1]) != key.dimension:
        raise SearchIndexError(
            f"Exact-store vectors must have dimension {key.dimension}, got {int(vectors.shape[1])}."
        )
    return sequence_ids, protein_ids, vectors


def _validated_batch(batch: IndexVectorBatch, *, key: IndexKey, np: Any) -> tuple[Any, Any]:
    """Convert one stream batch to the index representation and validate it."""
    sequence_ids = np.asarray(batch.sequence_ids, dtype=np.int64).reshape(-1)
    vectors = np.asarray(batch.vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise SearchIndexError("Index vectors must be a two-dimensional matrix.")
    if int(vectors.shape[0]) != int(sequence_ids.shape[0]):
        raise SearchIndexError("Each index vector batch must have one sequence ID per vector.")
    if int(vectors.shape[1]) != key.dimension:
        raise SearchIndexError(
            f"Index vectors must have dimension {key.dimension}, got {int(vectors.shape[1])}."
        )
    if key.metric == "cosine":
        vectors = _normalize_rows(vectors, np=np)
    return sequence_ids, vectors


def _normalize_rows(vectors: Any, *, np: Any) -> Any:
    """L2-normalize a float32 matrix while rejecting zero vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise SearchIndexError("Cosine indexes cannot contain zero vectors.")
    return vectors / norms


__all__ = [
    "ExactStoreArtifact",
    "ExactStoreInspection",
    "ExactStoreManifest",
    "ExactVectorBatch",
    "IndexArtifact",
    "IndexBuildSpec",
    "IndexCandidate",
    "IndexInspection",
    "IndexKey",
    "IndexManager",
    "IndexManifest",
    "IndexState",
    "IndexVectorBatch",
    "SearchIndexError",
    "SearchIndexNotFoundError",
    "SearchIndexStaleError",
]
