from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pytest

from CBBIO import ExactVectorBatch, IndexBuildSpec, IndexKey, IndexManager, IndexVectorBatch


def test_index_manager_builds_persists_and_updates_an_ivf_pq_index(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("faiss")
    key = IndexKey(
        database_label="test-database",
        embedding_type_id=3,
        layer_index=0,
        metric="cosine",
        dimension=4,
    )
    manager = IndexManager(tmp_path)
    vectors = numpy.asarray(
        [[1.0, 0.1 * index, 0.2, 0.3] for index in range(16)],
        dtype=numpy.float32,
    )
    sequence_ids = numpy.arange(100, 116, dtype=numpy.int64)

    def _source() -> Iterable[IndexVectorBatch]:
        return [
            IndexVectorBatch(sequence_ids=sequence_ids[:8], vectors=vectors[:8]),
            IndexVectorBatch(sequence_ids=sequence_ids[8:], vectors=vectors[8:]),
        ]

    manifest = manager.build_ivf_pq(
        key,
        _source,
        source_revision="16",
        spec=IndexBuildSpec(nlist=2, subquantizers=2, bits_per_code=1, training_sample_size=16, nprobe=1),
    )

    assert manifest.vector_count == 16
    assert manifest.index_parameters["training_sample_seed"] == 0
    assert manifest.index_parameters["training_sample_strategy"] == "smallest_splitmix64_sequence_id"
    assert manager.artifact_for(key).index_path.is_file()
    assert manager.inspect(key, source_revision="16").state == "current"
    assert manager.inspect(key, source_revision="17").state == "stale"
    first_inspection = manager.inspect(key, source_revision="16")
    assert first_inspection.artifact.index_path.is_file()
    assert first_inspection.artifact.manifest_path.is_file()
    assert (manager.artifact_for(key).directory / "current").is_symlink()

    class _VectorLike:
        def __init__(self, values: list[float]) -> None:
            self._values = values

        def to_list(self) -> list[float]:
            return self._values

    search_manager = IndexManager(tmp_path, search_nprobe=2)
    index = search_manager.load(key)
    assert int(index.nprobe) == 2

    candidates = search_manager.search(
        index,
        _VectorLike(vectors[0].tolist()),
        key=key,
        candidate_count=16,
    )

    assert {candidate.sequence_id for candidate in candidates} == set(sequence_ids.tolist())

    appended = manager.append_ivf_pq(
        key,
        [
            IndexVectorBatch(
                sequence_ids=numpy.asarray([200], dtype=numpy.int64),
                vectors=numpy.asarray([[1.0, 1.8, 0.2, 0.3]], dtype=numpy.float32),
            )
        ],
        source_revision="17",
    )

    assert appended.vector_count == 17
    assert manager.inspect(key, source_revision="17").state == "current"
    current_inspection = manager.inspect(key, source_revision="17")
    assert current_inspection.artifact.index_path != first_inspection.artifact.index_path
    assert current_inspection.artifact.manifest_path != first_inspection.artifact.manifest_path
    assert first_inspection.artifact.index_path.is_file()
    assert first_inspection.artifact.manifest_path.is_file()
    assert manager.artifact_for(key).index_path.resolve() == current_inspection.artifact.index_path


def test_index_manager_keeps_the_current_generation_when_a_rebuild_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("faiss")
    key = IndexKey(
        database_label="test-database",
        embedding_type_id=3,
        layer_index=0,
        metric="cosine",
        dimension=4,
    )
    manager = IndexManager(tmp_path)
    vectors = numpy.asarray(
        [[1.0, 0.1 * index, 0.2, 0.3] for index in range(16)],
        dtype=numpy.float32,
    )
    sequence_ids = numpy.arange(100, 116, dtype=numpy.int64)

    def _source() -> Iterable[IndexVectorBatch]:
        return [IndexVectorBatch(sequence_ids=sequence_ids, vectors=vectors)]

    spec = IndexBuildSpec(nlist=2, subquantizers=2, bits_per_code=1, training_sample_size=16, nprobe=1)
    manager.build_ivf_pq(key, _source, source_revision="16", spec=spec)
    before = manager.inspect(key, source_revision="16")

    def _fail_to_write_manifest(*args: object, **kwargs: object) -> None:
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(manager, "_write_manifest", _fail_to_write_manifest)

    with pytest.raises(OSError, match="simulated manifest write failure"):
        manager.build_ivf_pq(key, _source, source_revision="17", spec=spec, overwrite=True)

    after = manager.inspect(key, source_revision="16")
    assert after.state == "current"
    assert after.artifact.index_path == before.artifact.index_path
    assert after.artifact.manifest_path == before.artifact.manifest_path


def test_index_manager_uses_a_deterministic_training_sample_from_the_full_stream(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    key = IndexKey(
        database_label="test-database",
        embedding_type_id=3,
        layer_index=0,
        metric="l2",
        dimension=2,
    )
    manager = IndexManager(tmp_path)
    sequence_ids = numpy.arange(1, 101, dtype=numpy.int64)
    vectors = numpy.column_stack((sequence_ids, sequence_ids * 10)).astype(numpy.float32)

    def _source(batch_size: int) -> Iterable[IndexVectorBatch]:
        return [
            IndexVectorBatch(sequence_ids=sequence_ids[start : start + batch_size], vectors=vectors[start : start + batch_size])
            for start in range(0, len(sequence_ids), batch_size)
        ]

    sample_by_sevens = manager._collect_training_sample(_source(7), key=key, sample_size=10, seed=20260919)
    sample_by_thirteens = manager._collect_training_sample(_source(13), key=key, sample_size=10, seed=20260919)

    assert sample_by_sevens.tolist() == sample_by_thirteens.tolist()
    assert int(sample_by_sevens[:, 0].max()) > 10


def test_index_manager_builds_and_streams_a_portable_exact_store(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    key = IndexKey(
        database_label="test-database",
        embedding_type_id=3,
        layer_index=0,
        metric="cosine",
        dimension=2,
    )
    manager = IndexManager(tmp_path)
    vectors = numpy.asarray([[1.0, 0.5], [0.25, 2.0], [1.5, 1.25]], dtype=numpy.float32)

    def _source() -> Iterable[ExactVectorBatch]:
        return [
            ExactVectorBatch(
                sequence_ids=[11, 12],
                protein_ids=["P1", "P2"],
                vectors=vectors[:2],
            ),
            ExactVectorBatch(
                sequence_ids=[13],
                protein_ids=["P3"],
                vectors=vectors[2:],
            ),
        ]

    manifest = manager.build_exact_store(key, _source, source_revision="3")
    inspection = manager.load_exact_store(
        database_label="test-database",
        embedding_type_id=3,
        layer_index=0,
        source_revision="3",
    )
    batches = list(manager.iter_exact_store_batches(inspection, batch_size=2))

    assert manifest.vector_count == 3
    assert manifest.source_read_seconds is not None
    assert manifest.source_read_seconds >= 0.0
    assert manifest.vector_write_seconds is not None
    assert manifest.vector_write_seconds >= 0.0
    assert manifest.metadata_write_seconds is not None
    assert manifest.metadata_write_seconds >= 0.0
    assert inspection.manifest is not None
    assert inspection.manifest.vector_write_seconds == pytest.approx(manifest.vector_write_seconds)
    assert inspection.artifact.vectors_path.stat().st_size == 3 * 2 * 2
    assert [protein_ids for protein_ids, _ in batches] == [["P1", "P2"], ["P3"]]
    assert numpy.vstack([batch_vectors for _, batch_vectors in batches]) == pytest.approx(vectors, abs=0.001)


def test_index_manager_raises_value_error_when_search_nprobe_is_not_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="search_nprobe"):
        IndexManager(tmp_path, search_nprobe=0)
    with pytest.raises(ValueError, match="training_sample_seed"):
        IndexBuildSpec(nlist=1, training_sample_seed=-1)
