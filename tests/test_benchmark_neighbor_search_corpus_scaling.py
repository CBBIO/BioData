from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_neighbor_search_corpus_scaling.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_neighbor_search_corpus_scaling", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


class _Client:
    def __init__(self) -> None:
        self.sql = ""
        self.params: tuple[Any, ...] = ()
        self.executed: list[str] = []

    def execute(self, sql: str) -> None:
        self.executed.append(sql)

    def query_all(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        self.sql = sql
        self.params = params
        return [
            {"query_id": "Q1", "protein_id": "P1", "layer_index": 33, "distance": 0.1},
            {"query_id": "Q2", "protein_id": None, "layer_index": None, "distance": None},
        ]


def _args(
    *,
    corpus_sizes: list[int] | None = None,
    query_counts: list[int] | None = None,
    query_pool_size: int = 10_000,
) -> argparse.Namespace:
    return argparse.Namespace(
        repeats=5,
        k=10,
        corpus_sizes=corpus_sizes or [10_000, 100_000],
        query_counts=query_counts or [100, 1_000],
        query_pool_size=query_pool_size,
        ann_ef_search=200,
        ann_candidate_pool=None,
    )


def test_validate_args_accepts_unique_ascending_corpus_sizes() -> None:
    _MODULE._validate_args(_args())


def test_validate_args_rejects_unsorted_corpus_sizes() -> None:
    with pytest.raises(SystemExit, match="unique and sorted"):
        _MODULE._validate_args(_args(corpus_sizes=[100_000, 10_000]))


def test_validate_args_rejects_unsorted_query_batch_sizes() -> None:
    with pytest.raises(SystemExit, match="query-counts must be unique and sorted"):
        _MODULE._validate_args(_args(query_counts=[1_000, 100]))


def test_resolve_dsn_builds_connection_from_postgres_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "192.168.49.224")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "benchmark-user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "benchmark-password")
    monkeypatch.setenv("POSTGRES_DB", "BioData")

    dsn = _MODULE._resolve_dsn(None)

    assert dsn == "postgresql://benchmark-user:benchmark-password@192.168.49.224:5432/BioData"


def test_resolve_dsn_prefers_explicit_value_over_postgres_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "192.168.49.224")

    dsn = _MODULE._resolve_dsn("postgresql://override:password@other-host:5432/other-db")

    assert dsn == "postgresql://override:password@other-host:5432/other-db"


def test_build_query_samples_uses_every_requested_batch_size() -> None:
    numpy = pytest.importorskip("numpy")

    samples = _MODULE._build_query_samples(
        ["P1", "P2", "P3", "P4"],
        numpy.asarray([[1.0], [2.0], [3.0], [4.0]]),
        query_counts=[1, 3],
        repeats=2,
        seed="test-seed",
    )

    assert set(samples) == {1, 3}
    assert len(samples[1]) == 2
    assert len(samples[3]) == 2
    assert all(len(query_ids) == 3 and query_vectors.shape == (3, 1) for query_ids, query_vectors in samples[3])


def test_search_pgvector_uses_temporary_candidate_corpus() -> None:
    numpy = pytest.importorskip("numpy")
    client = _Client()

    grouped = _MODULE._search_pgvector(
        client,
        query_ids=["Q1", "Q2"],
        query_vectors=numpy.asarray([[0.1, 0.2], [0.3, 0.4]]),
        metric="cosine",
        k=10,
    )

    assert [neighbor.protein_id for neighbor in grouped["Q1"]] == ["P1"]
    assert grouped["Q2"] == []
    assert "benchmark_neighbor_candidates" in client.sql
    assert "LEFT JOIN LATERAL" in client.sql
    assert "<=>" in client.sql
    assert client.params[-1] == 10


def test_search_pgvector_ann_reranks_a_candidate_pool() -> None:
    numpy = pytest.importorskip("numpy")
    client = _Client()

    _MODULE._search_pgvector(
        client,
        query_ids=["Q1"],
        query_vectors=numpy.asarray([[0.1, 0.2]]),
        metric="cosine",
        k=10,
        use_ann=True,
        ann_ef_search=300,
        ann_candidate_pool=500,
    )

    assert "ann_candidates" in client.sql
    assert client.params[-2:] == (500, 10)
    assert client.executed == ["SET hnsw.ef_search = 300;"]
