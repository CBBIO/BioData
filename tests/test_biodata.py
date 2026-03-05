from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import pytest

import CBBIO.BioData as bd


@dataclass
class _Response:
    one: Optional[Any] = None
    all: Optional[List[Any]] = None
    description: Optional[Sequence[Tuple[str, ...]]] = None


class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn
        self._response = _Response()
        self.description = None

    def execute(self, sql: str, params: Any = ()) -> None:
        self._conn.executed.append((sql, params))
        if self._conn.responses:
            self._response = self._conn.responses.pop(0)
        else:
            self._response = _Response()
        self.description = self._response.description

    def fetchone(self) -> Any:
        return self._response.one

    def fetchall(self) -> List[Any]:
        return self._response.all or []

    def close(self) -> None:
        return None


class _FakeConn:
    def __init__(self, responses: List[_Response]) -> None:
        self.responses = list(responses)
        self.executed: List[Tuple[str, Any]] = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def close(self) -> None:
        self.closed = True


def _client_with_fake_conn(
    responses: List[_Response],
    *,
    autocommit: bool = True,
) -> Tuple[bd.BioDataClient, _FakeConn]:
    client = bd.BioDataClient(autocommit=autocommit)
    conn = _FakeConn(responses)
    client._conn = conn
    return client, conn


def test_load_config_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIODATA_DB_HOST", "db.example")
    monkeypatch.setenv("BIODATA_DB_PORT", "6543")
    monkeypatch.setenv("BIODATA_DB_NAME", "BioDataProd")
    monkeypatch.setenv("BIODATA_DB_USER", "alice")
    monkeypatch.setenv("BIODATA_DB_PASSWORD", "secret")
    monkeypatch.setenv("BIODATA_AUTOCOMMIT", "false")
    monkeypatch.setenv("BIODATA_REGISTER_HALFVEC", "true")
    monkeypatch.setenv("BIODATA_DEFAULT_METRIC", "cosine")
    monkeypatch.setenv("BIODATA_DEFAULT_K", "25")

    cfg = bd.load_config(config_path="/definitely/missing.yaml")
    assert cfg["database"]["host"] == "db.example"
    assert cfg["database"]["port"] == 6543
    assert cfg["database"]["name"] == "BioDataProd"
    assert cfg["database"]["user"] == "alice"
    assert cfg["database"]["password"] == "secret"
    assert cfg["client"]["autocommit"] is False
    assert cfg["client"]["register_halfvec"] is True
    assert cfg["search"]["default_metric"] == "cosine"
    assert cfg["search"]["default_k"] == 25


def test_find_nearest_neighbors_uses_metric_and_params() -> None:
    responses = [_Response(all=[("P1", 0, 0.1), ("P2", 0, 0.2)])]
    client, conn = _client_with_fake_conn(responses)

    neighbors = client.find_nearest_neighbors(
        [0.1, 0.2],
        embedding_type_id=1,
        layer_index=0,
        k=2,
        metric="cosine",
    )

    assert [n.protein_id for n in neighbors] == ["P1", "P2"]
    sql, params = conn.executed[0]
    assert "<=>" in sql
    assert "LIMIT %s" in sql
    assert params[0] == [0.1, 0.2]
    assert params[-1] == 2


def test_find_nearest_neighbors_adds_exclusion_clause() -> None:
    responses = [_Response(all=[("P2", 0, 0.2)])]
    client, conn = _client_with_fake_conn(responses)

    client.find_nearest_neighbors(
        [0.1, 0.2],
        embedding_type_id=1,
        layer_index=0,
        k=5,
        exclude_protein_ids=["P1", "P3"],
    )

    sql, params = conn.executed[0]
    assert "p.id <> ALL(%s)" in sql
    assert ["P1", "P3"] in params


def test_find_nearest_neighbors_rejects_invalid_k() -> None:
    client, _ = _client_with_fake_conn([])
    with pytest.raises(bd.BioDataError):
        client.find_nearest_neighbors([0.1], embedding_type_id=1, k=0)


def test_fetch_go_annotations_groups_rows() -> None:
    responses = [
        _Response(
            all=[
                ("P1", "GO:0001", "mf", "desc1", "EXP"),
                ("P1", "GO:0002", "bp", "desc2", "IDA"),
                ("P2", "GO:0003", "cc", "desc3", "IEA"),
            ]
        )
    ]
    client, _ = _client_with_fake_conn(responses)
    grouped = client.fetch_go_annotations(["P1", "P2"])

    assert sorted(grouped.keys()) == ["P1", "P2"]
    assert len(grouped["P1"]) == 2
    assert grouped["P1"][0].go_id == "GO:0001"
    assert grouped["P2"][0].evidence_code == "IEA"


def test_neighbors_with_go_raises_when_query_embedding_missing() -> None:
    responses = [_Response(one=None)]
    client, _ = _client_with_fake_conn(responses)

    with pytest.raises(bd.NotFoundError):
        client.neighbors_with_go(
            query_uniprot_id="P12345",
            embedding_type_id=1,
        )


def test_health_check_reports_missing_tables() -> None:
    responses = [
        _Response(one=("BioData",), description=[("current_database",)]),
        _Response(one=("16.3",), description=[("server_version",)]),
        _Response(one=(True,), description=[("exists",)]),
        _Response(all=[{"table_name": "protein"}, {"table_name": "sequence"}]),
    ]
    client, _ = _client_with_fake_conn(responses)

    status = client.health_check()

    assert status["database"] == "BioData"
    assert status["pgvector_installed"] is True
    assert "sequence_embeddings" in status["missing_tables"]


def test_transaction_commits_when_autocommit_disabled() -> None:
    client, conn = _client_with_fake_conn([], autocommit=False)
    with client.transaction():
        pass
    assert conn.committed is True
    assert conn.rolled_back is False


def test_transaction_rolls_back_on_error() -> None:
    client, conn = _client_with_fake_conn([], autocommit=False)
    with pytest.raises(RuntimeError):
        with client.transaction():
            raise RuntimeError("boom")
    assert conn.rolled_back is True
