from __future__ import annotations

from dataclasses import dataclass
import sys
import types
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


def test_build_dsn_uses_explicit_values() -> None:
    dsn = bd.build_dsn(
        user="alice",
        password="secret",
        host="db.example",
        port=6543,
        database="BioDataProd",
    )
    assert dsn == "postgresql://alice:secret@db.example:6543/BioDataProd"


def test_is_connected_property() -> None:
    client = bd.BioDataClient()
    assert client.is_connected is False
    client._conn = _FakeConn([])
    assert client.is_connected is True


def test_query_helpers_return_expected_shapes() -> None:
    responses = [
        _Response(all=[("P1", "AAAA"), ("P2", "BBBB")], description=[("id",), ("sequence",)]),
        _Response(one=("P3",), description=[("id",)]),
        _Response(one=(7,), description=[("count",)]),
    ]
    client, conn = _client_with_fake_conn(responses)

    rows = client.query_all("SELECT id, sequence FROM x;")
    one = client.query_one("SELECT id FROM y;")
    value = client.scalar("SELECT count(*) FROM z;")

    assert rows == [{"id": "P1", "sequence": "AAAA"}, {"id": "P2", "sequence": "BBBB"}]
    assert one == {"id": "P3"}
    assert value == 7
    assert len(conn.executed) == 3


def test_count_sequence_embeddings_and_list_embedding_types() -> None:
    responses = [
        _Response(one={"count": 12}),
        _Response(
            all=[
                {"id": 1, "name": "esm2_layer0", "model_name": "esm2", "task_name": "protein", "description": "d1"},
                {"id": 2, "name": "prot_t5", "model_name": None, "task_name": None, "description": None},
            ]
        ),
    ]
    client, _ = _client_with_fake_conn(responses)

    assert client.count_sequence_embeddings() == 12
    emb_types = client.list_embedding_types()
    assert [e.id for e in emb_types] == [1, 2]
    assert emb_types[0].name == "esm2_layer0"
    assert emb_types[1].model_name is None


def test_getters_for_protein_accession_go_and_structure_tables() -> None:
    responses = [
        _Response(one={"id": "P1", "description": "protein"}),
        _Response(all=[{"code": "Q1", "is_primary": True, "tag": "SwissProt"}]),
        _Response(all=[{"go_id": "GO:0001", "category": "mf", "description": "d", "evidence_code": "EXP"}]),
        _Response(all=[{"id": "AF-P1-F1", "method": "AF2"}]),
        _Response(all=[{"id": 10, "name": "A", "sequence_id": 1, "accession_code": "Q1"}]),
        _Response(all=[{"id": 100, "model_id": "1", "file_path": "/tmp/s.cif", "structure_id": "AF-P1-F1"}]),
        _Response(all=[{"id": 900, "state_id": 100, "embedding": "xyz"}]),
    ]
    client, conn = _client_with_fake_conn(responses)

    assert client.get_protein("P1") == {"id": "P1", "description": "protein"}
    assert client.list_accessions_for_protein("P1")[0]["code"] == "Q1"
    assert client.get_protein_go_annotations("P1")[0]["go_id"] == "GO:0001"
    assert client.get_protein_structures("P1")[0]["id"] == "AF-P1-F1"
    assert client.get_structure_chains("AF-P1-F1")[0]["id"] == 10
    assert client.get_chain_states(10)[0]["id"] == 100
    assert client.get_state_3di_embeddings(100)[0]["id"] == 900
    assert len(conn.executed) == 7


def test_get_embedding_type_by_name_and_list_available_layers() -> None:
    responses = [
        _Response(one={"id": 5, "name": "esm2_layer0", "model_name": "esm2", "task_name": "protein", "description": None}),
        _Response(all=[{"layer_index": 0}, {"layer_index": 12}]),
    ]
    client, conn = _client_with_fake_conn(responses)

    emb_type = client.get_embedding_type_by_name("esm2_layer0")
    layers = client.list_available_layers(5)

    assert emb_type is not None
    assert emb_type.id == 5
    assert layers == [0, 12]
    assert conn.executed[0][1] == ("esm2_layer0",)
    assert conn.executed[1][1] == (5,)


def test_get_protein_embedding_variants() -> None:
    responses = [
        _Response(one=([0.1, 0.2],)),
        _Response(one=([0.3, 0.4],)),
        _Response(one=None),
    ]
    client, _ = _client_with_fake_conn(responses)

    raw = client.get_protein_embedding("P1", embedding_type_id=1, layer_index=0)
    assert raw == [0.1, 0.2]

    np_vec = client.get_protein_embedding("P1", embedding_type_id=1, layer_index=0, as_numpy=True)
    assert np_vec.shape == (2,)
    assert str(np_vec.dtype) == "float32"

    missing = client.get_protein_embedding("P1", embedding_type_id=1, layer_index=0)
    assert missing is None


def test_get_protein_embeddings_variants() -> None:
    responses = [
        _Response(all=[("P1", [0.1, 0.2]), ("P2", [0.3, 0.4])]),
        _Response(all=[("P1", [0.5, 0.6])]),
    ]
    client, conn = _client_with_fake_conn(responses)

    raw = client.get_protein_embeddings(["P1", "P2"], embedding_type_id=9, layer_index=1)
    as_np = client.get_protein_embeddings(["P1"], embedding_type_id=9, layer_index=1, as_numpy=True)
    empty = client.get_protein_embeddings([], embedding_type_id=9, layer_index=1)

    assert raw["P1"] == [0.1, 0.2]
    assert as_np["P1"].shape == (2,)
    assert str(as_np["P1"].dtype) == "float32"
    assert empty == {}
    assert len(conn.executed) == 2


def test_find_nearest_neighbors_for_proteins_groups_rows_and_respects_include_query_flag() -> None:
    responses = [
        _Response(
            all=[
                ("Q1", "N1", 0, 0.1),
                ("Q1", "N2", 0, 0.2),
                ("Q2", None, None, None),
            ]
        )
    ]
    client, conn = _client_with_fake_conn(responses)

    grouped = client.find_nearest_neighbors_for_proteins(
        ["Q1", "Q2"],
        embedding_type_id=3,
        layer_index=0,
        k=2,
        metric="cosine",
        include_query=False,
    )

    assert [n.protein_id for n in grouped["Q1"]] == ["N1", "N2"]
    assert grouped["Q2"] == []
    sql, params = conn.executed[0]
    assert "<=>" in sql
    assert params == (["Q1", "Q2"], 3, 0, 3, 0, False, 2)


def test_find_nearest_neighbors_for_proteins_rejects_invalid_k() -> None:
    client, _ = _client_with_fake_conn([])
    with pytest.raises(bd.BioDataError):
        client.find_nearest_neighbors_for_proteins(["Q1"], embedding_type_id=1, k=0)


def test_find_nearest_neighbors_for_proteins_empty_input_short_circuit() -> None:
    client, conn = _client_with_fake_conn([])
    assert client.find_nearest_neighbors_for_proteins([], embedding_type_id=1) == {}
    assert conn.executed == []


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


def test_find_nearest_neighbors_use_ann_query_shape() -> None:
    responses = [_Response(), _Response(all=[("P2", 0, 0.2)])]
    client, conn = _client_with_fake_conn(responses)

    client.find_nearest_neighbors(
        [0.1, 0.2, 0.3],
        embedding_type_id=3,
        layer_index=0,
        k=1,
        metric="cosine",
        exclude_protein_ids=["P1"],
        use_ann=True,
    )

    assert "SET hnsw.ef_search = 200;" in conn.executed[0][0]

    sql, params = conn.executed[1]
    assert "WITH ann_candidates AS" in sql
    assert "ORDER BY (se.embedding::halfvec(3)) <=> %s::halfvec" in sql
    assert "JOIN protein p ON p.sequence_id = c.sequence_id" in sql
    assert "GROUP BY protein_id" in sql
    assert "LIMIT %s;" in sql
    assert params[2] == [0.1, 0.2, 0.3]
    assert ["P1"] in params
    assert params[-1] == 1


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


def test_fetch_protein_go_ids_groups_rows() -> None:
    responses = [
        _Response(
            all=[
                ("P1", "GO:0001"),
                ("P1", "GO:0002"),
                ("P2", "GO:0003"),
            ]
        )
    ]
    client, conn = _client_with_fake_conn(responses)
    grouped = client.fetch_protein_go_ids(["P1", "P2"])

    assert grouped["P1"] == {"GO:0001", "GO:0002"}
    assert grouped["P2"] == {"GO:0003"}
    sql, params = conn.executed[0]
    assert "WHERE protein_id = ANY(%s)" in sql
    assert params == (["P1", "P2"],)


def test_fetch_protein_go_ids_all_when_none() -> None:
    responses = [_Response(all=[("P1", "GO:0001")])]
    client, conn = _client_with_fake_conn(responses)
    grouped = client.fetch_protein_go_ids()

    assert grouped == {"P1": {"GO:0001"}}
    sql, params = conn.executed[0]
    assert "WHERE protein_id = ANY(%s)" not in sql
    assert params == ()


def test_get_protein_by_accession_returns_joined_row() -> None:
    responses = [
        _Response(
            one={
                "id": "P12345",
                "sequence_id": 7,
                "description": "Example protein",
                "accession_code": "Q99999",
                "is_primary_accession": True,
            }
        )
    ]
    client, _ = _client_with_fake_conn(responses)
    row = client.get_protein_by_accession("Q99999")

    assert row is not None
    assert row["id"] == "P12345"
    assert row["accession_code"] == "Q99999"


def test_get_protein_sequence_returns_string() -> None:
    responses = [_Response(one={"sequence": "MPEPTIDE"})]
    client, _ = _client_with_fake_conn(responses)
    value = client.get_protein_sequence("P12345")
    assert value == "MPEPTIDE"


def test_get_protein_sequence_returns_none_if_missing() -> None:
    responses = [_Response(one=None)]
    client, _ = _client_with_fake_conn(responses)
    assert client.get_protein_sequence("P00000") is None


def test_get_protein_sequences_batch() -> None:
    responses = [
        _Response(
            all=[
                {"id": "P1", "sequence": "AAAA"},
                {"id": "P2", "sequence": "BBBB"},
            ]
        )
    ]
    client, conn = _client_with_fake_conn(responses)
    values = client.get_protein_sequences(["P1", "P2"])

    assert values == {"P1": "AAAA", "P2": "BBBB"}
    sql, params = conn.executed[0]
    assert "WHERE p.id = ANY(%s)" in sql
    assert params == (["P1", "P2"],)


def test_get_protein_sequences_empty_input() -> None:
    client, conn = _client_with_fake_conn([])
    values = client.get_protein_sequences([])
    assert values == {}
    assert conn.executed == []


def test_get_protein_species_returns_value() -> None:
    responses = [_Response(one={"organism": "Drosophila melanogaster"})]
    client, _ = _client_with_fake_conn(responses)
    value = client.get_protein_species("P12345")
    assert value == "Drosophila melanogaster"


def test_get_protein_taxonomy_id_returns_value() -> None:
    responses = [_Response(one={"taxonomy_id": "7227"})]
    client, _ = _client_with_fake_conn(responses)
    value = client.get_protein_taxonomy_id("P12345")
    assert value == "7227"


def test_get_protein_species_taxonomy_batch() -> None:
    responses = [
        _Response(
            all=[
                {"id": "P1", "organism": "Mus musculus", "taxonomy_id": "10090"},
                {"id": "P2", "organism": "Homo sapiens", "taxonomy_id": "9606"},
            ]
        )
    ]
    client, conn = _client_with_fake_conn(responses)
    values = client.get_protein_species_taxonomy(["P1", "P2"])

    assert values == {
        "P1": {"species": "Mus musculus", "taxonomy_id": "10090"},
        "P2": {"species": "Homo sapiens", "taxonomy_id": "9606"},
    }
    sql, params = conn.executed[0]
    assert "WHERE id = ANY(%s)" in sql
    assert params == (["P1", "P2"],)


def test_get_protein_context_returns_none_if_protein_missing() -> None:
    responses = [_Response(one=None)]
    client, _ = _client_with_fake_conn(responses)

    assert client.get_protein_context("P00000") is None


def test_get_protein_context_aggregates_related_tables() -> None:
    responses = [
        _Response(one={"id": "P12345", "sequence_id": 1}),
        _Response(all=[{"code": "Q99999", "is_primary": True}]),
        _Response(all=[{"go_id": "GO:0001", "category": "mf", "evidence_code": "EXP"}]),
        _Response(all=[{"id": "AF-P12345-F1", "method": "AF2"}]),
        _Response(all=[{"id": 10, "name": "A", "sequence_id": 1, "accession_code": "Q99999"}]),
        _Response(all=[{"id": 100, "model_id": "1", "file_path": "/tmp/state1.cif"}]),
    ]
    client, _ = _client_with_fake_conn(responses)
    context = client.get_protein_context("P12345")

    assert context is not None
    assert context["protein"]["id"] == "P12345"
    assert context["accessions"][0]["code"] == "Q99999"
    assert context["go_annotations"][0]["go_id"] == "GO:0001"
    assert context["structures"][0]["id"] == "AF-P12345-F1"
    assert context["chains_by_structure"]["AF-P12345-F1"][0]["id"] == 10
    assert context["states_by_chain"][10][0]["id"] == 100
    assert "structure_3di_by_state" not in context


def test_get_protein_context_includes_3di_when_enabled() -> None:
    responses = [
        _Response(one={"id": "P12345", "sequence_id": 1}),
        _Response(all=[]),
        _Response(all=[]),
        _Response(all=[{"id": "AF-P12345-F1", "method": "AF2"}]),
        _Response(all=[{"id": 10, "name": "A", "sequence_id": 1, "accession_code": None}]),
        _Response(all=[{"id": 100, "model_id": "1", "file_path": "/tmp/state1.cif"}]),
        _Response(all=[{"id": 900, "state_id": 100, "embedding": "ABCDEF"}]),
    ]
    client, _ = _client_with_fake_conn(responses)
    context = client.get_protein_context("P12345", include_3di=True)

    assert context is not None
    assert context["structure_3di_by_state"][100][0]["id"] == 900


def test_distance_to_protein_with_model_name_and_metric() -> None:
    responses = [
        _Response(one={"id": 1, "name": "esm2_layer0"}),
        _Response(one={"distance": 0.123}),
    ]
    client, conn = _client_with_fake_conn(responses)

    distance = client.distance_to_protein(
        [0.1, 0.2],
        protein_id="P12345",
        model="esm2_layer0",
        layer_index=0,
        metric="cosine",
    )

    assert distance == pytest.approx(0.123)
    sql, params = conn.executed[1]
    assert "<=>" in sql
    assert params == ([0.1, 0.2], "P12345", 1, 0)


def test_distance_between_proteins_with_model_id_and_metric() -> None:
    responses = [_Response(one={"distance": 0.55})]
    client, conn = _client_with_fake_conn(responses)

    distance = client.distance_between_proteins(
        protein_a_id="P11111",
        protein_b_id="P22222",
        model=3,
        layer_index=2,
        metric="inner_product",
    )

    assert distance == pytest.approx(0.55)
    sql, params = conn.executed[0]
    assert "<#>" in sql
    assert params == (3, 2, "P22222", 3, 2, "P11111")


def test_connect_register_vector_accepts_single_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Conn:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    conn = _Conn()
    calls: List[Any] = []

    psycopg_module = types.ModuleType("psycopg")
    psycopg_module.connect = lambda *args, **kwargs: conn  # type: ignore[attr-defined]

    pgvector_psycopg_module = types.ModuleType("pgvector.psycopg")

    def _register_vector_single(context: Any) -> None:
        calls.append(context)

    pgvector_psycopg_module.register_vector = _register_vector_single  # type: ignore[attr-defined]

    pgvector_module = types.ModuleType("pgvector")
    pgvector_module.psycopg = pgvector_psycopg_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "psycopg", psycopg_module)
    monkeypatch.setitem(sys.modules, "pgvector", pgvector_module)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg", pgvector_psycopg_module)

    client = bd.BioDataClient(dsn="postgresql://user:pass@localhost:5432/db")
    client.connect()
    client.close()

    assert calls == [conn]
    assert conn.closed is True


def test_distance_to_protein_raises_not_found_when_missing_embedding() -> None:
    responses = [
        _Response(one={"id": 1, "name": "esm2_layer0"}),
        _Response(one=None),
    ]
    client, _ = _client_with_fake_conn(responses)

    with pytest.raises(bd.NotFoundError):
        client.distance_to_protein(
            [0.1, 0.2],
            protein_id="P12345",
            model="esm2_layer0",
            layer_index=0,
        )


def test_distance_between_proteins_raises_when_model_name_missing() -> None:
    responses = [_Response(one=None)]
    client, _ = _client_with_fake_conn(responses)

    with pytest.raises(bd.NotFoundError):
        client.distance_between_proteins(
            protein_a_id="P11111",
            protein_b_id="P22222",
            model="missing_model",
            layer_index=0,
        )


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
