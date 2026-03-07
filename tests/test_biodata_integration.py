from __future__ import annotations

import os
from pathlib import Path

import pytest

import CBBIO.BioData as bd


_RUN_HINT = (
    "Integration tests are skipped. To run them:\n"
    "1) Copy config_test.yaml.example to config_test.yaml and edit credentials.\n"
    "2) Ensure PostgreSQL is running and reachable.\n"
    "3) Ensure BioData schema is loaded (schema.sql).\n"
    "4) Run: poetry run pytest -q tests/test_biodata_integration.py"
)


@pytest.fixture(scope="module")
def integration_client() -> bd.BioDataClient:
    config_path = Path(os.getenv("BIODATA_TEST_CONFIG", "config_test.yaml"))
    if not config_path.exists():
        pytest.skip(f"{_RUN_HINT}\nMissing config file: {config_path}")

    client = bd.BioDataClient(config_path=config_path)
    try:
        client.connect()
    except Exception as exc:  # pragma: no cover - only hit in unavailable envs
        pytest.skip(f"{_RUN_HINT}\nCould not connect with {config_path}: {exc!r}")

    try:
        status = client.health_check(check_extension=True, check_required_tables=True)
        missing_tables = status.get("missing_tables", [])
        if missing_tables:
            pytest.skip(f"{_RUN_HINT}\nMissing required tables: {missing_tables}")
        yield client
    finally:
        client.close()


def test_integration_query_helpers(integration_client: bd.BioDataClient) -> None:
    row = integration_client.query_one("SELECT 1 AS value;")
    rows = integration_client.query_all("SELECT 1 AS value UNION ALL SELECT 2 AS value ORDER BY value;")
    scalar = integration_client.scalar("SELECT 3 AS value;")

    assert row == {"value": 1}
    assert rows == [{"value": 1}, {"value": 2}]
    assert scalar == 3


def test_integration_embedding_types_and_layers(integration_client: bd.BioDataClient) -> None:
    embedding_types = integration_client.list_embedding_types()
    assert isinstance(embedding_types, list)

    if not embedding_types:
        pytest.skip("No rows in sequence_embedding_type; load dataset to test embedding/layer methods.")

    layers = integration_client.list_available_layers(embedding_types[0].id)
    assert isinstance(layers, list)
    assert all(isinstance(layer, int) for layer in layers)


def test_integration_protein_access_and_sequences(integration_client: bd.BioDataClient) -> None:
    row = integration_client.query_one("SELECT id FROM protein LIMIT 1;")
    if row is None:
        pytest.skip("No protein rows found; load dataset backup to run protein integration tests.")

    protein_id = str(row["id"])
    protein = integration_client.get_protein(protein_id)
    assert protein is not None
    assert protein["id"] == protein_id

    seq = integration_client.get_protein_sequence(protein_id)
    species = integration_client.get_protein_species(protein_id)
    tax = integration_client.get_protein_taxonomy_id(protein_id)
    batch_seqs = integration_client.get_protein_sequences([protein_id])

    assert seq is None or isinstance(seq, str)
    assert species is None or isinstance(species, str)
    assert tax is None or isinstance(tax, str)
    assert protein_id in batch_seqs or batch_seqs == {}


def test_integration_go_fetch_methods(integration_client: bd.BioDataClient) -> None:
    row = integration_client.query_one("SELECT protein_id FROM protein_go_term_annotation LIMIT 1;")
    if row is None:
        pytest.skip("No GO annotation rows found; load dataset backup to run GO integration tests.")

    protein_id = str(row["protein_id"])
    ann = integration_client.fetch_go_annotations([protein_id])
    grouped_ids = integration_client.fetch_protein_go_ids([protein_id])

    assert protein_id in ann
    assert protein_id in grouped_ids
    assert len(grouped_ids[protein_id]) >= 1


def test_integration_embedding_neighbor_search(integration_client: bd.BioDataClient) -> None:
    row = integration_client.query_one(
        """
        SELECT p.id AS protein_id, se.embedding_type_id, se.layer_index
        FROM protein p
        JOIN sequence_embeddings se ON se.sequence_id = p.sequence_id
        LIMIT 1;
        """
    )
    if row is None:
        pytest.skip("No sequence embeddings found; load dataset backup to run neighbor integration tests.")

    protein_id = str(row["protein_id"])
    embedding_type_id = int(row["embedding_type_id"])
    layer_index = int(row["layer_index"])

    embedding = integration_client.get_protein_embedding(
        protein_id,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
    )
    assert embedding is not None

    neighbors = integration_client.find_nearest_neighbors(
        embedding,
        embedding_type_id=embedding_type_id,
        layer_index=layer_index,
        k=3,
        metric="cosine",
        exclude_protein_ids=[protein_id],
        use_ann=False,
    )
    assert isinstance(neighbors, list)
    assert len(neighbors) <= 3

