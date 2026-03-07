from __future__ import annotations

from pathlib import Path

import pytest

from CBBIO.GO import (
    GOCountsNotPreparedError,
    GOError,
    GOTermNotFoundError,
    load_go,
    read_annotations_tsv,
)


OBO_CONTENT = """format-version: 1.2

[Term]
id: GO:0000001
name: root process
namespace: biological_process

[Term]
id: GO:0000002
name: child one
namespace: biological_process
is_a: GO:0000001 ! root process

[Term]
id: GO:0000003
name: child two
namespace: biological_process
is_a: GO:0000001 ! root process

[Term]
id: GO:0000004
name: grandchild
namespace: biological_process
is_a: GO:0000002 ! child one
"""


@pytest.fixture()
def go_obo_path(tmp_path: Path) -> Path:
    path = tmp_path / "mini-go.obo"
    path.write_text(OBO_CONTENT, encoding="utf-8")
    return path


def test_go_ancestors_descendants(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    assert go.has_term("GO:0000004")
    assert "GO:0000001" in go.go_ids
    assert go.ancestors("GO:0000004") == ["GO:0000001", "GO:0000002"]
    assert go.descendants("GO:0000001") == ["GO:0000002", "GO:0000003", "GO:0000004"]


def test_go_term_not_found(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    with pytest.raises(GOTermNotFoundError):
        go.term("GO:9999999")


def test_go_common_ancestors(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    common = go.common_ancestors("GO:0000004", "GO:0000003")
    assert common == ["GO:0000001"]


def test_go_ic_requires_counts(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    with pytest.raises(GOCountsNotPreparedError):
        go.information_content("GO:0000004")


def test_go_ic_and_similarity(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    annots = {
        "P1": {"GO:0000002"},
        "P2": {"GO:0000003"},
        "P3": {"GO:0000004"},
    }
    go.prepare_term_counts(annots)

    ic = go.information_content("GO:0000004")
    resnik = go.semantic_similarity("GO:0000004", "GO:0000002", method="resnik")
    lin = go.semantic_similarity("GO:0000004", "GO:0000002", method="lin")
    schlicker = go.semantic_similarity("GO:0000004", "GO:0000002", method="schlicker")

    assert ic >= 0.0
    assert resnik >= 0.0
    assert lin >= 0.0
    assert schlicker >= 0.0


def test_go_similarity_unknown_method(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    go.prepare_term_counts({"P1": {"GO:0000002"}})
    with pytest.raises(GOError):
        go.semantic_similarity("GO:0000002", "GO:0000002", method="unknown")  # type: ignore[arg-type]


def test_go_group_similarity_bma(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    go.prepare_term_counts(
        {
            "P1": {"GO:0000002"},
            "P2": {"GO:0000003"},
            "P3": {"GO:0000004"},
        }
    )

    sim = go.group_similarity(
        {"GO:0000004", "GO:0000002"},
        {"GO:0000003"},
        method="lin",
    )
    assert sim is not None
    assert sim >= 0.0


def test_go_group_similarity_unknown_aggregate(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    go.prepare_term_counts({"P1": {"GO:0000002"}})
    with pytest.raises(GOError):
        go.group_similarity(
            {"GO:0000002"},
            {"GO:0000002"},
            method="lin",
            aggregate="unknown",
        )


def test_go_term_name_helpers(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    names = go.term_names(["GO:0000004", "GO:0000002"])
    assert names == ["child one", "grandchild"]
    formatted = go.format_term_names(["GO:0000004", "GO:0000002"])
    assert formatted == "child one; grandchild"


def test_split_annotations_by_category(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    assert go.category_for_term("GO:0000002") == "bp"
    split = go.split_annotations_by_category(
        {
            "P1": {"GO:0000002", "GO:9999999"},
            "P2": {"GO:0000004"},
        }
    )
    assert split["bp"]["P1"] == {"GO:0000002"}
    assert split["bp"]["P2"] == {"GO:0000004"}
    assert split["mf"] == {}
    assert split["cc"] == {}


def test_minimal_branch_length(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    assert go.minimal_branch_length("GO:0000004", "GO:0000002") == 1
    assert go.minimal_branch_length("GO:0000004", "GO:0000003") == 3


def test_minimal_branch_length_unknown_term(go_obo_path: Path) -> None:
    go = load_go(str(go_obo_path))
    with pytest.raises(GOTermNotFoundError):
        go.minimal_branch_length("GO:0000004", "GO:9999999")


def test_read_annotations_tsv(tmp_path: Path) -> None:
    path = tmp_path / "ann.tsv"
    path.write_text(
        "# comment\nP1\tGO:0000002\nP1\tGO:0000004\nP2\tGO:0000003\n",
        encoding="utf-8",
    )
    rows = read_annotations_tsv(str(path))
    assert rows["P1"] == {"GO:0000002", "GO:0000004"}
    assert rows["P2"] == {"GO:0000003"}
