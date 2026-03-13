from __future__ import annotations

from pathlib import Path

import pytest

from CBBIO.Taxonomy import (
    TaxonCountsNotPreparedError,
    TaxonNotFoundError,
    TaxonomyError,
    load_taxonomy,
    read_taxonomy_annotations_tsv,
)


NODES_CONTENT = """1\t|\t1\t|\tno rank\t|\n2\t|\t1\t|\tsuperkingdom\t|\n1224\t|\t2\t|\tphylum\t|\n562\t|\t1224\t|\tspecies\t|\n83333\t|\t562\t|\tstrain\t|\n2759\t|\t1\t|\tsuperkingdom\t|\n9606\t|\t2759\t|\tspecies\t|\n10090\t|\t2759\t|\tspecies\t|\n4000\t|\t2\t|\torder\t|\n5000\t|\t4000\t|\tfamily\t|\n5001\t|\t5000\t|\tgenus\t|\n5002\t|\t5001\t|\tspecies\t|\n5003\t|\t5000\t|\tgenus\t|\n5004\t|\t5003\t|\tspecies\t|\n"""


NAMES_CONTENT = """1\t|\troot\t|\t\t|\tscientific name\t|\n2\t|\tBacteria\t|\t\t|\tscientific name\t|\n1224\t|\tProteobacteria\t|\t\t|\tscientific name\t|\n562\t|\tEscherichia coli\t|\t\t|\tscientific name\t|\n83333\t|\tE. coli K-12\t|\t\t|\tscientific name\t|\n83333\t|\tK12\t|\t\t|\tsynonym\t|\n2759\t|\tEukaryota\t|\t\t|\tscientific name\t|\n9606\t|\tHomo sapiens\t|\t\t|\tscientific name\t|\n10090\t|\tMus musculus\t|\t\t|\tscientific name\t|\n4000\t|\tEnterobacterales\t|\t\t|\tscientific name\t|\n5000\t|\tEnterobacteriaceae\t|\t\t|\tscientific name\t|\n5001\t|\tEscherichia\t|\t\t|\tscientific name\t|\n5002\t|\tEscherichia alpha\t|\t\t|\tscientific name\t|\n5003\t|\tShigella\t|\t\t|\tscientific name\t|\n5004\t|\tShigella beta\t|\t\t|\tscientific name\t|\n"""


@pytest.fixture()
def taxdump_dir(tmp_path: Path) -> Path:
    root = tmp_path / "taxdump"
    root.mkdir(parents=True, exist_ok=True)
    (root / "nodes.dmp").write_text(NODES_CONTENT, encoding="utf-8")
    (root / "names.dmp").write_text(NAMES_CONTENT, encoding="utf-8")
    return root


def test_taxonomy_navigation_and_lookup(taxdump_dir: Path) -> None:
    tax = load_taxonomy(str(taxdump_dir))
    assert tax.has_taxon("83333")
    row = tax.taxon("83333")
    assert row["id"] == "83333"
    assert row["name"] == "E. coli K-12"
    assert row["rank"] == "strain"

    assert tax.lineage("83333") == ["1", "2", "1224", "562", "83333"]
    assert tax.ancestors("83333") == ["1", "2", "1224", "562"]
    assert tax.descendants("2") == ["1224", "4000", "5000", "562", "5001", "5003", "83333", "5002", "5004"]


def test_taxonomy_lca_and_branch_length(taxdump_dir: Path) -> None:
    tax = load_taxonomy(str(taxdump_dir))
    assert tax.common_ancestors("83333", "562") == ["1", "2", "1224", "562"]
    assert tax.lowest_common_ancestor("83333", "562") == "562"
    assert tax.lowest_common_ancestor("83333", "9606") == "1"
    assert tax.minimal_branch_length("83333", "562") == 1
    assert tax.minimal_branch_length("83333", "9606") == 6
    assert tax.normalized_lca_depth("83333", "562") == 0.75
    assert tax.normalized_lca_depth("83333", "9606") == 0.0
    assert tax.wu_palmer_similarity("83333", "562") == pytest.approx(6.0 / 7.0)
    assert tax.wu_palmer_similarity("83333", "9606") == 0.0
    assert tax.wu_palmer_similarity("9606", "10090") == 0.5
    assert tax.lowest_common_ancestor_rank("5002", "5004") == "family"
    clade = tax.lowest_common_ancestor_clade("5002", "5004")
    assert clade is not None
    assert clade["name"] == "Enterobacteriaceae"
    assert clade["rank"] == "family"
    assert tax.shares_clade_at_rank("5002", "5004", "genus") is False
    assert tax.shares_clade_at_rank("5002", "5004", "family") is True
    assert tax.shares_clade_at_rank("5002", "5004", "order") is True


def test_taxonomy_errors(taxdump_dir: Path) -> None:
    tax = load_taxonomy(str(taxdump_dir))
    with pytest.raises(TaxonNotFoundError):
        tax.taxon("999999")

    with pytest.raises(TaxonCountsNotPreparedError):
        tax.information_content("83333", mode="observed")

    with pytest.raises(TaxonomyError):
        tax.prepare_taxon_counts({}, mode="invalid")


def test_taxonomy_observed_ic(taxdump_dir: Path) -> None:
    tax = load_taxonomy(str(taxdump_dir))
    ann = {
        "Q1": {"83333"},
        "Q2": {"562"},
        "Q3": {"9606"},
    }
    tax.prepare_taxon_counts(ann, mode="observed")

    ic_root = tax.information_content("1", mode="observed")
    ic_parent = tax.information_content("562", mode="observed")
    ic_leaf = tax.information_content("83333", mode="observed")

    assert ic_root == 0.0
    assert ic_leaf > ic_parent
    assert ic_parent >= 0.0


def test_taxonomy_subtree_ic(taxdump_dir: Path) -> None:
    tax = load_taxonomy(str(taxdump_dir))
    tax.prepare_taxon_counts({}, mode="subtree")

    ic_root = tax.information_content("1", mode="subtree")
    ic_leaf = tax.information_content("83333", mode="subtree")
    ic_mid = tax.information_content("2", mode="subtree")

    assert ic_root <= ic_mid
    assert ic_leaf > ic_mid


def test_taxonomy_name_helpers(taxdump_dir: Path) -> None:
    tax = load_taxonomy(str(taxdump_dir))
    names = tax.taxon_names(["10090", "9606"])
    assert names == ["Homo sapiens", "Mus musculus"]
    assert tax.format_taxon_names(["9606", "10090"]) == "Homo sapiens; Mus musculus"


def test_read_taxonomy_annotations_tsv(tmp_path: Path) -> None:
    path = tmp_path / "tax.tsv"
    path.write_text(
        "# comment\nQ1\t9606\nQ1\t9606\nQ2\t10090\nBAD\n",
        encoding="utf-8",
    )

    rows = read_taxonomy_annotations_tsv(str(path))
    assert rows["Q1"] == {"9606"}
    assert rows["Q2"] == {"10090"}


def test_taxonomy_smoke_deterministic_outputs(taxdump_dir: Path) -> None:
    tax = load_taxonomy(str(taxdump_dir))
    ann = {"A": {"83333"}, "B": {"9606"}, "C": {"10090"}}

    tax.prepare_taxon_counts(ann, mode="observed")
    first = (
        tax.lowest_common_ancestor("9606", "10090"),
        round(tax.information_content("2759", mode="observed"), 6),
        round(tax.information_content("83333", mode="observed"), 6),
    )

    tax.prepare_taxon_counts(ann, mode="observed")
    second = (
        tax.lowest_common_ancestor("9606", "10090"),
        round(tax.information_content("2759", mode="observed"), 6),
        round(tax.information_content("83333", mode="observed"), 6),
    )

    assert first == second
