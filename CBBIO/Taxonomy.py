"""Taxonomy ontology helpers for NCBI taxdump files.

This module is intentionally independent from database access code.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Collection, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


class TaxonomyError(Exception):
    """Base exception for ``CBBIO.Taxonomy``."""


class TaxonNotFoundError(TaxonomyError):
    """Raised when a requested taxonomy ID is not present."""


class TaxonCountsNotPreparedError(TaxonomyError):
    """Raised when IC is requested for a mode without prepared counts."""


class TaxonomyOntology:
    """Wrapper around an NCBI taxonomy graph with convenience helpers."""

    def __init__(
        self,
        taxdump_dir: str,
        *,
        names_priority: Sequence[str] = ("scientific name",),
        quiet: bool = True,
    ) -> None:
        self.taxdump_dir = str(taxdump_dir)
        self.names_priority = tuple(str(value) for value in names_priority)
        self.quiet = bool(quiet)

        nodes_path = Path(taxdump_dir) / "nodes.dmp"
        names_path = Path(taxdump_dir) / "names.dmp"
        if not nodes_path.exists():
            raise TaxonomyError(f"nodes.dmp not found in taxdump directory: {taxdump_dir}")
        if not names_path.exists():
            raise TaxonomyError(f"names.dmp not found in taxdump directory: {taxdump_dir}")

        self._parent_by_taxon, self._rank_by_taxon = _parse_nodes_dmp(nodes_path)
        self._name_by_taxon = _parse_names_dmp(names_path, names_priority=self.names_priority)
        self._children_by_taxon = _build_children_index(self._parent_by_taxon)
        self._depth_by_taxon = _compute_depths(self._parent_by_taxon)

        self._prepared_probabilities: Dict[str, Dict[str, float]] = {}

    @property
    def taxon_ids(self) -> Set[str]:
        """Return all taxonomy IDs present in the graph."""
        return set(self._parent_by_taxon.keys())

    def has_taxon(self, tax_id: str) -> bool:
        """Return whether a taxonomy ID exists in the graph."""
        return str(tax_id) in self._parent_by_taxon

    def taxon(self, tax_id: str) -> Dict[str, object]:
        """Return metadata for one taxonomy ID."""
        value = self._require_taxon(tax_id)
        return {
            "id": value,
            "name": self._name_by_taxon.get(value, value),
            "rank": self._rank_by_taxon.get(value, ""),
            "parent_id": self._parent_by_taxon[value],
            "depth": self._depth_by_taxon[value],
        }

    def normalize_taxon_id(self, tax_id: object) -> str:
        """Return a canonical taxonomy-ID string suitable for lookups."""
        return normalize_taxonomy_id(tax_id)

    def ancestors(self, tax_id: str, *, include_self: bool = False) -> List[str]:
        """Return ancestor taxonomy IDs for one taxon."""
        taxon = self._require_taxon(tax_id)
        lineage = self.lineage(taxon, include_self=True, include_root=True)
        if not include_self:
            lineage = [value for value in lineage if value != taxon]
        return lineage

    def descendants(self, tax_id: str, *, include_self: bool = False) -> List[str]:
        """Return descendant taxonomy IDs for one taxon."""
        start = self._require_taxon(tax_id)
        visited: Set[str] = set()
        stack: List[str] = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(self._children_by_taxon.get(current, []))

        if not include_self:
            visited.discard(start)
        return sorted(visited, key=lambda value: (self._depth_by_taxon.get(value, 0), value))

    def lineage(self, tax_id: str, *, include_self: bool = True, include_root: bool = True) -> List[str]:
        """Return the root-to-taxon lineage for one taxonomy ID."""
        taxon = self._require_taxon(tax_id)
        chain: List[str] = []
        visited: Set[str] = set()
        current = taxon
        while True:
            if current in visited:
                break
            visited.add(current)
            chain.append(current)
            parent = self._parent_by_taxon.get(current)
            if parent is None or parent == current:
                break
            current = parent

        chain.reverse()

        if not include_root and chain:
            root = chain[0]
            if self._parent_by_taxon.get(root) == root:
                chain = chain[1:]

        if not include_self and taxon in chain:
            chain = [value for value in chain if value != taxon]
        return chain

    def common_ancestors(self, tax_id_a: str, tax_id_b: str, *, include_terms: bool = True) -> List[str]:
        """Return common ancestor taxonomy IDs for two taxa."""
        lineage_a = self.lineage(tax_id_a, include_self=include_terms, include_root=True)
        lineage_b = self.lineage(tax_id_b, include_self=include_terms, include_root=True)
        common = set(lineage_a).intersection(set(lineage_b))
        return sorted(common, key=lambda value: (self._depth_by_taxon.get(value, 0), value))

    def lowest_common_ancestor(self, tax_id_a: str, tax_id_b: str) -> str | None:
        """Return the deepest shared ancestor for two taxa."""
        self._require_taxon(tax_id_a)
        self._require_taxon(tax_id_b)

        common = self.common_ancestors(tax_id_a, tax_id_b, include_terms=True)
        if not common:
            return None
        return max(common, key=lambda value: (self._depth_by_taxon.get(value, -1), value))

    def lowest_common_ancestor_rank(self, tax_id_a: str, tax_id_b: str) -> str | None:
        """Return the rank of the lowest common ancestor (e.g. ``family``)."""
        lca = self.lowest_common_ancestor(tax_id_a, tax_id_b)
        if lca is None:
            return None
        return self._rank_by_taxon.get(lca, "")

    def lowest_common_ancestor_clade(self, tax_id_a: str, tax_id_b: str) -> Dict[str, object] | None:
        """Return metadata for the lowest common ancestor clade."""
        lca = self.lowest_common_ancestor(tax_id_a, tax_id_b)
        if lca is None:
            return None
        return {
            "id": lca,
            "name": self._name_by_taxon.get(lca, lca),
            "rank": self._rank_by_taxon.get(lca, ""),
            "depth": self._depth_by_taxon.get(lca, 0),
        }

    def shares_clade_at_rank(self, tax_id_a: str, tax_id_b: str, rank: str) -> bool:
        """Return whether both taxa belong to the same clade at a given rank."""
        rank_value = str(rank).strip().lower()
        if not rank_value:
            raise TaxonomyError("rank must be a non-empty string.")

        ancestors_a = self.lineage(tax_id_a, include_self=True, include_root=True)
        ancestors_b = set(self.lineage(tax_id_b, include_self=True, include_root=True))
        for ancestor in reversed(ancestors_a):
            if self._rank_by_taxon.get(ancestor, "").strip().lower() != rank_value:
                continue
            return ancestor in ancestors_b
        return False

    def minimal_branch_length(self, tax_id_a: str, tax_id_b: str) -> int | None:
        """Return the shortest branch distance between two taxa."""
        taxon_a = self._require_taxon(tax_id_a)
        taxon_b = self._require_taxon(tax_id_b)
        lca = self.lowest_common_ancestor(taxon_a, taxon_b)
        if lca is None:
            return None
        depth_lca = self._depth_by_taxon[lca]
        return (self._depth_by_taxon[taxon_a] - depth_lca) + (self._depth_by_taxon[taxon_b] - depth_lca)

    def normalized_lca_depth(self, tax_id_a: str, tax_id_b: str) -> float | None:
        """Return depth(LCA) / max(depth(a), depth(b)).

        Values are in [0, 1] when both taxonomy IDs are in the same rooted tree.
        """
        taxon_a = self._require_taxon(tax_id_a)
        taxon_b = self._require_taxon(tax_id_b)
        lca = self.lowest_common_ancestor(taxon_a, taxon_b)
        if lca is None:
            return None

        depth_a = int(self._depth_by_taxon[taxon_a])
        depth_b = int(self._depth_by_taxon[taxon_b])
        denom = max(depth_a, depth_b)
        if denom == 0:
            return 1.0
        return float(self._depth_by_taxon[lca]) / float(denom)

    def wu_palmer_similarity(self, tax_id_a: str, tax_id_b: str) -> float | None:
        """Return Wu-Palmer similarity: 2*depth(LCA) / (depth(a) + depth(b))."""
        taxon_a = self._require_taxon(tax_id_a)
        taxon_b = self._require_taxon(tax_id_b)
        lca = self.lowest_common_ancestor(taxon_a, taxon_b)
        if lca is None:
            return None

        depth_a = int(self._depth_by_taxon[taxon_a])
        depth_b = int(self._depth_by_taxon[taxon_b])
        denom = depth_a + depth_b
        if denom == 0:
            return 1.0
        return (2.0 * float(self._depth_by_taxon[lca])) / float(denom)

    def lin_similarity(self, tax_id_a: str, tax_id_b: str, *, mode: str = "observed") -> float | None:
        """Return Lin similarity: 2*IC(LCA) / (IC(a) + IC(b)).

        Returns None if the LCA cannot be determined.
        Returns 1.0 if both IC values are zero (e.g. both are the root taxon).
        Requires prepare_taxon_counts(..., mode=mode) to have been called first.
        """
        taxon_a = self._require_taxon(tax_id_a)
        taxon_b = self._require_taxon(tax_id_b)
        lca = self.lowest_common_ancestor(taxon_a, taxon_b)
        if lca is None:
            return None
        ic_a = self.information_content(taxon_a, mode=mode)
        ic_b = self.information_content(taxon_b, mode=mode)
        ic_lca = self.information_content(lca, mode=mode)
        denom = ic_a + ic_b
        if denom == 0.0:
            return 1.0
        return (2.0 * ic_lca) / denom

    def prepare_taxon_counts(
        self,
        annotations: Mapping[str, Collection[str]],
        *,
        mode: str = "observed",
    ) -> None:
        """Prepare taxonomy probabilities used by IC-based metrics."""
        mode_value = _normalize_mode(mode)

        if mode_value in {"observed", "whole_db"}:
            source_annotations = annotations if mode_value == "observed" else _fetch_whole_db_taxonomy_annotations()
            entity_to_expanded_taxa: Dict[str, Set[str]] = {}
            for entity_id, raw_taxa in source_annotations.items():
                expanded: Set[str] = set()
                for raw_tax_id in raw_taxa:
                    tax_id = str(raw_tax_id)
                    if not self.has_taxon(tax_id):
                        continue
                    expanded.update(self.lineage(tax_id, include_self=True, include_root=True))
                if expanded:
                    entity_to_expanded_taxa[str(entity_id)] = expanded

            counts: Dict[str, int] = defaultdict(int)
            for expanded_taxa in entity_to_expanded_taxa.values():
                for tax_id in expanded_taxa:
                    counts[tax_id] += 1

            total_entities = max(len(entity_to_expanded_taxa), 1)
            probabilities = {tax_id: float(count) / float(total_entities) for tax_id, count in counts.items()}
            self._prepared_probabilities[mode_value] = probabilities
            return

        subtree_sizes = _compute_subtree_sizes(self._children_by_taxon)
        total_nodes = max(len(self._parent_by_taxon), 1)
        probabilities = {
            tax_id: float(subtree_sizes.get(tax_id, 1)) / float(total_nodes)
            for tax_id in self._parent_by_taxon.keys()
        }
        self._prepared_probabilities[mode_value] = probabilities

    def information_content(self, tax_id: str, *, mode: str = "observed") -> float:
        """Return information content for one taxonomy ID."""
        value = self._require_taxon(tax_id)
        mode_value = _normalize_mode(mode)
        probabilities = self._prepared_probabilities.get(mode_value)
        if probabilities is None:
            raise TaxonCountsNotPreparedError(
                "Call prepare_taxon_counts(..., mode='observed'|'whole_db'|'subtree') before IC operations."
            )

        probability = float(probabilities.get(value, 0.0))
        if probability <= 0.0:
            return 0.0
        return float(-math.log(probability))

    def taxon_names(self, tax_ids: Collection[str], *, sort: bool = True) -> List[str]:
        """Return names for the provided taxonomy IDs."""
        names = [str(self.taxon(str(tax_id))["name"]) for tax_id in tax_ids]
        return sorted(names) if sort else names

    def format_taxon_names(
        self,
        tax_ids: Collection[str],
        *,
        separator: str = "; ",
        sort: bool = True,
    ) -> str:
        """Return taxonomy names joined into one string."""
        return separator.join(self.taxon_names(tax_ids, sort=sort))

    def _require_taxon(self, tax_id: str) -> str:
        value = str(tax_id)
        if value not in self._parent_by_taxon:
            raise TaxonNotFoundError(f"Taxonomy ID not found: {tax_id}")
        return value


def load_taxonomy(
    taxdump_dir: str,
    *,
    names_priority: Sequence[str] = ("scientific name",),
    quiet: bool = True,
) -> TaxonomyOntology:
    """Convenience loader for taxonomy graph."""
    return TaxonomyOntology(taxdump_dir, names_priority=names_priority, quiet=quiet)


def normalize_taxonomy_id(value: object) -> str:
    """Convert taxonomy IDs to canonical string form.

    Integer-like floats such as ``4577.0`` are normalized to ``"4577"``.
    Empty values and NaN-like strings normalize to ``""``.
    """
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        numeric = float(text)
    except Exception:
        return text
    if not math.isfinite(numeric):
        return ""
    if numeric.is_integer():
        return str(int(numeric))
    return text


def compute_taxon_ic_and_lin_maps(
    taxonomy: TaxonomyOntology,
    query_tax_ids: Iterable[object],
    subject_tax_ids: Iterable[object] | None = None,
    *,
    observed_tax_ids: Iterable[object] | None = None,
    observed_mode: str = "observed",
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float]]:
    """Compute taxonomy IC and Lin-similarity lookup maps.

    Returns ``(query_taxon_ic_map, taxon_lin_similarity_map)`` where:
    - ``query_taxon_ic_map`` maps normalized query taxonomy IDs to observed IC
    - ``taxon_lin_similarity_map`` maps ``(query_tax_id, subject_tax_id)`` pairs to
      Lin similarity using ``mode='subtree'``

    When ``observed_mode='whole_db'``, observed IC is prepared from the full BioData protein table.
    When ``observed_tax_ids`` is provided, observed IC is computed from that background.
    Otherwise, observed IC uses the union of query and subject taxonomy IDs.
    """
    normalized_query_tax_ids = [normalize_taxonomy_id(value) for value in query_tax_ids]
    normalized_subject_tax_ids = (
        [normalize_taxonomy_id(value) for value in subject_tax_ids]
        if subject_tax_ids is not None
        else None
    )

    observed_mode_value = _normalize_mode(observed_mode)
    if observed_tax_ids is not None:
        normalized_observed_tax_ids = [normalize_taxonomy_id(value) for value in observed_tax_ids]
        all_taxa: Set[str] = set()
        for tax_id in normalized_observed_tax_ids:
            if tax_id and taxonomy.has_taxon(tax_id):
                all_taxa.add(tax_id)
        annotations: Dict[str, Set[str]] = {tax_id: {tax_id} for tax_id in all_taxa}
        taxonomy.prepare_taxon_counts(annotations, mode="observed")
        observed_mode_value = "observed"
    elif observed_mode_value == "whole_db":
        taxonomy.prepare_taxon_counts({}, mode="whole_db")
    else:
        all_taxa: Set[str] = set()
        for tax_id in normalized_query_tax_ids:
            if tax_id and taxonomy.has_taxon(tax_id):
                all_taxa.add(tax_id)
        if normalized_subject_tax_ids is not None:
            for tax_id in normalized_subject_tax_ids:
                if tax_id and taxonomy.has_taxon(tax_id):
                    all_taxa.add(tax_id)
        annotations = {tax_id: {tax_id} for tax_id in all_taxa}
        taxonomy.prepare_taxon_counts(annotations, mode="observed")

    query_taxon_ic_map: Dict[str, float] = {}
    for tax_id in set(normalized_query_tax_ids):
        if not tax_id or not taxonomy.has_taxon(tax_id):
            query_taxon_ic_map[tax_id] = 0.0
        else:
            query_taxon_ic_map[tax_id] = taxonomy.information_content(tax_id, mode=observed_mode_value)

    taxon_lin_similarity_map: Dict[Tuple[str, str], float] = {}
    if normalized_subject_tax_ids is not None:
        taxonomy.prepare_taxon_counts({}, mode="subtree")
        for query_tax_id, subject_tax_id in set(zip(normalized_query_tax_ids, normalized_subject_tax_ids)):
            if not query_tax_id or not taxonomy.has_taxon(query_tax_id):
                taxon_lin_similarity_map[(query_tax_id, subject_tax_id)] = 0.0
                continue
            if not subject_tax_id or not taxonomy.has_taxon(subject_tax_id):
                taxon_lin_similarity_map[(query_tax_id, subject_tax_id)] = 0.0
                continue
            sim = taxonomy.lin_similarity(query_tax_id, subject_tax_id, mode="subtree")
            taxon_lin_similarity_map[(query_tax_id, subject_tax_id)] = 0.0 if sim is None else float(sim)

    return query_taxon_ic_map, taxon_lin_similarity_map


def read_taxonomy_annotations_tsv(
    path: str,
    *,
    entity_col: int = 0,
    taxon_col: int = 1,
    delimiter: str = "\t",
) -> Dict[str, Set[str]]:
    """Read simple TSV annotations into mapping: entity -> taxonomy IDs."""
    rows: Dict[str, Set[str]] = {}
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter)
            if len(parts) <= max(entity_col, taxon_col):
                continue
            entity = parts[entity_col].strip()
            tax_id = parts[taxon_col].strip()
            if not entity or not tax_id:
                continue
            rows.setdefault(entity, set()).add(tax_id)
    return rows


def _normalize_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in {"observed", "whole_db", "subtree"}:
        raise TaxonomyError("Unknown mode. Use one of: observed, whole_db, subtree.")
    return value


def _fetch_whole_db_taxonomy_annotations() -> Dict[str, Set[str]]:
    try:
        from .BioData import BioDataClient
    except ModuleNotFoundError as exc:
        raise TaxonomyError("Missing BioData client dependencies for mode='whole_db'.") from exc

    with BioDataClient() as client:
        rows = client.query_all(
            """
            SELECT id, taxonomy_id
            FROM protein
            WHERE taxonomy_id IS NOT NULL;
            """
        )

    annotations: Dict[str, Set[str]] = {}
    for row in rows:
        entity_id = str(row.get("id", "")).strip()
        tax_id = normalize_taxonomy_id(row.get("taxonomy_id"))
        if not entity_id or not tax_id:
            continue
        annotations.setdefault(entity_id, set()).add(tax_id)
    return annotations


def _split_taxdump_fields(line: str) -> List[str]:
    return [part.strip() for part in line.rstrip("\n").split("|")[:-1]]


def _parse_nodes_dmp(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    parent_by_taxon: Dict[str, str] = {}
    rank_by_taxon: Dict[str, str] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = _split_taxdump_fields(raw_line)
            if len(parts) < 3:
                continue
            tax_id = parts[0]
            parent_id = parts[1]
            rank = parts[2]
            if not tax_id:
                continue
            parent_by_taxon[tax_id] = parent_id or tax_id
            rank_by_taxon[tax_id] = rank

    if not parent_by_taxon:
        raise TaxonomyError(f"No taxonomy rows parsed from nodes file: {path}")

    for tax_id, parent_id in list(parent_by_taxon.items()):
        if parent_id not in parent_by_taxon:
            parent_by_taxon[tax_id] = tax_id

    return parent_by_taxon, rank_by_taxon


def _parse_names_dmp(path: Path, *, names_priority: Sequence[str]) -> Dict[str, str]:
    preferred_rank = {value: index for index, value in enumerate(names_priority)}
    selected_names: Dict[str, Tuple[int, str]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parts = _split_taxdump_fields(raw_line)
            if len(parts) < 4:
                continue
            tax_id = parts[0]
            name_txt = parts[1]
            name_class = parts[3]
            if not tax_id or not name_txt:
                continue

            if name_class in preferred_rank:
                rank = preferred_rank[name_class]
            elif "scientific name" not in preferred_rank:
                rank = len(preferred_rank)
            else:
                rank = len(preferred_rank) + 1

            current = selected_names.get(tax_id)
            if current is None or rank < current[0]:
                selected_names[tax_id] = (rank, name_txt)

    return {tax_id: name for tax_id, (_, name) in selected_names.items()}


def _build_children_index(parent_by_taxon: Mapping[str, str]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = defaultdict(list)
    for tax_id, parent_id in parent_by_taxon.items():
        if tax_id == parent_id:
            continue
        children[parent_id].append(tax_id)

    for value in children.values():
        value.sort()
    return dict(children)


def _compute_depths(parent_by_taxon: Mapping[str, str]) -> Dict[str, int]:
    depths: Dict[str, int] = {}

    def visit(tax_id: str, visiting: Set[str]) -> int:
        if tax_id in depths:
            return depths[tax_id]
        if tax_id in visiting:
            return 0

        visiting.add(tax_id)
        parent_id = parent_by_taxon.get(tax_id, tax_id)
        if parent_id == tax_id or parent_id not in parent_by_taxon:
            depth = 0
        else:
            depth = visit(parent_id, visiting) + 1
        visiting.remove(tax_id)
        depths[tax_id] = depth
        return depth

    for tax_id in parent_by_taxon.keys():
        visit(tax_id, set())
    return depths


def _compute_subtree_sizes(children_by_taxon: Mapping[str, Sequence[str]]) -> Dict[str, int]:
    all_taxa: Set[str] = set(children_by_taxon.keys())
    for children in children_by_taxon.values():
        all_taxa.update(children)

    memo: Dict[str, int] = {}

    def subtree_size(tax_id: str) -> int:
        if tax_id in memo:
            return memo[tax_id]
        size = 1
        for child in children_by_taxon.get(tax_id, []):
            size += subtree_size(child)
        memo[tax_id] = size
        return size

    for tax_id in all_taxa:
        subtree_size(tax_id)
    return memo


__all__ = [
    "TaxonomyOntology",
    "TaxonomyError",
    "TaxonNotFoundError",
    "TaxonCountsNotPreparedError",
    "compute_taxon_ic_and_lin_maps",
    "load_taxonomy",
    "normalize_taxonomy_id",
    "read_taxonomy_annotations_tsv",
]
