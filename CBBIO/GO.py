"""GO ontology helpers powered by goatools.

This module is intentionally independent from database access code.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Set, cast

from .types import SimilarityMethod


class GOError(Exception):
    """Base exception for ``CBBIO.GO``."""


class GOTermNotFoundError(GOError):
    """Raised when a GO ID does not exist in the loaded DAG."""


class GOCountsNotPreparedError(GOError):
    """Raised when IC/similarity is requested without term counts."""


class GOOntology:
    """Wrapper around a GO DAG with convenience helpers."""

    _RELATION_RANKS: Dict[str, int] = {
        "same": 0,
        "parent": 1,
        "child": 2,
        "ancestor": 3,
        "descendant": 4,
        "non-related": 5,
    }

    def __init__(
        self,
        obo_path: str,
        *,
        load_obsolete: bool = False,
        optional_attrs: Optional[Set[str]] = None,
        quiet: bool = True,
    ) -> None:
        try:
            from goatools.obo_parser import GODag  # type: ignore
        except ModuleNotFoundError as exc:
            raise GOError("Missing dependency 'goatools'. Install with: pip install goatools") from exc

        prt = None if quiet else sys.stdout
        dag_obj: Any = GODag(
            obo_file=str(obo_path),
            optional_attrs=optional_attrs,
            load_obsolete=load_obsolete,
            prt=prt,
        )
        self._dag: Any = dag_obj
        self._term_counts: Optional[Any] = None
        self._wang_ss: Optional[Any] = None
        self._direct_parent_cache: Dict[str, Set[str]] = {}
        self._direct_child_cache: Dict[str, Set[str]] = {}
        self._ancestor_cache: Dict[str, Set[str]] = {}
        self._descendant_cache: Dict[str, Set[str]] = {}
        self.obo_path = str(obo_path)

    @property
    def go_ids(self) -> Set[str]:
        return {str(go_id) for go_id in self._dag.keys()}

    def has_term(self, go_id: str) -> bool:
        return go_id in self._dag

    def term(self, go_id: str) -> Dict[str, Any]:
        term_obj = self._get_term(go_id)
        return {
            "id": str(term_obj.id),
            "name": str(term_obj.name),
            "namespace": str(term_obj.namespace),
            "level": int(term_obj.level),
            "depth": int(term_obj.depth),
            "is_obsolete": bool(getattr(term_obj, "is_obsolete", False)),
        }

    def ancestors(self, go_id: str, *, include_self: bool = False) -> List[str]:
        values = set(self._ancestor_set(go_id))
        if include_self:
            values.add(go_id)
        return sorted(values)

    def descendants(self, go_id: str, *, include_self: bool = False) -> List[str]:
        values = set(self._descendant_set(go_id))
        if include_self:
            values.add(go_id)
        return sorted(values)

    def direct_parents(self, go_id: str) -> List[str]:
        """Return direct parents of ``go_id`` as sorted GO IDs."""
        return sorted(self._direct_parent_set(go_id))

    def direct_children(self, go_id: str) -> List[str]:
        """Return direct children of ``go_id`` as sorted GO IDs."""
        return sorted(self._direct_child_set(go_id))

    def is_parent(self, candidate_parent_id: str, go_id: str) -> bool:
        """Return ``True`` when ``candidate_parent_id`` is a direct parent of ``go_id``."""
        self._get_term(candidate_parent_id)
        return str(candidate_parent_id) in self._direct_parent_set(go_id)

    def is_child(self, candidate_child_id: str, go_id: str) -> bool:
        """Return ``True`` when ``candidate_child_id`` is a direct child of ``go_id``."""
        self._get_term(candidate_child_id)
        return str(candidate_child_id) in self._direct_child_set(go_id)

    def is_ancestor(self, candidate_ancestor_id: str, go_id: str, *, include_self: bool = False) -> bool:
        """Return ``True`` when ``candidate_ancestor_id`` is an ancestor of ``go_id``."""
        self._get_term(candidate_ancestor_id)
        if include_self and str(candidate_ancestor_id) == str(go_id):
            return True
        return str(candidate_ancestor_id) in self._ancestor_set(go_id)

    def is_ascendant(self, candidate_ascendant_id: str, go_id: str, *, include_self: bool = False) -> bool:
        """Alias for :meth:`is_ancestor` using ascendant terminology."""
        return self.is_ancestor(candidate_ascendant_id, go_id, include_self=include_self)

    def is_descendant(self, candidate_descendant_id: str, go_id: str, *, include_self: bool = False) -> bool:
        """Return ``True`` when ``candidate_descendant_id`` is a descendant of ``go_id``."""
        self._get_term(candidate_descendant_id)
        if include_self and str(candidate_descendant_id) == str(go_id):
            return True
        return str(candidate_descendant_id) in self._descendant_set(go_id)

    def is_descendent(self, candidate_descendent_id: str, go_id: str, *, include_self: bool = False) -> bool:
        """Alias for :meth:`is_descendant` preserving alternate spelling."""
        return self.is_descendant(candidate_descendent_id, go_id, include_self=include_self)

    def are_in_the_same_path(self, go_id_a: str, go_id_b: str, *, include_self: bool = True) -> bool:
        """Return ``True`` when one term lies on the ancestor/descendant path of the other."""
        self._get_term(go_id_a)
        self._get_term(go_id_b)
        return self.is_ancestor(go_id_a, go_id_b, include_self=include_self) or self.is_ancestor(
            go_id_b, go_id_a, include_self=include_self
        )

    def find_relation(self, go_id_a: str, go_id_b: str) -> str:
        """Return the closest directed relation between two terms.

        Possible values are:
        ``same``, ``parent``, ``child``, ``ancestor``, ``descendant``, ``non-related``.
        """
        self._get_term(go_id_a)
        self._get_term(go_id_b)

        if str(go_id_a) == str(go_id_b):
            return "same"
        if self.is_parent(go_id_a, go_id_b):
            return "parent"
        if self.is_child(go_id_a, go_id_b):
            return "child"
        if self.is_ancestor(go_id_a, go_id_b):
            return "ancestor"
        if self.is_descendant(go_id_a, go_id_b):
            return "descendant"
        return "non-related"

    def relation_rank(self, relation: str) -> int:
        """Return integer priority for a relation label; lower means closer."""
        value = str(relation).strip().lower()
        if value not in self._RELATION_RANKS:
            raise GOError(f"Unknown relation: {relation}")
        return self._RELATION_RANKS[value]

    def best_relation_matches(self, go_id: str, other_terms: Collection[str]) -> Dict[str, object]:
        """Return the closest relation from ``go_id`` to a group of terms plus all matching terms."""
        self._get_term(go_id)
        valid_terms = self.filter_valid_terms(other_terms)
        if not valid_terms:
            return {"relation": "non-related", "matches": []}

        best_relation = "non-related"
        best_rank = self.relation_rank(best_relation)
        matches: List[str] = []

        for other_go_id in valid_terms:
            relation = self.find_relation(go_id, other_go_id)
            rank = self.relation_rank(relation)
            if rank < best_rank:
                best_relation = relation
                best_rank = rank
                matches = [other_go_id]
            elif rank == best_rank:
                matches.append(other_go_id)

        return {"relation": best_relation, "matches": matches}

    def best_relation_map(
        self,
        terms_a: Collection[str],
        terms_b: Collection[str],
    ) -> Dict[str, Dict[str, object]]:
        """Map each valid term in ``terms_a`` to its closest relation against ``terms_b``."""
        valid_terms_a = self.filter_valid_terms(terms_a)
        return {
            go_id: self.best_relation_matches(go_id, terms_b)
            for go_id in valid_terms_a
        }

    def common_ancestors(self, go_id_a: str, go_id_b: str, *, include_terms: bool = True) -> List[str]:
        ancestors_a = set(self.ancestors(go_id_a, include_self=include_terms))
        ancestors_b = set(self.ancestors(go_id_b, include_self=include_terms))
        return sorted(ancestors_a.intersection(ancestors_b))

    def filter_valid_terms(self, go_ids: Collection[str], *, sort: bool = True) -> List[str]:
        """Return unique GO IDs present in the loaded DAG."""
        values = {str(go_id) for go_id in go_ids if self.has_term(str(go_id))}
        return sorted(values) if sort else list(values)

    def prepare_term_counts(
        self,
        annotations: Mapping[str, Collection[str]],
        *,
        relationships: Optional[Set[str]] = None,
    ) -> None:
        """Build GO term counts from gene/protein -> GO annotations."""
        try:
            from goatools.semantic import TermCounts  # type: ignore
        except ModuleNotFoundError as exc:
            raise GOError("Missing dependency 'goatools'. Install with: pip install goatools") from exc

        prepared: Dict[str, Set[str]] = {}
        valid_go_ids = self.go_ids
        for entity_id, go_terms in annotations.items():
            filtered = {str(go_id) for go_id in go_terms if str(go_id) in valid_go_ids}
            if filtered:
                prepared[str(entity_id)] = filtered

        self._term_counts = TermCounts(self._dag, prepared, relationships=relationships)

    def information_content(self, go_id: str) -> float:
        """Return information content for a GO term using prepared counts."""
        self._get_term(go_id)
        term_counts = self._require_term_counts()
        from goatools.semantic import get_info_content  # type: ignore

        get_ic_fn = cast(Callable[..., Any], get_info_content)
        return _as_float(get_ic_fn(go_id, term_counts))

    def semantic_similarity(self, go_id_a: str, go_id_b: str, *, method: SimilarityMethod = "resnik") -> float:
        """Compute semantic similarity between two GO terms."""
        self._get_term(go_id_a)
        self._get_term(go_id_b)
        method_value = method.strip().lower()

        from goatools import semantic as goat_semantic  # type: ignore
        semantic_mod = cast(Any, goat_semantic)

        if method_value == "resnik":
            term_counts = self._require_term_counts()
            sim_fn = cast(Callable[..., Any], semantic_mod.resnik_sim)
            return _as_float(sim_fn(go_id_a, go_id_b, self._dag, term_counts))
        if method_value == "lin":
            term_counts = self._require_term_counts()
            sim_fn = cast(Callable[..., Any], semantic_mod.lin_sim)
            return _as_float(sim_fn(go_id_a, go_id_b, self._dag, term_counts))
        if method_value == "schlicker":
            term_counts = self._require_term_counts()
            sim_fn = cast(Callable[..., Any], semantic_mod.schlicker_sim)
            return _as_float(sim_fn(go_id_a, go_id_b, self._dag, term_counts))
        if method_value == "wang":
            # Compatibility path for goatools versions exposing a direct semantic.wang_sim.
            sim_fn = cast(Callable[..., Any], getattr(semantic_mod, "wang_sim", None))
            if callable(sim_fn):
                try:
                    return _as_float(sim_fn(go_id_a, go_id_b, self._dag))
                except TypeError:
                    term_counts = self._require_term_counts()
                    return _as_float(sim_fn(go_id_a, go_id_b, self._dag, term_counts))

            wang_ss = self._get_wang_ss()
            get_sim_fn = cast(Callable[..., Any], getattr(wang_ss, "get_sim", None))
            if not callable(get_sim_fn):
                raise GOError("Wang method is not available in the installed goatools version.")
            return _as_float(get_sim_fn(go_id_a, go_id_b))

        raise GOError("Unknown method. Use one of: resnik, lin, schlicker, wang.")

    def group_similarity(
        self,
        terms_a: Collection[str],
        terms_b: Collection[str],
        *,
        method: SimilarityMethod = "resnik",
        aggregate: str = "bma",
    ) -> Optional[float]:
        """Compute semantic similarity between two GO-term groups.

        Currently supports best-match average aggregation (``aggregate='bma'``).
        """
        if not terms_a or not terms_b:
            return None

        aggregate_value = aggregate.strip().lower()
        if aggregate_value != "bma":
            raise GOError("Unknown aggregate. Use: bma.")

        terms_a_list = [str(go_id) for go_id in terms_a]
        terms_b_list = [str(go_id) for go_id in terms_b]

        scores_a: List[float] = []
        for go_id_a in terms_a_list:
            row_scores = [
                float(self.semantic_similarity(go_id_a, go_id_b, method=method))
                for go_id_b in terms_b_list
            ]
            if row_scores:
                scores_a.append(max(row_scores))

        scores_b: List[float] = []
        for go_id_b in terms_b_list:
            row_scores = [
                float(self.semantic_similarity(go_id_b, go_id_a, method=method))
                for go_id_a in terms_a_list
            ]
            if row_scores:
                scores_b.append(max(row_scores))

        if not scores_a or not scores_b:
            return None

        avg_a = sum(scores_a) / float(len(scores_a))
        avg_b = sum(scores_b) / float(len(scores_b))
        return (avg_a + avg_b) / 2.0

    def term_names(self, go_ids: Collection[str], *, sort: bool = True) -> List[str]:
        """Return GO term names for provided GO IDs."""
        names = [str(self.term(str(go_id))["name"]) for go_id in go_ids]
        return sorted(names) if sort else names

    def format_term_names(
        self,
        go_ids: Collection[str],
        *,
        separator: str = "; ",
        sort: bool = True,
    ) -> str:
        """Return GO term names as one formatted string."""
        return separator.join(self.term_names(go_ids, sort=sort))

    def category_for_term(self, go_id: str) -> Optional[str]:
        """Return canonical GO category for a term: ``mf``, ``bp``, or ``cc``."""
        term_obj = self._get_term(go_id)
        namespace = str(getattr(term_obj, "namespace", "")).strip().lower()
        namespace_to_category = {
            "molecular_function": "mf",
            "biological_process": "bp",
            "cellular_component": "cc",
        }
        return namespace_to_category.get(namespace)

    def split_annotations_by_category(
        self,
        annotations: Mapping[str, Collection[str]],
    ) -> Dict[str, Dict[str, Set[str]]]:
        """Split entity->GO IDs mapping into canonical GO categories.

        Returns mapping:
        ``{"mf": {entity: {go...}}, "bp": {...}, "cc": {...}}``.
        Invalid/unknown GO IDs are skipped.
        """
        rows: Dict[str, Dict[str, Set[str]]] = {
            "mf": {},
            "bp": {},
            "cc": {},
        }
        for entity_id, go_ids in annotations.items():
            for go_id_raw in go_ids:
                go_id = str(go_id_raw)
                if not self.has_term(go_id):
                    continue
                category = self.category_for_term(go_id)
                if category is None:
                    continue
                rows[category].setdefault(str(entity_id), set()).add(go_id)
        return rows

    def minimal_branch_length(
        self,
        go_id_a: str,
        go_id_b: str,
        *,
        branch_dist: Optional[int] = None,
    ) -> Optional[int]:
        """Return minimum branch distance between two GO terms.

        This wraps ``goatools.semantic.min_branch_length`` when available.
        """
        term_a = self._get_term(go_id_a)
        term_b = self._get_term(go_id_b)

        try:
            from goatools import semantic as goat_semantic  # type: ignore
        except ModuleNotFoundError as exc:
            raise GOError("Missing dependency 'goatools'. Install with: pip install goatools") from exc

        semantic_mod = cast(Any, goat_semantic)
        fn = getattr(semantic_mod, "min_branch_length", None)
        if callable(fn):
            return _as_optional_int(fn(go_id_a, go_id_b, self._dag, branch_dist))

        return _fallback_min_branch_length(term_a, term_b, self._dag, branch_dist=branch_dist)

    def _get_term(self, go_id: str) -> Any:
        term_obj = self._dag.get(go_id)
        if term_obj is None:
            raise GOTermNotFoundError(f"GO term not found: {go_id}")
        return term_obj

    def _direct_parent_set(self, go_id: str) -> Set[str]:
        go_id = str(go_id)
        if go_id not in self._direct_parent_cache:
            term_obj = self._get_term(go_id)
            self._direct_parent_cache[go_id] = {
                str(parent.id) for parent in getattr(term_obj, "parents", set())
            }
        return self._direct_parent_cache[go_id]

    def _direct_child_set(self, go_id: str) -> Set[str]:
        go_id = str(go_id)
        if go_id not in self._direct_child_cache:
            term_obj = self._get_term(go_id)
            self._direct_child_cache[go_id] = {
                str(child.id) for child in getattr(term_obj, "children", set())
            }
        return self._direct_child_cache[go_id]

    def _ancestor_set(self, go_id: str) -> Set[str]:
        go_id = str(go_id)
        if go_id not in self._ancestor_cache:
            term_obj = self._get_term(go_id)
            self._ancestor_cache[go_id] = {str(value) for value in term_obj.get_all_parents()}
        return self._ancestor_cache[go_id]

    def _descendant_set(self, go_id: str) -> Set[str]:
        go_id = str(go_id)
        if go_id not in self._descendant_cache:
            term_obj = self._get_term(go_id)
            self._descendant_cache[go_id] = {str(value) for value in term_obj.get_all_children()}
        return self._descendant_cache[go_id]

    def _require_term_counts(self) -> Any:
        if self._term_counts is None:
            raise GOCountsNotPreparedError("Call prepare_term_counts(...) before IC-based similarity operations.")
        return self._term_counts

    def _get_wang_ss(self) -> Any:
        if self._wang_ss is None:
            try:
                from goatools.semsim.termwise.wang import SsWang  # type: ignore
            except ModuleNotFoundError as exc:
                raise GOError("Wang method is not available in the installed goatools version.") from exc
            self._wang_ss = SsWang(self.go_ids, self._dag)
        return self._wang_ss


def load_go(
    obo_path: str,
    *,
    load_obsolete: bool = False,
    optional_attrs: Optional[Set[str]] = None,
    quiet: bool = True,
) -> GOOntology:
    """Convenience loader for GO ontology."""
    return GOOntology(
        obo_path=obo_path,
        load_obsolete=load_obsolete,
        optional_attrs=optional_attrs,
        quiet=quiet,
    )


def read_annotations_tsv(
    path: str,
    *,
    entity_col: int = 0,
    go_col: int = 1,
    delimiter: str = "\t",
) -> Dict[str, Set[str]]:
    """Read simple TSV annotations into mapping: entity -> GO IDs.

    Expected row format (default): ``entity_id<TAB>go_id``.
    """
    rows: Dict[str, Set[str]] = {}
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter)
            if len(parts) <= max(entity_col, go_col):
                continue
            entity = parts[entity_col].strip()
            go_id = parts[go_col].strip()
            if not entity or not go_id:
                continue
            rows.setdefault(entity, set()).add(go_id)
    return rows


def _as_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return 0.0


def _as_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _fallback_min_branch_length(term_a: Any, term_b: Any, dag: Any, *, branch_dist: Optional[int]) -> Optional[int]:
    # This mirrors goatools behavior when min_branch_length is unavailable.
    if getattr(term_a, "namespace", None) == getattr(term_b, "namespace", None):
        common = set(term_a.get_all_parents()).intersection(set(term_b.get_all_parents()))
        common.update({term_a.id, term_b.id})
        if not common:
            return None
        deepest = max(common, key=lambda go_id: dag[go_id].depth)
        dca_depth = int(dag[deepest].depth)
        return (int(term_a.depth) - dca_depth) + (int(term_b.depth) - dca_depth)

    if branch_dist is not None:
        return int(term_a.depth) + int(term_b.depth) + int(branch_dist)
    return None


__all__ = [
    "GOOntology",
    "GOError",
    "GOTermNotFoundError",
    "GOCountsNotPreparedError",
    "SimilarityMethod",
    "load_go",
    "read_annotations_tsv",
]
