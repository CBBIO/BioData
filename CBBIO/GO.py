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
        term_obj = self._get_term(go_id)
        values = {str(value) for value in term_obj.get_all_parents()}
        if include_self:
            values.add(go_id)
        return sorted(values)

    def descendants(self, go_id: str, *, include_self: bool = False) -> List[str]:
        term_obj = self._get_term(go_id)
        values = {str(value) for value in term_obj.get_all_children()}
        if include_self:
            values.add(go_id)
        return sorted(values)

    def common_ancestors(self, go_id_a: str, go_id_b: str, *, include_terms: bool = True) -> List[str]:
        ancestors_a = set(self.ancestors(go_id_a, include_self=include_terms))
        ancestors_b = set(self.ancestors(go_id_b, include_self=include_terms))
        return sorted(ancestors_a.intersection(ancestors_b))

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
        term_counts = self._require_term_counts()
        method_value = method.strip().lower()

        from goatools import semantic as goat_semantic  # type: ignore
        semantic_mod = cast(Any, goat_semantic)

        if method_value == "resnik":
            sim_fn = cast(Callable[..., Any], semantic_mod.resnik_sim)
            return _as_float(sim_fn(go_id_a, go_id_b, self._dag, term_counts))
        if method_value == "lin":
            sim_fn = cast(Callable[..., Any], semantic_mod.lin_sim)
            return _as_float(sim_fn(go_id_a, go_id_b, self._dag, term_counts))
        if method_value == "schlicker":
            sim_fn = cast(Callable[..., Any], semantic_mod.schlicker_sim)
            return _as_float(sim_fn(go_id_a, go_id_b, self._dag, term_counts))

        raise GOError("Unknown method. Use one of: resnik, lin, schlicker.")

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

    def _require_term_counts(self) -> Any:
        if self._term_counts is None:
            raise GOCountsNotPreparedError("Call prepare_term_counts(...) before IC/similarity operations.")
        return self._term_counts


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
