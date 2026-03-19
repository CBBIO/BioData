# CBBIO/GO.py Documentation

## General Description
`CBBIO.GO` provides ontology utilities built on top of `goatools`, independent from database access. It wraps a GO DAG (`GOOntology`) and adds methods for term lookup, graph navigation, term-to-term relationship queries, information content, semantic similarity, annotation grouping by GO category, and branch-distance calculations. It also includes file-based annotation loading helpers.

Main dependencies:
- `goatools` for ontology parsing and semantic metrics.
- Standard library only for TSV parsing and basic transformations.

## Exceptions
- `GOError`: base module-level exception.
- `GOTermNotFoundError`: raised when a requested GO term is not present in the loaded DAG.
- `GOCountsNotPreparedError`: raised when IC/similarity methods are called before `prepare_term_counts(...)`.

## Class: `GOOntology`

### `__init__(obo_path, *, load_obsolete=False, optional_attrs=None, quiet=True)`
Initializes the ontology from an OBO file using `goatools.obo_parser.GODag`.
- Parameters:
  - `obo_path`: path to the `.obo` ontology file.
  - `load_obsolete`: include obsolete terms if `True`.
  - `optional_attrs`: extra GO term attributes to load.
  - `quiet`: suppress goatools parser output when `True`.
- Internal state:
  - `self._dag`: loaded GO DAG.
  - `self._term_counts`: initially `None` until prepared.
  - `self.obo_path`: stored source path.
- Raises:
  - `GOError` if `goatools` is not installed.

### `go_ids` (property)
Returns all GO IDs in the loaded DAG as `set[str]`.

### `has_term(go_id)`
Checks whether `go_id` exists in the DAG.
- Returns: `True` if present, otherwise `False`.

### `term(go_id)`
Returns normalized metadata for one GO term.
- Returns dict with keys:
  - `id`, `name`, `namespace`, `level`, `depth`, `is_obsolete`.
- Raises:
  - `GOTermNotFoundError` if term does not exist.

### `ancestors(go_id, *, include_self=False)`
Returns all ancestor GO IDs for a term.
- Behavior:
  - Uses `get_all_parents()` from goatools.
  - Optionally includes the input term.
  - Output is sorted.
- Raises:
  - `GOTermNotFoundError` for unknown term.

### `descendants(go_id, *, include_self=False)`
Returns all descendant GO IDs for a term.
- Behavior:
  - Uses `get_all_children()` from goatools.
  - Optionally includes the input term.
  - Output is sorted.
- Raises:
  - `GOTermNotFoundError` for unknown term.

### `direct_parents(go_id)`
Returns direct parents of a term.
- Behavior:
  - Uses the goatools term object's direct `parents`.
  - Output is sorted.
- Raises:
  - `GOTermNotFoundError` for unknown term.

### `direct_children(go_id)`
Returns direct children of a term.
- Behavior:
  - Uses the goatools term object's direct `children`.
  - Output is sorted.
- Raises:
  - `GOTermNotFoundError` for unknown term.

### `is_parent(candidate_parent_id, go_id)`
Returns `True` when `candidate_parent_id` is a direct parent of `go_id`.

### `is_child(candidate_child_id, go_id)`
Returns `True` when `candidate_child_id` is a direct child of `go_id`.

### `is_ancestor(candidate_ancestor_id, go_id, *, include_self=False)`
Returns `True` when `candidate_ancestor_id` is an ancestor of `go_id`.
- `include_self=True` treats identical terms as related.

### `is_ascendant(candidate_ascendant_id, go_id, *, include_self=False)`
Alias of `is_ancestor(...)`.

### `is_descendant(candidate_descendant_id, go_id, *, include_self=False)`
Returns `True` when `candidate_descendant_id` is a descendant of `go_id`.
- `include_self=True` treats identical terms as related.

### `is_descendent(candidate_descendent_id, go_id, *, include_self=False)`
Alias of `is_descendant(...)`.

### `are_in_the_same_path(go_id_a, go_id_b, *, include_self=True)`
Returns `True` when one term lies on the ancestor/descendant path of the other.
- Equivalent to checking whether `a` is an ancestor of `b` or `b` is an ancestor of `a`.

### `find_relation(go_id_a, go_id_b)`
Returns the closest directed relation between two terms.
- Possible values:
  - `same`
  - `parent`
  - `child`
  - `ancestor`
  - `descendant`
  - `non-related`
- Precedence is exact match first, then direct relations, then indirect relations.

### `relation_rank(relation)`
Returns integer priority for a relation label.
- Lower rank means closer relation.
- Current ordering:
  - `same`
  - `parent`
  - `child`
  - `ancestor`
  - `descendant`
  - `non-related`
- Raises:
  - `GOError` on unknown relation label.

### `best_relation_matches(go_id, other_terms)`
Finds the closest relation from one term to a group of terms.
- Behavior:
  - Filters invalid GO IDs from `other_terms`.
  - Computes `find_relation(go_id, other_go_id)` for each valid term.
  - Keeps all terms tied for the best relation level.
- Returns dict:
  - `{"relation": <best_relation>, "matches": [<go_id>, ...]}`

### `best_relation_map(terms_a, terms_b)`
Applies `best_relation_matches(...)` to each valid term in `terms_a`.
- Returns mapping:
  - `{term_in_a: {"relation": <best_relation>, "matches": [...]}, ...}`

### `common_ancestors(go_id_a, go_id_b, *, include_terms=True)`
Computes shared ancestors between two terms.
- Behavior:
  - Computes ancestors for both terms, optionally including each term itself.
  - Returns sorted intersection.
- Raises:
  - `GOTermNotFoundError` if either term is unknown.

### `prepare_term_counts(annotations, *, relationships=None)`
Builds `TermCounts` needed for IC and semantic similarity.
- Parameters:
  - `annotations`: mapping `entity_id -> collection[go_id]`.
  - `relationships`: optional relationship filter passed to goatools.
- Behavior:
  - Filters invalid GO IDs (not present in current DAG).
  - Stores counts in `self._term_counts`.
- Raises:
  - `GOError` if `goatools` is not installed.

### `information_content(go_id)`
Returns IC value for one GO term using prepared term counts.
- Behavior:
  - Validates term existence.
  - Calls `goatools.semantic.get_info_content`.
  - Normalizes output to `float` via `_as_float`.
- Raises:
  - `GOTermNotFoundError` for unknown term.
  - `GOCountsNotPreparedError` if term counts are missing.

### `semantic_similarity(go_id_a, go_id_b, *, method="resnik")`
Computes semantic similarity between two terms.
- Supported methods:
  - `resnik`
  - `lin`
  - `schlicker`
  - `wang`
- Behavior:
  - Requires valid terms.
  - For `resnik`, `lin`, and `schlicker`, requires prepared term counts.
  - For `wang`, uses goatools Wang termwise implementation.
  - Dispatches to the corresponding goatools semantic implementation.
- Raises:
  - `GOTermNotFoundError` if any term is unknown.
  - `GOCountsNotPreparedError` if term counts are missing for IC-based methods.
  - `GOError` if method is unsupported.

### `group_similarity(terms_a, terms_b, *, method="resnik", aggregate="bma")`
Computes group-to-group similarity using best-match average (BMA).
- Parameters:
  - `terms_a`, `terms_b`: GO-term collections.
  - `method`: semantic similarity method (`resnik`, `lin`, `schlicker`, `wang`).
  - `aggregate`: only `bma` is supported.
- Behavior:
  - For each term in one set, takes max similarity against the other set.
  - Averages both directions and returns the mean.
- Returns:
  - `float` similarity, or `None` if an input group is empty or no scores were produced.
- Raises:
  - `GOError` if aggregate mode is unsupported.
  - Propagates errors from `semantic_similarity`.

### `term_names(go_ids, *, sort=True)`
Returns term names for a set/list of GO IDs.
- Behavior:
  - Resolves each ID via `term(...)`.
  - Returns sorted names by default.
- Raises:
  - `GOTermNotFoundError` if any ID is invalid.

### `format_term_names(go_ids, *, separator='; ', sort=True)`
Formats GO term names as a single string.
- Behavior:
  - Uses `term_names(...)` then joins with `separator`.

### `category_for_term(go_id)`
Maps a term namespace to canonical short category.
- Namespace mapping:
  - `molecular_function -> mf`
  - `biological_process -> bp`
  - `cellular_component -> cc`
- Returns:
  - `"mf"`, `"bp"`, `"cc"`, or `None` for unknown namespace.
- Raises:
  - `GOTermNotFoundError` if term is invalid.

### `split_annotations_by_category(annotations)`
Splits `entity -> GO IDs` into category-specific mappings.
- Returns dict shape:
  - `{"mf": {...}, "bp": {...}, "cc": {...}}`
- Behavior:
  - Skips unknown GO IDs.
  - Skips IDs whose namespace does not map to canonical categories.
  - Keeps values as sets for de-duplication.

### `filter_valid_terms(go_ids, *, sort=True)`
Returns unique GO IDs that exist in the currently loaded DAG.
- Behavior:
  - Drops invalid/unknown GO IDs.
  - Removes duplicates.
  - Sorts by default.

### `minimal_branch_length(go_id_a, go_id_b, *, branch_dist=None)`
Returns minimum branch distance between two GO terms.
- Behavior:
  - Uses `goatools.semantic.min_branch_length` when available.
  - Falls back to `_fallback_min_branch_length` otherwise.
- Returns:
  - `int` distance or `None` when not computable.
- Raises:
  - `GOTermNotFoundError` if any term is unknown.
  - `GOError` if `goatools` import is missing.

### `_get_term(go_id)`
Internal validator and getter for DAG term objects.
- Raises:
  - `GOTermNotFoundError` if term is missing.

### `_require_term_counts()`
Internal guard ensuring term counts have been prepared.
- Raises:
  - `GOCountsNotPreparedError` if `prepare_term_counts(...)` has not been called.

## Module-Level Functions

### `load_go(obo_path, *, load_obsolete=False, optional_attrs=None, quiet=True)`
Convenience constructor returning a `GOOntology` instance.

### `read_annotations_tsv(path, *, entity_col=0, go_col=1, delimiter='\t')`
Reads annotation rows from a delimited text file into `dict[str, set[str]]`.
- Expected default format per line: `entity_id<TAB>go_id`.
- Behavior:
  - Skips blank/comment lines (`#...`).
  - Skips rows without enough columns.
  - Skips empty entity or GO values.
  - De-duplicates GO IDs per entity.

### `_as_float(value)`
Internal converter used to normalize goatools outputs to `float`.
- Returns `0.0` for `None` or non-convertible values.

### `_as_optional_int(value)`
Internal converter to `Optional[int]`.
- Returns `None` when conversion is not possible.
- Converts `bool` to `0/1`.

### `_fallback_min_branch_length(term_a, term_b, dag, *, branch_dist)`
Fallback implementation for branch-length calculation.
- Behavior:
  - If namespaces match: computes distance via deepest common ancestor depth.
  - If namespaces differ: returns `depth(a) + depth(b) + branch_dist` when `branch_dist` is provided.
  - Otherwise returns `None`.
