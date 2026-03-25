# CBBIO/Taxonomy.py Documentation

## General Description
`CBBIO.Taxonomy` provides taxonomy utilities for NCBI taxdump files (`nodes.dmp`, `names.dmp`), independent from database access. It wraps the taxonomy graph (`TaxonomyOntology`) and adds lineage navigation, LCA/branch-distance operations, taxon information-content (IC) utilities, Lin similarity, ID normalization helpers, and TSV annotation loading helpers.

Main dependencies:
- Standard library only (no external runtime dependency).

## Exceptions
- `TaxonomyError`: base module-level exception.
- `TaxonNotFoundError`: raised when a requested taxonomy ID does not exist in the loaded graph.
- `TaxonCountsNotPreparedError`: raised when IC methods are called before `prepare_taxon_counts(...)` for the requested mode.

## Class: `TaxonomyOntology`

### `__init__(taxdump_dir, *, names_priority=("scientific name",), quiet=True)`
Initializes taxonomy from local taxdump files.
- Required files under `taxdump_dir`:
  - `nodes.dmp`
  - `names.dmp`
- Internal state:
  - parent/rank mappings
  - preferred display names
  - child index and depth cache
  - prepared IC probabilities by mode

### `taxon_ids` (property)
Returns all loaded taxonomy IDs as `set[str]`.

### `has_taxon(tax_id)`
Checks whether `tax_id` exists in the graph.

### `taxon(tax_id)`
Returns normalized metadata:
- `id`, `name`, `rank`, `parent_id`, `depth`.

### `ancestors(tax_id, *, include_self=False)`
Returns lineage-aligned ancestors for one taxon.

### `descendants(tax_id, *, include_self=False)`
Returns descendants sorted by `(depth, tax_id)`.

### `lineage(tax_id, *, include_self=True, include_root=True)`
Returns lineage in root-to-leaf order with configurable root/self inclusion.

### `common_ancestors(tax_id_a, tax_id_b, *, include_terms=True)`
Returns sorted shared ancestors.

### `lowest_common_ancestor(tax_id_a, tax_id_b)`
Returns deepest shared ancestor ID, or `None` when unavailable.

### `lowest_common_ancestor_rank(tax_id_a, tax_id_b)`
Returns the rank of the lowest common ancestor (for example: `family`, `genus`, `order`).

### `lowest_common_ancestor_clade(tax_id_a, tax_id_b)`
Returns metadata for the shared clade:
- `id`, `name`, `rank`, `depth`.

### `shares_clade_at_rank(tax_id_a, tax_id_b, rank)`
Returns `True` when both taxa share the same ancestor at the requested rank.
Example: `shares_clade_at_rank(a, b, "genus") -> False`,
`shares_clade_at_rank(a, b, "order") -> True`.

### `minimal_branch_length(tax_id_a, tax_id_b)`
Returns branch distance via LCA depth:
`dist(a,lca) + dist(b,lca)`.

### `normalized_lca_depth(tax_id_a, tax_id_b)`
Returns normalized shared-depth score:
`depth(LCA) / max(depth(a), depth(b))`.
- Returns `1.0` when both taxa are at root depth (`0/0` case).

### `wu_palmer_similarity(tax_id_a, tax_id_b)`
Returns taxonomy Wu-Palmer similarity:
`2 * depth(LCA) / (depth(a) + depth(b))`.
- Returns `1.0` when both taxa are at root depth (`0/0` case).

### `lin_similarity(tax_id_a, tax_id_b, *, mode="observed")`
Returns taxonomy Lin similarity:
`2 * IC(LCA) / (IC(a) + IC(b))`.
- Uses the prepared IC probabilities for the requested mode.
- Returns `1.0` when both IC values are zero.
- Requires `prepare_taxon_counts(..., mode=mode)` first.

### `prepare_taxon_counts(annotations, *, mode="observed")`
Prepares IC probabilities.
- `mode="observed"`:
  - `annotations` is `entity_id -> taxonomy_ids`.
  - Expands each assigned taxon to its lineage (ancestor propagation).
  - Computes probability by entity frequency.
- `mode="whole_db"`:
  - Reads all `protein.taxonomy_id` assignments from the BioData database through `BioDataClient`.
  - Expands each assigned taxon to its lineage.
  - Computes probability by whole-database protein frequency.
- `mode="subtree"`:
  - Ignores annotation frequencies.
  - Computes probability from subtree-size mass over total node count.

### `information_content(tax_id, *, mode="observed")`
Returns `-log(p(tax_id))` for a prepared mode.

### `normalize_taxon_id(tax_id)`
Returns a canonical taxonomy-ID string for lookups.
- Integer-like floats such as `4577.0` are normalized to `"4577"`.
- Empty values and `nan`-like strings normalize to `""`.

### `taxon_names(tax_ids, *, sort=True)`
Returns display names for provided taxonomy IDs.

### `format_taxon_names(tax_ids, *, separator='; ', sort=True)`
Formats names as one string.

## Module-Level Functions

### `load_taxonomy(taxdump_dir, *, names_priority=("scientific name",), quiet=True)`
Convenience constructor returning a `TaxonomyOntology` instance.

### `normalize_taxonomy_id(value)`
Module-level taxonomy-ID normalizer.
- Useful when IDs arrive from pandas/CSV/DB paths as floats or strings.

### `compute_taxon_ic_and_lin_maps(taxonomy, query_tax_ids, subject_tax_ids=None, *, observed_tax_ids=None, observed_mode="observed")`
Builds reusable lookup maps for taxonomy IC and Lin similarity.

Returns:
- `query_taxon_ic_map`: normalized `query_taxonomy_id -> observed IC`
- `taxon_lin_similarity_map`: `(query_taxonomy_id, subject_taxonomy_id) -> Lin similarity`

Behavior:
- Lin similarity always uses `mode="subtree"` so shared non-root clades remain informative.
- Observed IC can be prepared with `observed_mode="whole_db"` directly from the BioData database.
- Observed IC uses `observed_tax_ids` when provided.
- If `observed_tax_ids` is omitted, observed IC is computed from the union of query and subject taxonomy IDs.

### `read_taxonomy_annotations_tsv(path, *, entity_col=0, taxon_col=1, delimiter='\t')`
Reads delimited annotations into `dict[str, set[str]]`.
- Expected default row format: `entity_id<TAB>taxon_id`.
- Skips blank/comment lines and malformed rows.
- De-duplicates taxonomy IDs per entity.

## Notes
- Taxonomy IDs are stored/returned as strings.
- Root handling follows `nodes.dmp` self-parent or parent-missing fallback semantics.
- Taxonomy IC modes are now `observed`, `whole_db`, and `subtree`.
