# Ontologies

```python
from CBBIO import load_go, load_taxonomy

go = load_go("go-basic.obo")
taxonomy = load_taxonomy("taxdump")
```

## Gene Ontology

::: CBBIO.GOOntology
    options:
      members: null

::: CBBIO.load_go

::: CBBIO.read_annotations_tsv

## Taxonomy

::: CBBIO.TaxonomyOntology
    options:
      members: null

::: CBBIO.load_taxonomy

::: CBBIO.read_taxonomy_annotations_tsv

## Exceptions

::: CBBIO.GOError

::: CBBIO.GOTermNotFoundError

::: CBBIO.GOCountsNotPreparedError

::: CBBIO.TaxonomyError

::: CBBIO.TaxonNotFoundError

::: CBBIO.TaxonCountsNotPreparedError
