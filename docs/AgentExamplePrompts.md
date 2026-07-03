# Agent Example Prompts

```text
Write a Python script that accepts a FASTA file, generates mean-pooled ESM-2 embeddings for every model returned by available_generator_models("esm2"), runs every ready PEER protein-level dataset with a LinearProbe across all available layers, and writes a CSV leaderboard with model name, dataset id, layer, metric, and score.
```

Use these prompts to exercise CBBIO end to end. They are written for coding agents that should use the public `CBBIO` namespace, inspect the focused docs first, and avoid scanning unrelated modules. Matching code examples live in [examples/](examples/).

## Prompt Set

### 1. ESM-2 PEER Sweep

```text
Write a CLI script named sweep_esm2_peer.py. It should take --fasta, --data-dir, --out-dir, --device, and --max-batch-tokens. Use CBBIO.available_generator_models("esm2") to discover ESM-2 models, generate mean-pooled embeddings with CBBIO.Generator, CBBIO.FastaBatcher, CBBIO.EmbeddingWriter, and CBBIO.run_embedding_generation, then run all ready PEER protein-level datasets with CBBIO.LinearProbe. Sweep every generated layer. Save per-task metrics to peer_esm2_sweep.csv and save one HDF5 embedding file per model. Include argparse, type annotations, logging, and graceful skips for datasets that cannot be downloaded.
```

### 2. Wnt Signaling Graph

```text
Write a script that connects to the BioData PostgreSQL database with CBBIO.connect, finds all proteins annotated to Wnt signaling or its descendant GO terms, retrieves or generates ProstT5 mean-pooled embeddings for those proteins, computes pairwise cosine distances, and builds a NetworkX graph where edges connect proteins below a configurable distance threshold. Save graph.graphml and nodes.tsv with UniProt id, organism, taxonomy id, GO terms, and embedding metadata. Use public CBBIO APIs where available and keep database SQL isolated in one typed helper.
```

### 3. CAFA Transfer Baseline

```text
Build a reproducible CAFA baseline pipeline. Load CAFA5 and CAFA6 datasets with CBBIO.load_dataset, generate protein-level embeddings for ESM-C and AMPLIFY, train no model with CBBIO.TransferProbe, and evaluate GO protein-centric metrics including weighted Fmax when information accretion weights are available. The script should compare search backends "numpy", "faiss_cpu", and "torch_gpu" when installed, skip missing optional dependencies with clear messages, and write one JSON report per dataset, model, layer, and backend.
```

### 4. Residue PTM Probe Matrix

```text
Create a residue-level probing benchmark for phosphorylation. Load dbPTM, PhosphoELM, and MusiteDeep residue datasets through CBBIO catalog or source loaders, generate unpooled residue embeddings with ProtT5, Ankh3, and ESM-2, and train both CBBIO.LinearProbe and CBBIO.MlpProbe for every requested layer. Validate that embedding matrix lengths match sequence lengths before training. Save metrics, confusion counts, and skipped protein ids. Add focused pytest tests for the length-validation helper.
```

### 5. GO Semantic Neighbor Audit

```text
Write an analysis notebook or script that samples proteins from the BioData database, retrieves nearest neighbors with CBBIO.BioDataClient.find_nearest_neighbors, loads go-basic.obo with CBBIO.load_go, prepares GO term counts from database annotations, and compares embedding distance with GO semantic similarity using Resnik, Lin, Schlicker, and Wang. Produce a Spearman correlation table by embedding type, model layer, ontology aspect, and distance metric. Save the result as markdown and CSV.
```

### 6. Taxonomy-Aware Embedding Report

```text
Write a report generator that groups proteins by taxonomy lineage. Use CBBIO.load_taxonomy to load NCBI taxdump, CBBIO.connect to fetch protein taxonomy ids and embeddings, and CBBIO.Taxonomy.compute_taxon_ic_and_lin_maps to compute taxonomy information content and pairwise Lin similarity for query species. For each species pair, compare taxonomy similarity with mean embedding distance. Output a concise markdown report with tables and plots, and keep plotting code optional if matplotlib is unavailable.
```

### 7. Embedding Store Converter

```text
Write a robust conversion tool for embedding files. It should accept pickle, NumPy, or HDF5 inputs through CBBIO.load_embedding_records, optionally mean-pool residue-level records with CBBIO.mean_pool_embedding_record, filter by ids, layer_index, model_reference, and pool_method, then write HDF5 with CBBIO.save_embedding_records_h5. Include a dry-run mode that reports record count, payload shapes, layer distribution, model references, and estimated output size without writing files.
```

### 8. Custom Dataset Adapter

```text
Add support for a new residue-level CSV dataset with columns protein_id, sequence, split, and binding_site_mask. Implement a typed loader that returns CBBIO.ResidueDataset, register it in the probing dataset catalog as status "ready", document it in the relevant docs, and add unit tests with a tiny fixture. Follow docs/AddingProbingDataSource.md and keep imports in tests and examples from the public CBBIO namespace unless an internal adapter base class is required.
```

### 9. New Embedding Model Family

```text
Add a new embedding model family named "mymodel" under CBBIO/embeddings/models. Follow docs/ModelSpecificEmbeddingModules.md. Implement preprocessor, tokenizer adapter, model adapter, postprocessor, and MyModelEmbeddingGenerator. Define GENERATOR_CLASS, DEFAULT_MODEL_NAME, FAMILY_MODELS, and SUPPORTED_POOLERS. Import optional heavy dependencies lazily and raise CBBIO.EmbeddingDependencyError with install guidance. Export the generator from CBBIO, register it in the factory, update docs, and add tests for factory discovery, dependency errors, layer selection, poolers, and metadata.
```

### 10. Cross-Modal Protein Similarity Explorer

```text
Build a script that takes a query FASTA and a BioData database connection, aligns query sequences to database sequences with CBBIO.align_sequences, generates query embeddings with ESM-2 and ProteinGLM, retrieves database nearest neighbors for matching embedding types, and combines sequence identity, embedding distance, GO similarity, and taxonomy Lin similarity into a ranked candidate table. The output should include one TSV for machine use and one markdown report explaining the top candidates and any missing annotations.
```

## Coverage Map

| Prompt | Main CBBIO coverage |
|---|---|
| 1 | Embedding generation, model discovery, PEER probing, layer sweeps |
| 2 | BioData database, GO annotations, ProstT5 embeddings, graph analysis |
| 3 | CAFA datasets, TransferProbe, GO metrics, search backends |
| 4 | Residue datasets, PTM sources, unpooled embeddings, linear and MLP probes |
| 5 | Nearest-neighbor search, GO ontology, semantic similarity |
| 6 | Taxonomy ontology, taxonomy IC, embedding distance analysis |
| 7 | Embedding IO, HDF5, pooling, file conversion |
| 8 | Dataset catalog extension, residue dataset loading, tests, docs |
| 9 | New model adapter, factory registration, lazy dependencies |
| 10 | Sequence alignment, embeddings, database search, GO, taxonomy |

## Exceptions

| Exception | When raised |
|---|---|
| `EmbeddingDependencyError` | Optional model, HDF5, or search dependency is missing |
| `EmbeddingInputError` | FASTA, layer, pooler, dataset, or embedding payload is invalid |
| `BioDataError` | Database connection, configuration, or query operation fails |
| `GOError` | GO ontology loading or similarity operation fails |
| `TaxonomyError` | Taxonomy loading or similarity operation fails |
| `SequenceSimilarityError` | Sequence alignment operation fails |
