# ForCitePool

ForCitePool is an additive CITE-seq extension for CytoFuse. It learns
batch-local protein clusters, aligns compatible clusters across batches, and
builds a confidence-weighted protein phenotype forest.

This forest describes marker-profile similarity. It is **not** a developmental
lineage or evolutionary tree: marker names and cell-type annotations do not
impose biological relationships.

## Install

Place `forcitepool/` directly inside a CytoFuse checkout:

```text
CytoFuse/
├── cytofuse/
└── forcitepool/
```

From the CytoFuse root:

```bash
python -m pip install -r requirements.txt
python -m forcitepool --help
```

## Run

The input is an `.h5ad` or `.h5ad.gz` file. By default, protein values come
from `adata.X`, marker names from `adata.var_names`, and batch labels from
`adata.obs["batch"]`.

```bash
python -m forcitepool run \
  --input path/to/data.h5ad \
  --output path/to/output
```

All batches are used automatically. A different batch column or an intentional
subset can be selected with:

```bash
python -m forcitepool run \
  --input path/to/data.h5ad \
  --output path/to/output \
  --batch-key donor \
  --batches donor_A donor_C
```

Do not pass `--batches` when the full dataset should be analyzed. Arbitrary
batch labels, including spaces and Unicode, are supported.

### Protein location and normalization

Use one of these alternatives when proteins are not stored in `adata.X`:

```bash
# AnnData layer; names are adata.var_names
python -m forcitepool run ... --protein-layer ADT

# DataFrame in obsm; column names are used
python -m forcitepool run ... --protein-obsm-key protein

# ndarray in obsm; names are stored in uns
python -m forcitepool run ... \
  --protein-obsm-key protein \
  --protein-names-key protein_names
```

Raw nonnegative values receive per-cell log1p CLR normalization. Use
`--normalization none` for already transformed values. Run
`python -m forcitepool run --help` for all thresholds and model options.

## Evaluate

Ground-truth labels are optional and are used only after tree construction:

```bash
python -m forcitepool evaluate --run-dir path/to/output
```

For datasets with different annotation columns:

```bash
python -m forcitepool evaluate \
  --run-dir path/to/output \
  --truth-keys coarse_type fine_type \
  --tree-truth-key fine_type
```

## Method

1. Normalize and robustly standardize every batch independently. Each batch
   independently decides whether each marker is unimodal or reliably bimodal
   and estimates its own Gaussian parameters. No cross-batch peak mapping,
   quantile matching, or peak-separation transform is performed.
2. Fit phenotype mixtures independently in every batch. Marker-state Gaussian
   parameters are shared only among phenotype components inside that batch.
3. Call each cluster-marker identity state as positive (`+`), negative (`-`),
   or uncertain (`?`) using that batch's marker model, posterior dominance, and
   within-cluster support. A marker that is unimodal or unreliable in one batch
   remains unknown there; it is not forced to adopt another batch's states.
4. Align local clusters using only marker states that are reliably observed in
   both configurations. Unknown states provide neither matches nor conflicts.
   Pairs require no observed conflicts, sufficient shared states, and reciprocal
   configuration coverage. A coarse configuration with too little evidence is
   retained as unresolved instead of being forced into a fine configuration.
5. Merge the already aligned leaves bottom-up by weighted common positive
   markers, then
   weighted common negative markers, then fewer weighted conflicts.
6. Stop at unresolved ties instead of forcing a single tree.

Marker votes are weighted by confidence and downweighted when nearly
ubiquitous. Parent nodes retain only states shared by both children. Wilson
lower bounds, score tolerances, minimum evidence, and cache fingerprints reduce
sensitivity to weak or small-sample evidence.

## Outputs

```text
output/
├── batch_models/
├── tables/
├── evaluation/
├── run_config.json
└── run.log
```

The main tables are:

- `cluster_marker_states_long.csv`: marker states and confidence diagnostics;
- `alignment_marker_reliability.csv`: per-marker callable-batch and
  callable-cluster coverage;
- `alignment_membership.csv`: local clusters assigned to aligned leaves;
- `alignment_pair_audit.csv`: accepted and rejected alignment candidates;
- `taxonomy_nodes.csv`, `taxonomy_edges.csv`, `taxonomy_merges.csv`;
- `taxonomy_pair_audit.csv`: merge scores and tie decisions;
- `cell_taxonomy_assignments.csv`: leaf and tree path for each cell.

Evaluation adds agreement metrics and figures. Ground-truth annotations never
affect fitting, alignment, or tree construction.

## Python API and development

```python
from forcitepool import (
    StateThresholds,
    align_clusters,
    build_cluster_marker_states,
    build_taxonomy,
)
```

Run the self-contained test suite from the CytoFuse root:

```bash
python -m unittest forcitepool.tests -v
```

The package is intentionally compact:

```text
forcitepool/
├── run.py          # workflow and input handling
├── tree.py         # states, alignment, and forest construction
├── evaluate.py     # post-hoc labels, metrics, and plots
├── tests.py
└── README.md
```
