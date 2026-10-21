# CytoFuse

CytoFuse is a marker-state based framework for clustering and integrating
single-cell protein measurements across batches or partially overlapping panels.
It first fits a parameter-shared mixture model in the full marker space within
each batch, converts clusters into qualitative marker-state signatures, and then
builds a consensus atlas by matching compatible signatures. Gaussian marker
parameters are never shared across batches, and cross-batch alignment uses only
the resulting reliable qualitative configurations. Marker participation in
batch-local clustering is soft and deliberately broader than marker eligibility
for qualitative `+/-` configuration calls.

This repository contains the core method code and anonymized example datasets
for exercising the CITE-seq, CyTOF, and FACS workflows. Benchmark outputs and
paper result files are intentionally not included.

## Repository layout

- `cytofuse/methods/`: shared model fitting, marker-state summaries, clustering,
  and atlas utilities.
- `cytofuse/citeseq/`: CITE-seq workflow.
- `cytofuse/cytof/`: CyTOF workflow.
- `cytofuse/facs/`: FACS panel workflow.
- `cytofuse/crosstech/`: signature matching and consensus construction helpers.
- `data/citeseq_example.h5ad.gz`: anonymized CITE-seq example data.
- `data/cytof_example.h5ad.gz`: anonymized CyTOF example data.
- `data/facs_example.h5ad.gz`: anonymized FACS example data.
- `run_cytofuse.py`: minimal command-line launcher for the bundled examples.

Generated outputs are written to `outputs/` and are ignored by git.

## Installation

```bash
conda create -n cytofuse python=3.10
conda activate cytofuse
pip install -r requirements.txt
```

## Run bundled examples

Examples write generated files to `outputs/`, which is ignored by git.
Decompress the bundled data before running:

```bash
gunzip data/*.h5ad.gz
```

```bash
python run_cytofuse.py citeseq --force --extra-args --batches batch_000 batch_001
python run_cytofuse.py cytof --force --extra-args --samples batch_000
python run_cytofuse.py facs --force --extra-args --panels T --max-cells 50000
```

## Optional GPU acceleration

The validated reference implementation remains the default CPU path above.
No method parameters, Python filenames, or output filenames change when GPU
acceleration is enabled. On Linux or WSL 2 with a compatible NVIDIA GPU,
install RAPIDS cuML for the local CUDA version, then prefix the same command
with its zero-code-change accelerator:

```bash
CUDA_VISIBLE_DEVICES=0 CUML_ACCEL_LOG_LEVEL=info \
python -m cuml.accel run_cytofuse.py citeseq --force \
  --extra-args --batches batch_000 batch_001
```

The same mechanism can be used with the batch-local atlas workflow without
renaming any module or result file:

```bash
CUDA_VISIBLE_DEVICES=0 CUML_ACCEL_ENABLED=1 CUML_ACCEL_LOG_LEVEL=info \
python -m forcitepool run --input data/citeseq_example.h5ad \
  --output outputs/citeseq_example --batch-key batch --k-init 40
```

`CUML_ACCEL_LOG_LEVEL=info` reports which operations ran on the GPU. Operations
not supported by cuML automatically use their original CPU implementation. The
CPU command should be used when bitwise reproduction of the validated result is
required; GPU and CPU numerical implementations can produce small differences
despite using the same model and random seed. See the
[RAPIDS cuML accelerator documentation](https://docs.rapids.ai/api/cuml/stable/cuml-accel/)
for supported CUDA, Python, and scikit-learn versions.

Main output files include:

- `batch_metrics.csv`: per-batch clustering and evaluation summaries.
- `*_cell_clusters.csv`: per-cell local phenotype assignments.
- `*_cluster_marker_library_valid.csv`: marker-state signatures for local clusters.
- `platform_consensus_nodes.csv`: consensus phenotypes across batches.
- `platform_consensus_cells.csv`: per-cell consensus assignments.

## Input format

The input should be an `.h5ad` file where rows are cells and columns are protein
markers. The example workflow expects:

- `adata.X`: raw or transformed protein abundance matrix.
- `adata.obs["batch"]`: batch identifier.
- Optional labels for evaluation: `subtype` and `minor_subset`.

## Note

The bundled example data is anonymized for demonstration. It should be used to
test workflow mechanics, not as a benchmark result bundle.
