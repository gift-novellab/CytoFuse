"""FACS panel preprocessing and phenotype fitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytofuse.methods.cluster import evaluate_phenotypes, summarize_phenotypes
from cytofuse.methods.paramshared import fit_phenotypes

DEFAULT_PANELS = {
    "T": PROJECT_ROOT / "data" / "facs_example.h5ad",
}


def dense_matrix(adata):
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def robust_transform(X):
    """Use raw/arcsinh-like FACS intensities without count normalization."""
    X = np.asarray(X, dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=np.nanmax(X[np.isfinite(X)]), neginf=0.0)


def panel_id_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    for name in DEFAULT_PANELS:
        if name.lower() in stem.lower():
            return name
    return stem.replace("CBD_FACS_", "")


def sample_indices(adata, max_cells=300_000, sample_key="key_pbmc_combat_id", random_state=0):
    n = adata.n_obs
    if max_cells is None or max_cells <= 0 or n <= max_cells:
        return np.arange(n, dtype=int)
    rng = np.random.default_rng(random_state)
    if sample_key not in adata.obs:
        return np.sort(rng.choice(n, size=max_cells, replace=False))
    values = adata.obs[sample_key].astype(str).to_numpy()
    groups = pd.Series(values).groupby(values).groups
    per_group = max(1, int(np.ceil(max_cells / max(len(groups), 1))))
    chosen = []
    for _, idx in groups.items():
        idx = np.asarray(list(idx), dtype=int)
        take = min(len(idx), per_group)
        chosen.append(rng.choice(idx, size=take, replace=False) if take < len(idx) else idx)
    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if chosen.size > max_cells:
        chosen = rng.choice(chosen, size=max_cells, replace=False)
    return np.sort(chosen.astype(int))


def fit_panel(
    input_path: str | Path,
    panel_id: str | None,
    output_dir: str | Path,
    max_cells: int | None = 300_000,
    sample_key: str = "key_pbmc_combat_id",
    k_init: int = 20,
    max_iter: int = 50,
    min_bhattacharyya: float = 0.08,
    min_marker_component_weight: float = 0.01,
    min_marker_posterior_confidence: float = 0.60,
    random_state: int = 0,
):
    """Fit one FACS panel as a self-contained local fingerprint source."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    panel_id = panel_id or panel_id_from_path(input_path)
    source = sc.read_h5ad(input_path, backed="r")
    idx = sample_indices(source, max_cells=max_cells, sample_key=sample_key, random_state=random_state)
    adata = source[idx].to_memory()
    source.file.close()

    X = robust_transform(dense_matrix(adata))
    marker_names = np.asarray(adata.var_names, dtype=str)
    result = fit_phenotypes(
        X,
        marker_names=marker_names,
        K_init=k_init,
        max_iter=max_iter,
        min_component_weight=0.02,
        min_component_weight_3=0.04,
        min_bic_gain=8.0,
        bic_gain_23=30.0,
        icl_gain_23=10.0,
        min_separation=1.0,
        min_bhattacharyya=min_bhattacharyya,
        min_mean_posterior_conf=0.70,
        jitter_noise_sd=1e-4,
        jitter_mode="zero",
        shrinkage=15.0,
        alpha_theta=0.2,
        alpha_pi=0.2,
        warmup_iter=5,
        min_component_cells=200,
        min_component_fraction=0.001,
        merge_threshold=0.035,
        redundant_merge_threshold=0.07,
        min_marker_component_weight=min_marker_component_weight,
        min_marker_posterior_confidence=min_marker_posterior_confidence,
        random_state=random_state,
    )
    summary = summarize_phenotypes(
        adata_batch=adata,
        result=result,
        marker_names=marker_names,
        batch_id=panel_id,
        subtype_key="key_new_classification",
        minor_subset_key="key_severity",
        batch_key="panel",
        min_cluster_fraction=0.25,
        min_marker_effect=0.08,
        max_markers_per_cluster=20,
        save_csv=True,
        output_dir=output_dir,
    )
    params = result["marker_params"].copy()
    confidence = np.asarray(result["phenotype_confidence"])
    row = {
        "panel": panel_id,
        "input": str(input_path),
        "status": "complete",
        "source_n_cells": int(source.n_obs if hasattr(source, "n_obs") else adata.n_obs),
        "n_cells_used": int(adata.n_obs),
        "n_markers": int(adata.n_vars),
        "markers": "; ".join(marker_names.tolist()),
        "n_phenotypes": int(len(result["pi"])),
        "n_unimodal": int((params["n_states"] == 1).sum()),
        "n_bimodal": int((params["n_states"] == 2).sum()),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "min_phenotype_cells": int(np.min(result["phenotype_counts"])),
        "max_phenotype_cells": int(np.max(result["phenotype_counts"])),
        "n_iterations": int(result["n_iter"]),
    }
    for truth_key in ("key_new_classification", "key_severity"):
        if truth_key in summary["adata_cluster"].obs:
            values, _, _ = evaluate_phenotypes(
                summary["adata_cluster"],
                truth_key=truth_key,
                ignore_unknown=True,
                display_table=False,
            )
            row.update({f"{truth_key}_{k.lower()}": v for k, v in values.items()})
    return row
