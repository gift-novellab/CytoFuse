"""CyTOF preprocessing and phenotype fitting."""

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

DEFAULT_DATA = PROJECT_ROOT / "data" / "cytof_example.h5ad"
TECHNICAL_MARKERS = {"Ir191Di", "Ir193Di", "Event_length", "IgG"}
AUXILIARY_MARKERS = {"IgA", "IgM", "CD66", "Siglec_8"}


def load_sample(
    sample_id,
    data_path=DEFAULT_DATA,
    max_cells=25_000,
    random_state=0,
):
    """Load one sample and optionally take a reproducible random subset."""
    source = sc.read_h5ad(data_path, backed="r")
    mask = source.obs["sample_id"].astype(str).values == str(sample_id)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        source.file.close()
        raise ValueError(f"Unknown sample_id: {sample_id}")
    if max_cells is not None and indices.size > max_cells:
        rng = np.random.default_rng(random_state)
        indices = np.sort(rng.choice(indices, size=max_cells, replace=False))
    adata = source[indices].to_memory()
    source.file.close()
    return adata


def phenotype_markers(adata, auxiliary_weight=0.15):
    """Return CyTOF markers, indices, roles, and platform weights."""
    if "marker_class" not in adata.var:
        raise ValueError("adata.var['marker_class'] is required.")
    names = np.asarray(adata.var_names, dtype=str)
    marker_class = adata.var["marker_class"].astype(str).values
    keep = ~np.isin(names, list(TECHNICAL_MARKERS))
    kept_names = names[keep]
    kept_class = marker_class[keep]
    roles = np.where(
        kept_class == "type",
        "phenotype",
        np.where(np.isin(kept_names, list(AUXILIARY_MARKERS)), "auxiliary", "excluded"),
    )
    use = roles != "excluded"
    weights = np.where(roles[use] == "phenotype", 1.0, float(auxiliary_weight))
    return kept_names[use], np.flatnonzero(keep)[use], roles[use], weights


def dense_matrix(adata, columns):
    X = adata.X[:, columns]
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def fit_sample(
    adata,
    sample_id,
    output_dir,
    random_state=0,
    save=True,
    k_init=40,
    min_component_weight=0.025,
    min_bic_gain=10.0,
    min_separation=1.2,
    min_bhattacharyya=0.08,
    min_mean_posterior_conf=0.75,
    shrinkage=8.0,
    alpha_pi=0.08,
    min_component_cells=80,
    merge_threshold=0.012,
    redundant_merge_threshold=0.025,
    min_marker_component_weight=0.01,
    min_marker_posterior_confidence=0.60,
):
    """Fit one CyTOF sample using its existing arcsinh expression."""
    marker_names, marker_idx, marker_roles, platform_weights = phenotype_markers(adata)
    X = dense_matrix(adata, marker_idx)
    result = fit_phenotypes(
        X,
        marker_names=marker_names,
        K_init=k_init,
        min_component_weight=min_component_weight,
        min_component_weight_3=0.07,
        min_bic_gain=min_bic_gain,
        bic_gain_23=100.0,
        icl_gain_23=30.0,
        min_separation=min_separation,
        min_bhattacharyya=min_bhattacharyya,
        min_mean_posterior_conf=min_mean_posterior_conf,
        jitter_noise_sd=1e-4,
        jitter_mode="zero",
        shrinkage=shrinkage,
        alpha_theta=0.2,
        alpha_pi=alpha_pi,
        warmup_iter=5,
        min_component_cells=min_component_cells,
        min_component_fraction=0.0008,
        merge_threshold=merge_threshold,
        redundant_merge_threshold=redundant_merge_threshold,
        min_marker_component_weight=min_marker_component_weight,
        min_marker_posterior_confidence=min_marker_posterior_confidence,
        marker_weight_multipliers=platform_weights,
        random_state=random_state,
    )
    role_map = dict(zip(marker_names, marker_roles))
    result["marker_params"]["marker_role"] = result["marker_params"]["marker"].map(role_map)
    minor_key = "fine_cluster_id" if "fine_cluster_id" in adata.obs else "major_cell_type"
    summary = summarize_phenotypes(
        adata_batch=adata,
        result=result,
        marker_names=marker_names,
        batch_id=str(sample_id),
        subtype_key="major_cell_type",
        minor_subset_key=minor_key,
        batch_key="batch",
        min_cluster_fraction=0.3,
        min_marker_effect=0.10,
        max_markers_per_cluster=20,
        save_csv=save,
        output_dir=output_dir,
    )
    metrics = {}
    for truth_key in ("major_cell_type", "fine_cluster_id"):
        if truth_key not in summary["adata_cluster"].obs:
            continue
        values, table, contingency = evaluate_phenotypes(
            summary["adata_cluster"],
            truth_key=truth_key,
            ignore_unknown=True,
            display_table=False,
        )
        metrics[truth_key] = values
        metrics[f"{truth_key}_summary"] = table
        metrics[f"{truth_key}_contingency"] = contingency
    return {
        "adata": adata,
        "X": X,
        "marker_names": marker_names,
        "marker_roles": marker_roles,
        "platform_weights": platform_weights,
        "model_result": result,
        "cluster_result": summary,
        "metrics": metrics,
    }
