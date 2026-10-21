"""Batch-wise phenotype mixture clustering and marker-library construction."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
try:
    from umap import UMAP
except Exception:  # pragma: no cover
    UMAP = None
try:
    import scanpy as sc
except Exception:  # pragma: no cover
    sc = None

EPS = 1e-12


def _as_str_series(values: Iterable, default: str = "Unknown") -> list[str]:
    return [str(x) if pd.notna(x) else default for x in values]


def _sort_cluster_ids(cluster_ids: Iterable) -> list[str]:
    def key(x: str):
        x = str(x)
        return (0, int(x)) if x.isdigit() else (1, x)
    return sorted([str(x) for x in cluster_ids], key=key)


def _parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def _marker_weights(marker_params: pd.DataFrame) -> np.ndarray:
    if "configuration_weight" in marker_params:
        w = marker_params["configuration_weight"].astype(float).values
    elif "phenotype_weight" in marker_params:
        w = marker_params["phenotype_weight"].astype(float).values
    elif "marker_weight" in marker_params:
        w = marker_params["marker_weight"].astype(float).values
    elif "separation_score" in marker_params and "mean_posterior_conf" in marker_params:
        w = marker_params["separation_score"].astype(float).values * marker_params["mean_posterior_conf"].astype(float).values
    else:
        status = marker_params.get("status", pd.Series("unimodal", index=marker_params.index)).astype(str)
        w = status.str.startswith("valid").astype(float).values
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(w, 0.0, 1.0)


def _state_label(n_states: int, state: int) -> str:
    if n_states == 1:
        return "unimodal"
    return "low" if state == 0 else "high"


def _posterior_state_enrichment(probs, idx, state_id, prior=1.0):
    other = ~idx
    state_prob = probs[:, state_id]
    alt_prob = probs[:, 1 - state_id]
    a = float(state_prob[idx].sum() + prior)
    b = float(alt_prob[idx].sum() + prior)
    bg = float(state_prob[other].mean()) if other.any() else float(state_prob.mean())
    posterior = float(1.0 - beta_dist.cdf(bg, a, b))
    dominance = float(1.0 - beta_dist.cdf(0.5, a, b))
    mean_in = float(state_prob[idx].mean())
    return posterior, dominance, mean_in, bg


def build_state_matrix(
    result: dict,
    marker_params: pd.DataFrame,
    min_state_prob: float = 0.60,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert state posteriors to hard marker states for summaries only."""
    state_post = result.get("state_posteriors")
    if state_post is None:
        low = np.asarray(result["low_prob"])
        high = np.asarray(result["high_prob"])
        n_states = marker_params.get("n_states", pd.Series(2, index=marker_params.index)).astype(int).values
        state_post = []
        for j, L in enumerate(n_states):
            if L == 1:
                state_post.append(np.ones((high.shape[0], 1)))
            else:
                state_post.append(np.column_stack([low[:, j], high[:, j]]))
    n_cells = state_post[0].shape[0]
    n_markers = len(state_post)
    state = np.full((n_cells, n_markers), -1, dtype=np.int16)
    confidence = np.zeros((n_cells, n_markers), dtype=np.float32)
    for j, post in enumerate(state_post):
        best = np.argmax(post, axis=1)
        conf = np.max(post, axis=1)
        state[:, j] = np.where(conf >= min_state_prob, best, -1)
        confidence[:, j] = conf.astype(np.float32)
    return state, confidence


def _build_cell_cluster_table(
    adata_batch,
    batch_id: str,
    labels: np.ndarray,
    resp: np.ndarray,
    subtype_key: str,
    minor_subset_key: str,
    batch_key: str,
) -> pd.DataFrame:
    best = resp.max(axis=1) if resp.size else np.ones(len(labels))
    second = np.full(len(labels), "", dtype=object)
    if resp.shape[1] > 1:
        order = np.argsort(-resp, axis=1)
        second = order[:, 1].astype(str)
    obs = adata_batch.obs.copy()
    out = pd.DataFrame({
        "cell_barcode": adata_batch.obs_names.astype(str),
        "batch": batch_id,
        "protein_phenotype": labels.astype(str),
        "assignment_confidence": best.astype(float),
        "possible_secondary_type": second,
    })
    if batch_key in obs:
        out[batch_key] = _as_str_series(obs[batch_key].values)
    if subtype_key in obs:
        out[subtype_key] = _as_str_series(obs[subtype_key].values)
    if minor_subset_key in obs:
        out[minor_subset_key] = _as_str_series(obs[minor_subset_key].values)
    return out


def _summarize_cluster_markers(
    batch_id: str,
    labels: np.ndarray,
    result: dict,
    marker_names: np.ndarray,
    marker_params: pd.DataFrame,
    obs: pd.DataFrame,
    subtype_key: str,
    minor_subset_key: str,
    min_cluster_fraction: float,
    min_marker_effect: float,
    max_markers_per_cluster: int,
) -> pd.DataFrame:
    high_prob = np.asarray(result["high_prob"])
    low_prob = np.asarray(result["low_prob"])
    state, confidence = build_state_matrix(result, marker_params)
    weights = _marker_weights(marker_params)
    n_states = marker_params.get("n_states", pd.Series(2, index=marker_params.index)).astype(int).values
    status = marker_params.get("status", pd.Series("unknown", index=marker_params.index)).astype(str).values

    records = []
    for clust in _sort_cluster_ids(np.unique(labels)):
        idx = labels.astype(str) == str(clust)
        other = ~idx
        n_cells = int(idx.sum())
        obs_c = obs.loc[idx]
        subtype_counts = obs_c[subtype_key].astype(str).value_counts() if subtype_key in obs_c else pd.Series(dtype=int)
        minor_counts = obs_c[minor_subset_key].astype(str).value_counts() if minor_subset_key in obs_c else pd.Series(dtype=int)

        selected = []
        positive = []
        negative = []
        details = []
        effects = np.zeros(len(marker_names), dtype=float)
        state_names = []
        for j, marker in enumerate(marker_names):
            if weights[j] <= 0 or n_states[j] == 1:
                state_names.append("unimodal")
                continue
            probs = np.column_stack([low_prob[:, j], high_prob[:, j]])
            low_post, low_dom, low_in, low_bg = _posterior_state_enrichment(probs, idx, 0)
            high_post, high_dom, high_in, high_bg = _posterior_state_enrichment(probs, idx, 1)
            if max(low_post, high_post) < 0.95:
                state_names.append("unrelated")
                continue
            dominant_state = int(high_post > low_post)
            posterior = high_post if dominant_state == 1 else low_post
            dominance = high_dom if dominant_state == 1 else low_dom
            if dominance < 0.99:
                state_names.append("unrelated")
                continue
            mean_in = high_in if dominant_state == 1 else low_in
            mean_out = high_bg if dominant_state == 1 else low_bg
            effect = float((mean_in - mean_out) * weights[j])
            if effect <= 0:
                state_names.append("unrelated")
                continue
            effects[j] = effect
            label = _state_label(int(n_states[j]), dominant_state)
            state_names.append(label)
            details.append((j, marker, label, effect, posterior))
        details = sorted(details, key=lambda x: -x[3])[:max_markers_per_cluster]
        for j, marker, label, effect, posterior in details:
            selected.append(marker)
            if label == "high":
                positive.append(marker)
            elif label == "low":
                negative.append(marker)
        pattern = "; ".join([f"{m} {s}" for _, m, s, _, _ in details])
        records.append({
            "batch": batch_id,
            "cluster": str(clust),
            "phenotype": str(clust),
            "n_cells": n_cells,
            "major_subtype": subtype_counts.index[0] if len(subtype_counts) else "Unknown",
            "major_subtype_fraction": float(subtype_counts.iloc[0] / n_cells) if len(subtype_counts) else np.nan,
            "major_minor_subset": minor_counts.index[0] if len(minor_counts) else "Unknown",
            "major_minor_fraction": float(minor_counts.iloc[0] / n_cells) if len(minor_counts) else np.nan,
            "positive_markers": positive,
            "negative_markers": negative,
            "selected_markers": selected,
            "n_positive_markers": len(positive),
            "n_negative_markers": len(negative),
            "n_selected_markers": len(selected),
            "mean_marker_confidence": float(np.mean([confidence[idx, j].mean() for j, *_ in details])) if details else 0.0,
            "mean_marker_weight": float(np.mean([weights[j] for j, *_ in details])) if details else 0.0,
            "mean_marker_effect": float(np.mean([effect for *_, effect, _ in details])) if details else 0.0,
            "subtype_composition": subtype_counts.to_dict(),
            "minor_subset_composition": minor_counts.to_dict(),
            "typical_protein_peak_pattern": pattern,
        })
    return pd.DataFrame(records)


def _save_step3_outputs(output_dir, batch_id, cell_clusters, cluster_library, marker_params, result, marker_names):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_clusters.to_csv(output_dir / f"{batch_id}_cell_clusters.csv", index=False)
    cluster_library.to_csv(output_dir / f"{batch_id}_cluster_marker_library_valid.csv", index=False)
    marker_params.to_csv(output_dir / f"{batch_id}_marker_params.csv", index=False)
    arrays = {
        "high_prob": result["high_prob"],
        "low_prob": result["low_prob"],
        "unimodal_prob": result.get("unimodal_prob", np.zeros_like(result["high_prob"])),
        "empty_prob": result.get("empty_prob", np.zeros_like(result["high_prob"])),
        "state_argmax": result.get("state_argmax", np.zeros_like(result["high_prob"], dtype=np.int16)),
        "phenotype_posterior": result.get("phenotype_posterior", np.ones((result["high_prob"].shape[0], 1))),
        "phenotype_assignment": result.get(
            "phenotype_assignment",
            result.get(
                "cluster_assignment",
                np.zeros(result["high_prob"].shape[0], dtype=str),
            ),
        ),
        "marker_names": np.asarray(marker_names, dtype=str),
    }
    if "theta_flat" in result:
        arrays["theta_flat"] = result["theta_flat"]
    if "pi" in result:
        arrays["pi"] = result["pi"]
    np.savez_compressed(output_dir / f"{batch_id}_protein_probabilities.npz", **arrays)


def summarize_phenotypes(
    adata_batch,
    result: dict,
    marker_names,
    batch_id: str,
    subtype_key: str = "subtype",
    minor_subset_key: str = "minor_subset",
    batch_key: str = "COMBAT_ID",
    min_cluster_fraction: float = 0.3,
    min_marker_effect: float = 0.10,
    max_markers_per_cluster: int = 25,
    save_csv: bool = True,
    output_dir: str | Path = "results",
    output_path: str | Path | None = None,
):
    """Build phenotype summaries without graph clustering."""
    marker_names = np.asarray(marker_names, dtype=str)
    marker_params = result["marker_params"].copy()
    labels = np.asarray(
        result.get("phenotype_assignment", result.get("cluster_assignment", None))
    )
    resp = np.asarray(result.get("phenotype_posterior", None))
    if labels is None or labels.size == 0:
        resp = np.asarray(result.get("posterior_likelihood"))
        labels = np.argmax(resp, axis=1).astype(str)
    labels = labels.astype(str)
    if resp is None or resp.size == 0:
        unique = _sort_cluster_ids(np.unique(labels))
        code = {c: i for i, c in enumerate(unique)}
        resp = np.zeros((labels.size, len(unique)), dtype=float)
        resp[np.arange(labels.size), [code[x] for x in labels]] = 1.0
    clusters = _sort_cluster_ids(np.unique(labels))

    state_post = result.get("state_posteriors")
    X_state = (
        np.concatenate(state_post, axis=1)
        if state_post is not None
        else resp.astype(float)
    )
    if sc is not None:
        adata_cluster = sc.AnnData(X_state.astype(np.float32))
        adata_cluster.obs_names = adata_batch.obs_names.astype(str)
    else:
        adata_cluster = {
            "X": X_state.astype(np.float32),
            "obs": pd.DataFrame(index=adata_batch.obs_names.astype(str)),
        }
    obs_target = adata_cluster.obs if sc is not None else adata_cluster["obs"]
    obs_target["protein_phenotype"] = labels
    if subtype_key in adata_batch.obs:
        obs_target[subtype_key] = _as_str_series(adata_batch.obs[subtype_key].values)
    if minor_subset_key in adata_batch.obs:
        obs_target[minor_subset_key] = _as_str_series(adata_batch.obs[minor_subset_key].values)
    if batch_key in adata_batch.obs:
        obs_target[batch_key] = _as_str_series(adata_batch.obs[batch_key].values)

    cell_clusters = _build_cell_cluster_table(
        adata_batch, batch_id, labels, resp, subtype_key, minor_subset_key, batch_key
    )
    cluster_library = _summarize_cluster_markers(
        batch_id,
        labels,
        result,
        marker_names,
        marker_params,
        cell_clusters,
        subtype_key,
        minor_subset_key,
        min_cluster_fraction,
        min_marker_effect,
        max_markers_per_cluster,
    )
    if save_csv:
        if output_path is not None:
            output_dir = Path(output_path).parent
        _save_step3_outputs(output_dir, batch_id, cell_clusters, cluster_library, marker_params, result, marker_names)
    return {
        "adata_cluster": adata_cluster,
        "cell_clusters": cell_clusters,
        "cluster_marker_library": cluster_library,
        "cluster_marker_library_valid": cluster_library,
        "marker_params": marker_params,
        "protein_probabilities": {
            "high_prob": result["high_prob"],
            "low_prob": result["low_prob"],
            "unimodal_prob": result.get("unimodal_prob"),
            "empty_prob": result.get("empty_prob"),
            "marker_names": marker_names,
        },
        "valid_mask": _marker_weights(marker_params) > 0,
        "valid_marker_names": marker_names[_marker_weights(marker_params) > 0],
        "marker_weights": _marker_weights(marker_params),
        "state_matrix": build_state_matrix(result, marker_params)[0],
        "confidence_matrix": build_state_matrix(result, marker_params)[1],
        "assignment_probability": resp,
        "clusters": clusters,
    }


def plot_phenotypes(
    cluster_result,
    batch_id,
    color=("protein_phenotype", "subtype", "minor_subset"),
    size=8,
    figsize=None,
    method="umap",
    random_state=0,
):
    """Plot phenotype and ground truth on one post-hoc protein embedding."""
    adata_cluster = cluster_result["adata_cluster"]
    if sc is not None and hasattr(adata_cluster, "X"):
        X = np.asarray(adata_cluster.X)
        obs = adata_cluster.obs
    else:
        X = np.asarray(adata_cluster["X"])
        obs = adata_cluster["obs"]
    if method == "umap" and UMAP is not None and X.shape[1] > 1:
        emb = UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.25,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(X)
        axis_names = ("protein state UMAP1", "protein state UMAP2")
    else:
        emb = (
            PCA(n_components=2, random_state=random_state).fit_transform(X)
            if X.shape[1] > 1
            else np.column_stack([np.arange(X.shape[0]), np.zeros(X.shape[0])])
        )
        axis_names = ("protein state PC1", "protein state PC2")
    keys = [key for key in color if key in obs]
    if not keys:
        raise ValueError("None of the requested color columns are available.")
    if figsize is None:
        figsize = (5 * len(keys), 4)
    fig, axes = plt.subplots(
        1,
        len(keys),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, key in zip(axes[0], keys):
        labels = obs[key].astype(str).values
        values = _sort_cluster_ids(np.unique(labels))
        cmap = plt.get_cmap("tab20", max(len(values), 1))
        for index, label in enumerate(values):
            mask = labels == label
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=size,
                color=cmap(index),
                label=label,
                alpha=0.75,
                linewidths=0,
            )
        ax.set_title(f"{batch_id} | {key}")
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.legend(
            title=key,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            markerscale=1.5,
        )
    plt.tight_layout()
    return axes[0]


def evaluate_phenotypes(adata_cluster, truth_key="subtype", ignore_unknown=True, display_table=True):
    """Evaluate phenotype components against a selected groundtruth column."""
    obs = adata_cluster.obs if hasattr(adata_cluster, "obs") else adata_cluster["obs"]
    if truth_key not in obs:
        raise ValueError(f"{truth_key!r} is not available in adata_cluster.obs.")
    clusters = obs["protein_phenotype"].astype(str).values
    subtypes = np.array(_as_str_series(obs[truth_key].values))
    keep = subtypes != "Unknown" if ignore_unknown else np.ones_like(subtypes, dtype=bool)
    clusters_eval = clusters[keep]
    subtypes_eval = subtypes[keep]
    n_correct = 0
    summary = []
    for clust in _sort_cluster_ids(np.unique(clusters_eval)):
        mask = clusters_eval == clust
        counts = pd.Series(subtypes_eval[mask]).value_counts()
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue
        n_correct += int(counts.iloc[0])
        summary.append({
            "cluster": clust,
            "phenotype": clust,
            "n_cells": n_cells,
            "major_subtype": counts.index[0],
            "major_count": int(counts.iloc[0]),
            "major_subtype_fraction": float(counts.iloc[0] / n_cells),
        })
    metrics = {
        "purity": n_correct / max(len(clusters_eval), 1),
        "ARI": adjusted_rand_score(subtypes_eval, clusters_eval),
        "NMI": normalized_mutual_info_score(subtypes_eval, clusters_eval),
        "n_clusters": len(np.unique(clusters_eval)),
        "n_cells_used": len(clusters_eval),
    }
    summary_df = pd.DataFrame(summary)
    contingency_df = pd.crosstab(
        pd.Series(clusters_eval, name="phenotype"),
        pd.Series(subtypes_eval, name=truth_key),
    )
    if display_table:
        print("Phenotype mixture evaluation")
        print(f"Purity:    {metrics['purity']:.4f}")
        print(f"ARI:       {metrics['ARI']:.4f}")
        print(f"NMI:       {metrics['NMI']:.4f}")
        print(f"n_clusters:{metrics['n_clusters']}")
        print(f"n_cells:   {metrics['n_cells_used']}")
        display = globals().get("display")
        if display is not None:
            display(summary_df)
            display(contingency_df)
        else:
            print(summary_df)
            print(contingency_df)
    return metrics, summary_df, contingency_df


cluster_one_batch = summarize_phenotypes
plot_cluster_umap = plot_phenotypes


def evaluate_clustering(
    adata_cluster,
    subtype_key="subtype",
    cluster_key="protein_phenotype",
    ignore_unknown=True,
    display_table=True,
):
    del cluster_key
    return evaluate_phenotypes(
        adata_cluster,
        truth_key=subtype_key,
        ignore_unknown=ignore_unknown,
        display_table=display_table,
    )
