"""Cross-batch phenotype alignment for heterogeneous marker states."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

EPS = 1e-12


def parse_marker_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def _sort_ids(ids):
    def key(x):
        x = str(x)
        return (0, int(x)) if x.isdigit() else (1, x)
    return sorted([str(x) for x in ids], key=key)


def _marker_weight_table(marker_params):
    mp = marker_params.copy()
    if "configuration_weight" in mp:
        mp["marker_weight"] = mp["configuration_weight"].astype(float)
    elif "phenotype_weight" in mp:
        mp["marker_weight"] = mp["phenotype_weight"].astype(float)
    elif "marker_weight" not in mp:
        status = mp.get("status", pd.Series("unimodal", index=mp.index)).astype(str)
        mp["marker_weight"] = status.str.startswith("valid").astype(float)
    mp["marker_weight"] = mp["marker_weight"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0, 1)
    if "n_states" not in mp:
        mp["n_states"] = np.where(mp.get("status", "").astype(str).str.contains("trimodal"), 3, 2)
    return mp


def load_batch_results(batch_id, result_dir):
    """Load Step 1-3 outputs."""
    result_dir = Path(result_dir)
    cell_clusters = pd.read_csv(result_dir / f"{batch_id}_cell_clusters.csv")
    cluster_library = pd.read_csv(result_dir / f"{batch_id}_cluster_marker_library_valid.csv")
    marker_params = pd.read_csv(result_dir / f"{batch_id}_marker_params.csv")
    prob = np.load(result_dir / f"{batch_id}_protein_probabilities.npz", allow_pickle=True)
    marker_names = prob["marker_names"].astype(str)
    high_prob = prob["high_prob"]
    low_prob = prob["low_prob"]
    unimodal_prob = prob["unimodal_prob"] if "unimodal_prob" in prob.files else np.zeros_like(high_prob)
    empty_prob = prob["empty_prob"] if "empty_prob" in prob.files else np.zeros_like(high_prob)
    cluster_library["cluster"] = cluster_library["cluster"].astype(str)
    if "selected_markers" in cluster_library:
        cluster_library["selected_markers"] = cluster_library["selected_markers"].apply(parse_marker_list)
    else:
        cluster_library["selected_markers"] = [[] for _ in range(len(cluster_library))]
    marker_params = _marker_weight_table(marker_params.set_index("marker").loc[marker_names].reset_index())
    return {
        "batch": batch_id,
        "cell_clusters": cell_clusters,
        "cluster_library": cluster_library,
        "marker_params": marker_params,
        "high_prob": high_prob,
        "low_prob": low_prob,
        "unimodal_prob": unimodal_prob,
        "empty_prob": empty_prob,
        "marker_names": marker_names,
    }


def build_selected_marker_map(cluster_library):
    return {str(row["cluster"]): set(row.get("selected_markers", [])) for _, row in cluster_library.iterrows()}


def build_cluster_signatures(batch_data, cluster_key="protein_phenotype", high_threshold=0.5):
    """Build marker-state signatures for every phenotype component."""
    cell_clusters = batch_data["cell_clusters"].copy()
    if cluster_key not in cell_clusters and cluster_key == "protein_phenotype":
        cluster_key = "protein_leiden"
    high = batch_data["high_prob"]
    low = batch_data["low_prob"]
    empty = batch_data.get("empty_prob", np.zeros_like(high))
    marker_names = batch_data["marker_names"]
    mp = batch_data["marker_params"]
    selected_map = build_selected_marker_map(batch_data["cluster_library"])
    clusters = _sort_ids(cell_clusters[cluster_key].astype(str).unique())
    records = []
    for clust in clusters:
        idx = cell_clusters[cluster_key].astype(str).values == str(clust)
        mean_high = high[idx].mean(axis=0)
        mean_low = low[idx].mean(axis=0)
        mean_empty = empty[idx].mean(axis=0)
        selected_set = selected_map.get(str(clust), set())
        state_stack = np.vstack([mean_low, mean_high]).T
        dominant_state = np.argmax(state_stack, axis=1)
        dominant_prob = state_stack[np.arange(len(marker_names)), dominant_state]
        sig = pd.DataFrame({
            "batch": batch_data["batch"],
            "cluster": str(clust),
            "marker": marker_names,
            "mean_low": mean_low,
            "mean_high": mean_high,
            "mean_empty": mean_empty,
            "frac_low": (low[idx] >= high_threshold).mean(axis=0),
            "frac_high": (high[idx] >= high_threshold).mean(axis=0),
            "dominant_state": dominant_state,
            "dominant_prob": dominant_prob,
            "marker_status": mp["status"].astype(str).values,
            "n_states": mp["n_states"].astype(int).values,
            "marker_weight": mp["marker_weight"].astype(float).values,
            "is_selected": np.array([m in selected_set for m in marker_names]),
            "n_cells": int(idx.sum()),
        })
        records.append(sig)
    sig = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    sig["specificity"] = 0.0
    sig["direction"] = 0.0
    for marker, df_m in sig.groupby("marker"):
        vals = df_m["dominant_prob"].values
        sizes = df_m["n_cells"].values.astype(float)
        effects = np.zeros_like(vals)
        for i in range(len(vals)):
            keep = np.arange(len(vals)) != i
            if keep.any():
                rest = np.average(vals[keep], weights=sizes[keep])
                effects[i] = vals[i] - rest
        sig.loc[df_m.index, "specificity"] = np.clip(np.abs(effects), 0.0, 1.0)
        sig.loc[df_m.index, "direction"] = np.sign(effects)
    return sig


def _state_vector(df):
    return df[["mean_low", "mean_high"]].values


def compute_pair_score_specific(
    sig_A_clust,
    sig_B_clust,
    max_empty=0.5,
    min_common_markers=3,
    min_weight=0.0,
    high_state_threshold=0.5,
    use_selected_union=True,
    max_batch_conflicts=2,
    strong_conflict_threshold=0.80,
):
    """Compare two independently fitted qualitative configurations."""
    del high_state_threshold
    del use_selected_union, min_weight

    def observed(signature):
        return signature.loc[
            (signature["marker_weight"] > 0)
            & (signature["n_states"] == 2)
            & (signature["mean_empty"] < max_empty)
            & (signature["dominant_prob"] >= 0.60)
            & signature["marker_status"].astype(str).isin(
                {"valid_bimodal", "weak_bimodal", "unknown"}
            )
        ].copy()

    observed_A = observed(sig_A_clust)
    observed_B = observed(sig_B_clust)
    df = observed_A.merge(
        observed_B,
        on="marker",
        suffixes=("_A", "_B"),
    )
    if df.shape[0] < min_common_markers:
        return {
            "score": 0.0,
            "n_common_markers": int(df.shape[0]),
            "common_markers": df.get("marker", pd.Series(dtype=str)).tolist(),
            "mean_prob_sim": 0.0,
            "mean_state_agree": 0.0,
            "mean_specificity": 0.0,
            "n_state_conflicts": 0,
            "n_strong_state_conflicts": 0,
            "configuration_coverage": 0.0,
        }
    state_agree = (
        df["dominant_state_A"].values == df["dominant_state_B"].values
    )
    n_conflicts = int((~state_agree).sum())
    conflict_strength = np.minimum(
        df["dominant_prob_A"].to_numpy(float)
        * df["marker_weight_A"].to_numpy(float),
        df["dominant_prob_B"].to_numpy(float)
        * df["marker_weight_B"].to_numpy(float),
    )
    n_strong_conflicts = int(
        np.sum((~state_agree) & (conflict_strength >= strong_conflict_threshold))
    )
    agreement = float(state_agree.mean())
    coverage = float(
        len(df) / max(len(observed_A), len(observed_B), 1)
    )
    tolerable = (
        n_conflicts <= max(int(max_batch_conflicts), 0)
        and n_strong_conflicts == 0
    )
    score = agreement * coverage if tolerable else 0.0
    return {
        "score": float(score),
        "n_common_markers": int(df.shape[0]),
        "common_markers": df["marker"].tolist(),
        "mean_prob_sim": agreement,
        "mean_state_agree": agreement,
        "mean_specificity": coverage,
        "n_state_conflicts": n_conflicts,
        "n_strong_state_conflicts": n_strong_conflicts,
        "configuration_coverage": coverage,
    }


def build_score_matrix_specific(sig_A, sig_B, min_common_markers=3, max_empty=0.5, min_weight=0.0, use_selected_union=True, max_batch_conflicts=2):
    clusters_A = _sort_ids(sig_A["cluster"].astype(str).unique())
    clusters_B = _sort_ids(sig_B["cluster"].astype(str).unique())
    score_mat = np.zeros((len(clusters_A), len(clusters_B)))
    n_common_mat = np.zeros_like(score_mat, dtype=int)
    detail_dict = {}
    for i, cA in enumerate(clusters_A):
        sig_A_c = sig_A[sig_A["cluster"].astype(str) == str(cA)]
        for j, cB in enumerate(clusters_B):
            sig_B_c = sig_B[sig_B["cluster"].astype(str) == str(cB)]
            detail = compute_pair_score_specific(sig_A_c, sig_B_c, min_common_markers=min_common_markers, max_empty=max_empty, min_weight=min_weight, use_selected_union=use_selected_union, max_batch_conflicts=max_batch_conflicts)
            score_mat[i, j] = detail["score"]
            n_common_mat[i, j] = detail["n_common_markers"]
            detail_dict[(str(cA), str(cB))] = detail
    return score_mat, n_common_mat, detail_dict, clusters_A, clusters_B


def hungarian_match_clusters(score_mat, clusters_A, clusters_B, detail_dict, min_score=0.75, min_margin=0.03):
    row_ind, col_ind = linear_sum_assignment(-score_mat)
    records = []
    for r, c in zip(row_ind, col_ind):
        cA, cB = str(clusters_A[r]), str(clusters_B[c])
        score = float(score_mat[r, c])
        row_scores = score_mat[r].copy(); row_scores[c] = -np.inf
        col_scores = score_mat[:, c].copy(); col_scores[r] = -np.inf
        row_margin = score - np.max(row_scores) if np.isfinite(np.max(row_scores)) else score
        col_margin = score - np.max(col_scores) if np.isfinite(np.max(col_scores)) else score
        margin = min(row_margin, col_margin)
        status = "matched" if score >= min_score and margin >= min_margin else ("unmatched_ambiguous" if score >= min_score else "unmatched_low_score")
        detail = detail_dict[(cA, cB)]
        records.append({"cluster_A": cA, "cluster_B": cB, "score": score, "row_margin": float(row_margin), "col_margin": float(col_margin), "margin": float(margin), "status": status, **{k: detail[k] for k in ["n_common_markers", "common_markers", "mean_prob_sim", "mean_state_agree", "mean_specificity"]}})
    return pd.DataFrame(records).sort_values("score", ascending=False)


def rejectable_many_to_one_match(
    score_mat,
    clusters_A,
    clusters_B,
    detail_dict,
    min_score=0.80,
    min_margin=0.03,
    secondary_score=0.83,
):
    candidates = {(i, int(np.argmax(score_mat[i]))) for i in range(score_mat.shape[0])}
    candidates |= {(int(np.argmax(score_mat[:, j])), j) for j in range(score_mat.shape[1])}
    records = []
    for r, c in sorted(candidates):
        cA, cB = str(clusters_A[r]), str(clusters_B[c])
        score = float(score_mat[r, c])
        row_alt = np.delete(score_mat[r], c)
        col_alt = np.delete(score_mat[:, c], r)
        row_margin = score - row_alt.max() if row_alt.size else score
        col_margin = score - col_alt.max() if col_alt.size else score
        row_best = c == int(np.argmax(score_mat[r]))
        col_best = r == int(np.argmax(score_mat[:, c]))
        mutual_best = row_best and col_best
        primary = (
            score >= min_score
            and mutual_best
            and row_margin >= min_margin
            and col_margin >= min_margin
        )
        secondary = (
            score >= secondary_score
            and (row_best or col_best)
            and max(row_margin, col_margin) >= min_margin
        )
        status = (
            "matched"
            if primary
            else "matched_many_to_one"
            if secondary
            else "unmatched_ambiguous"
            if score >= min_score
            else "unmatched_low_score"
        )
        detail = detail_dict[(cA, cB)]
        records.append({"cluster_A": cA, "cluster_B": cB, "score": score, "row_margin": float(row_margin), "col_margin": float(col_margin), "margin": float(min(row_margin, col_margin)), "row_best": bool(row_best), "col_best": bool(col_best), "mutual_best": bool(mutual_best), "status": status, **{k: detail[k] for k in ["n_common_markers", "common_markers", "mean_prob_sim", "mean_state_agree", "mean_specificity"]}})
    return pd.DataFrame(records).sort_values("score", ascending=False)


def build_consensus_library(match_df, sig_A, sig_B, max_empty=0.5, use_matched_only=True):
    records = []
    pairs = match_df[match_df["status"].str.startswith("matched")].copy() if use_matched_only else match_df.copy()
    for _, row in pairs.iterrows():
        cA, cB = str(row["cluster_A"]), str(row["cluster_B"])
        df = sig_A[sig_A["cluster"].astype(str) == cA].merge(sig_B[sig_B["cluster"].astype(str) == cB], on="marker", suffixes=("_A", "_B"))
        usable = (df["marker_weight_A"] > 0) & (df["marker_weight_B"] > 0) & (df["mean_empty_A"] < max_empty) & (df["mean_empty_B"] < max_empty) & (df["is_selected_A"] | df["is_selected_B"])
        df = df.loc[usable].copy()
        if df.empty:
            continue
        n_A, n_B = float(df["n_cells_A"].iloc[0]), float(df["n_cells_B"].iloc[0])
        weight_A = n_A / max(n_A + n_B, 1.0)
        weight_B = 1.0 - weight_A
        for state in ["low", "high", "empty"]:
            df[f"consensus_{state}"] = weight_A * df[f"mean_{state}_A"] + weight_B * df[f"mean_{state}_B"]
        df["consensus_weight"] = 0.5 * (df["marker_weight_A"] + df["marker_weight_B"]) * (0.25 + 0.5 * (df["specificity_A"] + df["specificity_B"]))
        state_mat = df[["consensus_low", "consensus_high"]].values
        state_names = np.array(["low", "high"], dtype=object)
        df["consensus_state"] = state_names[np.argmax(state_mat, axis=1)]
        consensus_id = f"{cA}|{cB}"
        for _, m in df.iterrows():
            records.append({
                "consensus_id": consensus_id,
                "cluster_A": cA,
                "cluster_B": cB,
                "marker": m["marker"],
                "consensus_low": float(m["consensus_low"]),
                "consensus_high": float(m["consensus_high"]),
                "consensus_empty": float(m["consensus_empty"]),
                "consensus_weight": float(m["consensus_weight"]),
                "consensus_specificity": float(0.5 * (m["specificity_A"] + m["specificity_B"])),
                "consensus_state": m["consensus_state"],
                "match_score": float(row["score"]),
                "match_status": row["status"],
            })
    return pd.DataFrame(records)


def annotate_cells_with_consensus(batch_data, consensus_df, min_confidence=0.65, min_margin=0.03):
    high = batch_data["high_prob"]
    low = batch_data["low_prob"]
    empty = batch_data.get("empty_prob", np.zeros_like(high))
    marker_names = list(batch_data["marker_names"])
    cell_clusters = batch_data["cell_clusters"].copy()
    if consensus_df.empty:
        out = cell_clusters.copy()
        out["consensus_annotation"] = "Uncertain"
        out["annotation_confidence"] = 0.0
        out["annotation_margin"] = 0.0
        return out, np.zeros((high.shape[0], 0)), []
    consensus_ids = consensus_df["consensus_id"].unique().tolist()
    score_mat = np.zeros((high.shape[0], len(consensus_ids)))
    for k, cid in enumerate(consensus_ids):
        lib = consensus_df[consensus_df["consensus_id"] == cid].copy()
        usable_markers = [m for m in lib["marker"].tolist() if m in marker_names]
        if not usable_markers:
            continue
        marker_idx = [marker_names.index(m) for m in usable_markers]
        lib = lib.set_index("marker").loc[usable_markers].reset_index()
        w = lib["consensus_weight"].values
        cell_state = np.stack([low[:, marker_idx], high[:, marker_idx]], axis=2)
        target = lib[["consensus_low", "consensus_high"]].values[None, :, :]
        sim = 1.0 - 0.5 * np.abs(cell_state - target).sum(axis=2)
        sim = np.clip(sim, 0.0, 1.0)
        usable = empty[:, marker_idx] < 0.5
        denom = np.sum(w[None, :] * usable, axis=1)
        score_mat[:, k] = np.sum(sim * w[None, :] * usable, axis=1) / np.maximum(denom, EPS)
    best_idx = np.argmax(score_mat, axis=1)
    best_score = score_mat[np.arange(score_mat.shape[0]), best_idx]
    second_score = np.partition(score_mat, -2, axis=1)[:, -2] if score_mat.shape[1] > 1 else np.zeros(score_mat.shape[0])
    margin = best_score - second_score
    annotation = np.array(consensus_ids, dtype=object)[best_idx]
    annotation[(best_score < min_confidence) | (margin < max(min_margin, 0.02))] = "Uncertain"
    out = cell_clusters.copy()
    out["consensus_annotation"] = annotation
    out["annotation_confidence"] = best_score
    out["annotation_margin"] = margin
    return out, score_mat, consensus_ids


def align_two_batches(batch_A, batch_B, result_dir, min_common_markers=3, max_empty=0.5, min_score=0.80, min_margin=0.03, min_annotation_confidence=0.65, max_batch_conflicts=2, save=True):
    result_dir = Path(result_dir)
    A = load_batch_results(batch_A, result_dir)
    B = load_batch_results(batch_B, result_dir)
    sig_A = build_cluster_signatures(A)
    sig_B = build_cluster_signatures(B)
    score_mat, n_common_mat, detail_dict, clusters_A, clusters_B = build_score_matrix_specific(sig_A, sig_B, min_common_markers=min_common_markers, max_empty=max_empty, use_selected_union=True, max_batch_conflicts=max_batch_conflicts)
    match_df = rejectable_many_to_one_match(score_mat, clusters_A, clusters_B, detail_dict, min_score=min_score, min_margin=min_margin)
    consensus_df = build_consensus_library(match_df, sig_A, sig_B, max_empty=max_empty, use_matched_only=True)
    anno_A, score_cells_A, consensus_ids = annotate_cells_with_consensus(A, consensus_df, min_confidence=min_annotation_confidence, min_margin=min_margin)
    anno_B, score_cells_B, _ = annotate_cells_with_consensus(B, consensus_df, min_confidence=min_annotation_confidence, min_margin=min_margin)
    if save:
        match_df.to_csv(result_dir / f"{batch_A}_{batch_B}_cluster_matching_specific.csv", index=False)
        consensus_df.to_csv(result_dir / f"{batch_A}_{batch_B}_consensus_marker_library_specific.csv", index=False)
        anno_A.to_csv(result_dir / f"{batch_A}_consensus_cell_annotation_specific.csv", index=False)
        anno_B.to_csv(result_dir / f"{batch_B}_consensus_cell_annotation_specific.csv", index=False)
    return {"A": A, "B": B, "sig_A": sig_A, "sig_B": sig_B, "score_mat": score_mat, "n_common_mat": n_common_mat, "clusters_A": clusters_A, "clusters_B": clusters_B, "match_df": match_df, "consensus_df": consensus_df, "anno_A": anno_A, "anno_B": anno_B, "score_cells_A": score_cells_A, "score_cells_B": score_cells_B, "consensus_ids": consensus_ids}


def add_groundtruth_to_match_df(match_df, A, B, subtype_key="subtype"):
    def cluster_major(cell_df):
        key = "protein_phenotype" if "protein_phenotype" in cell_df else "protein_leiden"
        records = []
        for clust, df in cell_df.groupby(key):
            if subtype_key in df.columns and len(df):
                vc = df[subtype_key].astype(str).value_counts()
                major, frac = vc.index[0], vc.iloc[0] / len(df)
            else:
                major, frac = "Unknown", np.nan
            records.append({"cluster": str(clust), "major_subtype": major, "major_fraction": frac, "n_cells": len(df)})
        return pd.DataFrame(records)
    major_A = cluster_major(A["cell_clusters"]).rename(columns={"cluster": "cluster_A", "major_subtype": "major_subtype_A", "major_fraction": "major_fraction_A", "n_cells": "n_cells_A"})
    major_B = cluster_major(B["cell_clusters"]).rename(columns={"cluster": "cluster_B", "major_subtype": "major_subtype_B", "major_fraction": "major_fraction_B", "n_cells": "n_cells_B"})
    out = match_df.copy()
    out["cluster_A"] = out["cluster_A"].astype(str)
    out["cluster_B"] = out["cluster_B"].astype(str)
    out = out.merge(major_A, on="cluster_A", how="left").merge(major_B, on="cluster_B", how="left")
    out["major_subtype_match"] = out["major_subtype_A"] == out["major_subtype_B"]
    matched = out[out["status"].str.startswith("matched")]
    metrics = {
        "matched_pairs": int(matched.shape[0]),
        "cluster_match_accuracy": float(matched["major_subtype_match"].mean()) if len(matched) else np.nan,
        "cell_weighted_match_accuracy": float((matched["major_subtype_match"].astype(float) * matched["n_cells_A"]).sum() / matched["n_cells_A"].sum()) if len(matched) and matched["n_cells_A"].sum() > 0 else np.nan,
    }
    return out, metrics


def evaluate_cell_annotation(annotation_df, subtype_key="subtype", annotation_key="consensus_annotation"):
    df = annotation_df.copy()
    if subtype_key not in df.columns:
        return None, None
    keep = df[annotation_key].astype(str) != "Uncertain"
    if keep.sum() == 0:
        return {"coverage": 0.0, "ARI": np.nan, "NMI": np.nan, "purity": np.nan}, pd.DataFrame()
    y_true = df.loc[keep, subtype_key].astype(str).values
    y_pred = df.loc[keep, annotation_key].astype(str).values
    tab = pd.crosstab(pd.Series(y_pred, name=annotation_key), pd.Series(y_true, name=subtype_key))
    purity = tab.max(axis=1).sum() / max(tab.values.sum(), 1)
    metrics = {"coverage": float(keep.mean()), "ARI": adjusted_rand_score(y_true, y_pred), "NMI": normalized_mutual_info_score(y_true, y_pred), "purity": float(purity)}
    return metrics, tab


def plot_score_heatmap(score_mat, clusters_A, clusters_B, match_df=None, batch_A="A", batch_B="B", figsize=(8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(score_mat, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(clusters_B))); ax.set_xticklabels(clusters_B, rotation=90)
    ax.set_yticks(np.arange(len(clusters_A))); ax.set_yticklabels(clusters_A)
    ax.set_xlabel(batch_B); ax.set_ylabel(batch_A); ax.set_title("Phenotype alignment score")
    fig.colorbar(im, ax=ax, label="score")
    if match_df is not None and not match_df.empty:
        idx_A = {str(c): i for i, c in enumerate(clusters_A)}; idx_B = {str(c): i for i, c in enumerate(clusters_B)}
        for _, r in match_df[match_df["status"] == "matched"].iterrows():
            ax.scatter(idx_B[str(r["cluster_B"])], idx_A[str(r["cluster_A"])], marker="s", facecolors="none", edgecolors="black", s=90)
    plt.tight_layout()
    return ax


def plot_match_score_distribution(match_df, min_score=0.75):
    ax = match_df["score"].hist(bins=20)
    ax.axvline(min_score, linestyle="--")
    ax.set_xlabel("match score"); ax.set_ylabel("pairs")
    return ax


def plot_matched_pair_marker_pattern(*args, **kwargs):
    raise NotImplementedError("Use sig_A/sig_B tables to plot marker patterns for heterogeneous states.")
