"""Generic marker-state atlas construction.

This module is intentionally platform agnostic. Inputs are local cluster
fingerprints with low/high marker states; outputs keep every local cluster as a
fine atlas entry and add cross-source compatibility links plus candidate sets.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-12


def join_or_none(values: Iterable[object]) -> str:
    values = [str(value) for value in values if str(value)]
    return "; ".join(values) if values else "none"


def ensure_signature_columns(signatures: pd.DataFrame, source_col: str = "source") -> pd.DataFrame:
    """Normalize local-cluster signature columns used by atlas builders."""
    out = signatures.copy()
    if source_col not in out.columns:
        for candidate in ("platform", "panel", "batch"):
            if candidate in out.columns:
                out[source_col] = out[candidate].astype(str)
                break
    if source_col not in out.columns:
        raise KeyError(f"Missing source column: {source_col}")
    if "source" not in out.columns:
        out["source"] = out[source_col].astype(str)
    for col in ["low_prob", "high_prob", "marker_weight", "specificity", "state_confidence"]:
        if col not in out.columns:
            out[col] = 1.0 if col in {"marker_weight", "state_confidence"} else 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "dominant_state" not in out.columns:
        probs = out[["low_prob", "high_prob"]].to_numpy(float)
        out["dominant_state"] = np.array(["low", "high"], dtype=object)[probs.argmax(axis=1)]
        out["state_confidence"] = probs.max(axis=1)
    if "n_cells" not in out.columns:
        out["n_cells"] = 0
    if "cluster" not in out.columns:
        out["cluster"] = out["node_id"].astype(str)
    out["dominant_state"] = out["dominant_state"].astype(str).replace({"nan": "unknown"})
    out["source"] = out["source"].astype(str)
    out["node_id"] = out["node_id"].astype(str)
    out["common_marker"] = out["common_marker"].astype(str)
    return out


def score_marker_state_pair(
    sig_a: pd.DataFrame,
    sig_b: pd.DataFrame,
    min_shared_markers: int = 3,
    min_state_confidence: float = 0.60,
    min_state_separation: float = 0.15,
) -> dict:
    """Classify two local fingerprints by shared low/high marker states."""
    sig_a = ensure_signature_columns(sig_a)
    sig_b = ensure_signature_columns(sig_b)
    for sig in (sig_a, sig_b):
        sig["_state_separation"] = (sig["high_prob"].astype(float) - sig["low_prob"].astype(float)).abs()
    merged = sig_a.merge(sig_b, on="common_marker", suffixes=("_a", "_b"))
    merged = merged[
        merged["dominant_state_a"].isin(["low", "high"])
        & merged["dominant_state_b"].isin(["low", "high"])
        & (merged["state_confidence_a"].astype(float) >= min_state_confidence)
        & (merged["state_confidence_b"].astype(float) >= min_state_confidence)
        & (merged["_state_separation_a"].astype(float) >= min_state_separation)
        & (merged["_state_separation_b"].astype(float) >= min_state_separation)
    ].copy()
    if merged.shape[0] < min_shared_markers:
        return {
            "score": 0.0,
            "relation": "weak_overlap",
            "integration_action": "keep_separate_missing_evidence",
            "n_shared_markers": int(merged.shape[0]),
            "n_state_matches": 0,
            "n_state_conflicts": 0,
            "mean_state_agreement": np.nan,
            "conflict_fraction": np.nan,
            "shared_markers": join_or_none(merged.get("common_marker", [])),
            "matched_markers": "none",
            "conflict_markers": "none",
        }

    state_match = merged["dominant_state_a"].to_numpy() == merged["dominant_state_b"].to_numpy()
    n_matches = int(state_match.sum())
    n_conflicts = int((~state_match).sum())
    state_jaccard = float(n_matches / max(len(merged), 1))
    x = merged[["low_prob_a", "high_prob_a"]].to_numpy(float)
    y = merged[["low_prob_b", "high_prob_b"]].to_numpy(float)
    prob_sim = np.clip(1.0 - 0.5 * np.abs(x - y).sum(axis=1), 0.0, 1.0)
    weights = 0.5 * (merged["marker_weight_a"].to_numpy(float) + merged["marker_weight_b"].to_numpy(float))
    specificity = 0.5 * (merged["specificity_a"].to_numpy(float) + merged["specificity_b"].to_numpy(float))
    weights = np.nan_to_num(weights * (0.25 + specificity), nan=0.0)
    if weights.sum() <= 0:
        weights = np.ones(len(merged), dtype=float)

    markers_a = set(sig_a["common_marker"].astype(str))
    markers_b = set(sig_b["common_marker"].astype(str))
    shared = set(merged["common_marker"].astype(str))
    complementarity = float((len(markers_a - shared) + len(markers_b - shared)) / max(len(markers_a | markers_b), 1))
    relation = "match" if n_conflicts == 0 else "conflict"
    return {
        "score": float(np.average(prob_sim, weights=weights)),
        "relation": relation,
        "integration_action": "candidate_fine_subtype_set" if relation == "match" else "exclude_from_same_candidate_set",
        "n_shared_markers": int(len(merged)),
        "mean_prob_similarity": float(np.average(prob_sim, weights=weights)),
        "n_state_matches": n_matches,
        "n_state_conflicts": n_conflicts,
        "mean_state_agreement": state_jaccard,
        "state_jaccard": state_jaccard,
        "state_containment": state_jaccard,
        "conflict_fraction": float(n_conflicts / max(len(merged), 1)),
        "complementarity": complementarity,
        "markers_unique_to_a": join_or_none(sorted(markers_a - shared)),
        "markers_unique_to_b": join_or_none(sorted(markers_b - shared)),
        "shared_markers": join_or_none(merged["common_marker"].astype(str)),
        "matched_markers": join_or_none(merged.loc[state_match, "common_marker"].astype(str)),
        "conflict_markers": join_or_none(merged.loc[~state_match, "common_marker"].astype(str)),
    }


def build_atlas_links(
    signatures: pd.DataFrame,
    source_col: str = "source",
    min_shared_markers: int = 3,
    compare_within_source: bool = False,
    min_state_confidence: float = 0.60,
    min_state_separation: float = 0.15,
) -> pd.DataFrame:
    """Score all compatible local-cluster pairs."""
    sig = ensure_signature_columns(signatures, source_col=source_col)
    groups = {node: sub for node, sub in sig.groupby("node_id", sort=False)}
    sources = sig.drop_duplicates("node_id").set_index("node_id")["source"].to_dict()
    rows = []
    nodes = sorted(groups)
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1:]:
            if not compare_within_source and sources[node_a] == sources[node_b]:
                continue
            score = score_marker_state_pair(
                groups[node_a],
                groups[node_b],
                min_shared_markers,
                min_state_confidence=min_state_confidence,
                min_state_separation=min_state_separation,
            )
            rows.append(
                {
                    "node_a": node_a,
                    "node_b": node_b,
                    "source_a": sources[node_a],
                    "source_b": sources[node_b],
                    **score,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["relation", "n_state_conflicts", "n_state_matches", "score"],
        ascending=[False, True, False, False],
    ).reset_index(drop=True)


def build_fine_atlas(signatures: pd.DataFrame, links: pd.DataFrame, source_col: str = "source") -> dict[str, pd.DataFrame]:
    """Keep local clusters as fine atlas entries and attach compatibility links."""
    sig = ensure_signature_columns(signatures, source_col=source_col)
    fine_nodes = sig[["node_id", "source", "cluster", "n_cells"]].drop_duplicates("node_id").copy()
    fine_nodes = fine_nodes.sort_values(["source", "cluster"]).reset_index(drop=True)
    fine_nodes["fine_atlas_id"] = [f"ATLAS_FINE_{i + 1:04d}" for i in range(len(fine_nodes))]
    fine_nodes["is_local_fine_entry"] = True
    id_map = fine_nodes.set_index("node_id")["fine_atlas_id"].to_dict()

    out_links = links.copy()
    if out_links.empty:
        out_links = pd.DataFrame(columns=["node_a", "node_b", "fine_atlas_a", "fine_atlas_b", "relation"])
    else:
        out_links["fine_atlas_a"] = out_links["node_a"].map(id_map)
        out_links["fine_atlas_b"] = out_links["node_b"].map(id_map)
        out_links["is_match"] = out_links["relation"].eq("match")
        out_links["is_conflict"] = out_links["relation"].eq("conflict")
        out_links["is_usable_support_link"] = out_links["is_match"]
        out_links["rank_from_a"] = out_links.groupby("node_a")["score"].rank(method="first", ascending=False).astype(int)
        out_links["rank_from_b"] = out_links.groupby("node_b")["score"].rank(method="first", ascending=False).astype(int)
        out_links["is_mutual_top_link"] = out_links["rank_from_a"].eq(1) & out_links["rank_from_b"].eq(1)

    marker = sig.merge(fine_nodes[["node_id", "fine_atlas_id"]], on="node_id", how="left")
    marker["marker_evidence_score"] = (
        marker["state_confidence"].astype(float)
        * marker["specificity"].astype(float)
        * marker["marker_weight"].astype(float)
    )
    marker = marker.sort_values(["fine_atlas_id", "dominant_state", "marker_evidence_score"], ascending=[True, True, False])
    marker["rank_within_entry_state"] = marker.groupby(["fine_atlas_id", "dominant_state"])["marker_evidence_score"].rank(method="first", ascending=False).astype(int)

    def top_markers(df: pd.DataFrame, state: str, limit: int = 12) -> str:
        sub = df[df["dominant_state"].eq(state)].sort_values("marker_evidence_score", ascending=False).head(limit)
        return join_or_none(sub["common_marker"].astype(str))

    rows = []
    for row in fine_nodes.itertuples(index=False):
        sub = marker[marker["fine_atlas_id"].eq(row.fine_atlas_id)]
        node_links = out_links[(out_links.get("node_a", pd.Series(dtype=str)).eq(row.node_id)) | (out_links.get("node_b", pd.Series(dtype=str)).eq(row.node_id))].copy()
        usable = node_links[node_links.get("is_usable_support_link", False)].copy() if not node_links.empty else node_links
        if not usable.empty:
            usable["linked_node"] = np.where(usable["node_a"].eq(row.node_id), usable["node_b"], usable["node_a"])
            usable["linked_fine_atlas"] = np.where(usable["node_a"].eq(row.node_id), usable["fine_atlas_b"], usable["fine_atlas_a"])
            usable = usable.sort_values("score", ascending=False)
        rows.append(
            {
                "fine_atlas_id": row.fine_atlas_id,
                "node_id": row.node_id,
                "source": row.source,
                "cluster": row.cluster,
                "n_cells": row.n_cells,
                "n_match_links": int(node_links["relation"].eq("match").sum()) if not node_links.empty else 0,
                "n_conflict_links": int(node_links["relation"].eq("conflict").sum()) if not node_links.empty else 0,
                "best_linked_fine_atlas": join_or_none(usable["linked_fine_atlas"].head(8).astype(str)) if not usable.empty else "none",
                "best_linked_nodes": join_or_none(usable["linked_node"].head(8).astype(str)) if not usable.empty else "none",
                "high_markers": top_markers(sub, "high"),
                "low_markers": top_markers(sub, "low"),
                "measured_markers": join_or_none(sorted(sub["common_marker"].unique())),
            }
        )
    return {
        "fine_atlas_nodes": fine_nodes,
        "fine_atlas_links": out_links,
        "fine_atlas_marker_summary": marker,
        "fine_atlas_library": pd.DataFrame(rows),
    }


def build_atlas_candidate_sets(
    signatures: pd.DataFrame,
    fine_nodes: pd.DataFrame,
    links: pd.DataFrame,
    source_col: str = "source",
    max_direct_candidates: int = 12,
) -> pd.DataFrame:
    """Optimize compatible candidate sets without forcing coarse nodes to merge."""
    sig = ensure_signature_columns(signatures, source_col=source_col)
    if links.empty:
        return pd.DataFrame()
    node_to_fine = fine_nodes.set_index("node_id")["fine_atlas_id"].to_dict()
    node_to_source = fine_nodes.set_index("node_id")["source"].to_dict()
    marker_sets = {node: set(sub["common_marker"].astype(str)) for node, sub in sig.groupby("node_id")}
    pair_relation = {frozenset([r.node_a, r.node_b]): r.relation for r in links.itertuples(index=False)}
    pair_state_matches = {frozenset([r.node_a, r.node_b]): int(getattr(r, "n_state_matches", 0)) for r in links.itertuples(index=False)}

    def direct_matches(seed: str) -> list[str]:
        sub = links[((links["node_a"].eq(seed)) | (links["node_b"].eq(seed))) & links["relation"].eq("match")].copy()
        if sub.empty:
            return []
        sub["candidate_node"] = np.where(sub["node_a"].eq(seed), sub["node_b"], sub["node_a"])
        sub = sub.sort_values(["n_state_matches", "n_shared_markers", "score"], ascending=[False, False, False])
        return sub["candidate_node"].head(max_direct_candidates).astype(str).tolist()

    def feasible(nodes: tuple[str, ...]) -> bool:
        for a, b in itertools.combinations(nodes, 2):
            if node_to_source.get(a) == node_to_source.get(b):
                continue
            if pair_relation.get(frozenset([a, b])) != "match":
                return False
        return True

    rows = []
    for seed in fine_nodes["node_id"].astype(str):
        candidates = direct_matches(seed)
        best_nodes = (seed,)
        best_objective = (-1, -1, -1, -1)
        for size in range(len(candidates) + 1):
            for combo in itertools.combinations(candidates, size):
                nodes = (seed, *combo)
                if not feasible(nodes):
                    continue
                covered = set().union(*(marker_sets.get(node, set()) for node in nodes))
                sources = {node_to_source.get(node, "") for node in nodes}
                state_support = sum(
                    pair_state_matches.get(frozenset([a, b]), 0)
                    for a, b in itertools.combinations(nodes, 2)
                    if node_to_source.get(a) != node_to_source.get(b)
                )
                objective = (len(covered), len(sources), state_support, len(nodes))
                if objective > best_objective:
                    best_objective = objective
                    best_nodes = nodes
        possible_nodes = [node for node in best_nodes if node != seed]
        rows.append(
            {
                "seed_fine_atlas_id": node_to_fine.get(seed, ""),
                "seed_node_id": seed,
                "seed_source": node_to_source.get(seed, ""),
                "possible_fine_atlas_ids": join_or_none(node_to_fine.get(node, "") for node in possible_nodes),
                "possible_node_ids": join_or_none(possible_nodes),
                "n_possible_fine_subtypes": int(len(possible_nodes)),
                "optimized_node_set": join_or_none(node_to_fine.get(node, "") for node in best_nodes),
                "optimized_sources": join_or_none(sorted({node_to_source.get(node, "") for node in best_nodes})),
                "marker_coverage": int(best_objective[0]),
                "state_match_support": int(best_objective[2]),
                "candidate_status": "no_cross_source_candidate" if not possible_nodes else "has_cross_source_candidate",
                "interpretation": "specific_fine_entry" if len(possible_nodes) <= 1 else "coarse_entry_multiple_fine_subtypes_possible",
            }
        )
    return pd.DataFrame(rows)


def build_marker_state_atlas(
    signatures: pd.DataFrame,
    source_col: str = "source",
    min_shared_markers: int = 3,
    compare_within_source: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build a generic atlas from local cluster marker-state signatures."""
    sig = ensure_signature_columns(signatures, source_col=source_col)
    links = build_atlas_links(sig, source_col="source", min_shared_markers=min_shared_markers, compare_within_source=compare_within_source)
    fine = build_fine_atlas(sig, links, source_col="source")
    candidates = build_atlas_candidate_sets(sig, fine["fine_atlas_nodes"], fine["fine_atlas_links"], source_col="source")
    out = {
        "cluster_signatures": sig,
        "atlas_links": links,
        "atlas_candidate_sets": candidates,
    }
    out.update(fine)
    return out


def save_atlas_outputs(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in outputs.items():
        df.to_csv(out / f"{name}.csv", index=False)
