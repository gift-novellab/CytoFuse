"""Panel-aware FACS atlas assembly from local phenotype fingerprints."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from cytofuse.methods.atlas import (
    build_atlas_candidate_sets,
    build_fine_atlas as build_generic_fine_atlas,
    build_atlas_links,
    join_or_none,
    save_atlas_outputs,
    score_marker_state_pair,
)

EPS = 1e-12

MARKER_ALIASES = {
    "HLADR": "HLA_DR",
    "HLA_DR": "HLA_DR",
    "IL7RA": "CD127",
    "CD127": "CD127",
    "PD1": "PD1",
    "PDL1": "PDL1",
    "VA72": "VA7_2",
    "VA7_2": "VA7_2",
    "VD2": "VD2",
    "GD": "GD",
}


def normalize_marker_name(marker: str) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "", str(marker)).upper()
    return MARKER_ALIASES.get(name, name)


def parse_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
    except Exception:
        pass
    text = str(value).strip().strip("[]")
    return [x.strip().strip("'\"") for x in text.split(",") if x.strip()]


def load_panel(panel: str, result_dir: str | Path) -> dict:
    result_dir = Path(result_dir)
    cells = pd.read_csv(result_dir / f"{panel}_cell_clusters.csv", low_memory=False)
    library = pd.read_csv(result_dir / f"{panel}_cluster_marker_library_valid.csv", low_memory=False)
    markers = pd.read_csv(result_dir / f"{panel}_marker_params.csv", low_memory=False)
    prob = np.load(result_dir / f"{panel}_protein_probabilities.npz", allow_pickle=True)
    marker_names = prob["marker_names"].astype(str)
    cells["panel"] = panel
    cells["node_id"] = "facs|" + panel + "|" + cells["protein_phenotype"].astype(str)
    library["panel"] = panel
    library["node_id"] = "facs|" + panel + "|" + library["cluster"].astype(str)
    for col in ["positive_markers", "negative_markers", "selected_markers"]:
        if col in library:
            library[col] = library[col].apply(parse_list)
    markers["panel"] = panel
    return {
        "panel": panel,
        "cells": cells,
        "library": library,
        "markers": markers,
        "marker_names": marker_names,
        "low_prob": np.asarray(prob["low_prob"], dtype=float),
        "high_prob": np.asarray(prob["high_prob"], dtype=float),
    }


def marker_panel_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    all_common = sorted({
        normalize_marker_name(marker)
        for rec in records
        for marker in rec["marker_names"]
    })
    for common in all_common:
        row = {"common_marker": common}
        present = []
        for rec in records:
            lookup = {normalize_marker_name(marker): str(marker) for marker in rec["marker_names"]}
            marker = lookup.get(common, "")
            row[f"{rec['panel']}_marker"] = marker
            if marker:
                present.append(rec["panel"])
        row["panels"] = "; ".join(present)
        row["n_panels"] = len(present)
        rows.append(row)
    return pd.DataFrame(rows)


def _marker_weight_table(rec: dict) -> dict[str, float]:
    markers = rec["markers"].copy()
    weight_col = "phenotype_weight" if "phenotype_weight" in markers else "marker_weight"
    if weight_col not in markers:
        markers[weight_col] = 1.0
    out = {}
    for row in markers.itertuples(index=False):
        out[normalize_marker_name(getattr(row, "marker"))] = float(getattr(row, weight_col, 0.0))
    return out


def _marker_param_lookup(rec: dict) -> dict[str, dict]:
    out = {}
    for row in rec["markers"].itertuples(index=False):
        marker = normalize_marker_name(getattr(row, "marker"))
        out[marker] = row._asdict()
    return out


def build_cluster_signatures(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        marker_lookup = {normalize_marker_name(marker): i for i, marker in enumerate(rec["marker_names"])}
        weight_map = _marker_weight_table(rec)
        param_map = _marker_param_lookup(rec)
        cells = rec["cells"]
        for cluster_id, idx in cells.groupby("protein_phenotype", sort=False).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            node_id = f"facs|{rec['panel']}|{cluster_id}"
            n_cells = int(idx.size)
            for common_marker, pos in marker_lookup.items():
                low = float(rec["low_prob"][idx, pos].mean())
                high = float(rec["high_prob"][idx, pos].mean())
                params = param_map.get(common_marker, {})
                n_states = int(params.get("n_states", 2))
                if n_states == 1:
                    state = str(params.get("unimodal_state", "unknown"))
                    if state not in {"low", "high"}:
                        state = "unknown"
                    state_conf = 1.0 if state in {"low", "high"} else 0.0
                else:
                    probs = np.array([low, high])
                    state = ("low", "high")[int(np.argmax(probs))]
                    state_conf = float(probs.max())
                rows.append({
                    "node_id": node_id,
                    "panel": rec["panel"],
                    "cluster": str(cluster_id),
                    "n_cells": n_cells,
                    "common_marker": common_marker,
                    "low_prob": low,
                    "high_prob": high,
                    "dominant_state": state,
                    "state_confidence": state_conf,
                    "activation_score": high - low,
                    "marker_weight": float(weight_map.get(common_marker, 0.0)),
                    "is_measured": True,
                })
    sig = pd.DataFrame(rows)
    if sig.empty:
        return sig
    sig["specificity"] = 0.0
    for marker, idx in sig.groupby("common_marker").groups.items():
        sub = sig.loc[idx]
        weights = np.maximum(sub["n_cells"].to_numpy(float), 1.0)
        values = sub["activation_score"].to_numpy(float)
        total = weights.sum()
        weighted_total = float((values * weights).sum())
        other = (weighted_total - values * weights) / np.maximum(total - weights, 1.0)
        sig.loc[idx, "specificity"] = np.abs(values - other)
    sig["specificity"] = sig["specificity"].clip(0.0, 1.0)
    return sig


def score_pair(sig_a: pd.DataFrame, sig_b: pd.DataFrame, min_shared_markers=3) -> dict:
    """Classify two local fingerprints by marker peak states.

    Atlas assembly is based on each cluster's marker-state fingerprint:
    low/high. Continuous probabilities are retained as confidence
    diagnostics, but the relation itself is determined by whether measured
    shared markers have the same dominant peak state.
    """
    return score_marker_state_pair(sig_a, sig_b, min_shared_markers=min_shared_markers)


def build_fine_atlas(
    signatures: pd.DataFrame,
    matches: pd.DataFrame,
    coarse_nodes: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    del coarse_nodes
    sig = signatures.copy()
    sig["source"] = sig["panel"].astype(str)
    out = build_generic_fine_atlas(sig, matches, source_col="source")
    out["fine_atlas_nodes"] = out["fine_atlas_nodes"].rename(columns={"source": "panel"})
    out["fine_atlas_library"] = out["fine_atlas_library"].rename(columns={"source": "panel"})
    return out


def build_facs_atlas(
    result_dir: str | Path,
    panels: Iterable[str] = ("CCR", "Myeloid", "T"),
    min_edge_score: float = 0.78,
    min_shared_markers: int = 3,
) -> dict[str, pd.DataFrame]:
    records = [load_panel(panel, result_dir) for panel in panels]
    panel_markers = marker_panel_table(records)
    signatures = build_cluster_signatures(records)
    sig_for_atlas = signatures.copy()
    sig_for_atlas["source"] = sig_for_atlas["panel"].astype(str)
    matches = build_atlas_links(sig_for_atlas, source_col="source", min_shared_markers=min_shared_markers)
    matches = matches.rename(columns={"source_a": "panel_a", "source_b": "panel_b"})
    cells = []
    for rec in records:
        c = rec["cells"].copy()
        cells.append(c)
    cells = pd.concat(cells, ignore_index=True)
    fine_outputs = build_fine_atlas(signatures, matches, coarse_nodes=None)
    fine_mapping = fine_outputs["fine_atlas_nodes"].set_index("node_id")["fine_atlas_id"].to_dict()
    cells["fine_atlas_id"] = cells["node_id"].map(fine_mapping)
    fine_cells = cells.drop(columns=["atlas_id"], errors="ignore")
    candidate_sig = signatures.copy()
    candidate_sig["source"] = candidate_sig["panel"].astype(str)
    candidate_nodes = fine_outputs["fine_atlas_nodes"].rename(columns={"panel": "source"})
    candidate_sets = build_atlas_candidate_sets(candidate_sig, candidate_nodes, fine_outputs["fine_atlas_links"], source_col="source")
    candidate_sets = candidate_sets.rename(columns={"seed_source": "seed_panel", "optimized_sources": "optimized_panels"})
    outputs = {
        "panel_marker_mapping": panel_markers,
        "cluster_signatures": signatures,
        "matches": matches,
        "atlas_links": matches,
        "atlas_candidate_sets": candidate_sets,
        "fine_atlas_cells": fine_cells,
    }
    outputs.update(fine_outputs)
    return outputs


def save_facs_atlas(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    save_atlas_outputs(outputs, output_dir)
