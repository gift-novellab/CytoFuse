"""Cross-platform integration of protein EM phenotype results.

The module works on saved CITE-seq and CyTOF batch outputs. Matching is based on
shared-marker posterior signatures only; truth labels are carried along for
diagnostics and are not used by the integration logic.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from cytofuse.methods.atlas import build_marker_state_atlas

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None

try:
    import scanpy as sc
except ImportError:  # pragma: no cover
    sc = None

try:
    import harmonypy
except ImportError:  # pragma: no cover
    harmonypy = None

try:
    from umap import UMAP
except ImportError:  # pragma: no cover
    UMAP = None


STATE_COLUMNS = ("low_prob", "high_prob")
STATE_LABELS = np.array(("low", "high"), dtype=object)
EPS = 1e-12
MARKER_ALIASES = {
    "CCR7": "CD197",
    "PD1": "CD279",
    "CTLA4": "CD152",
    "VA72": "TCRVA72",
    "VD2": "TCRVD2",
    "CD57": "CD57RECOMBINANT",
}

LINEAGE_MARKERS = {
    "B": {"CD19", "CD20", "IGM", "IGA", "HLADR"},
    "CD4_T": {"CD3", "CD4"},
    "CD8_T": {"CD3", "CD8"},
    "NK": {"CD56", "CD16", "KLRG1", "CD57"},
    "Mono": {"CD14", "CD33", "CD11C", "HLADR"},
    "pDC": {"CD123", "HLADR"},
    "Basophil": {"CD123", "CD38"},
    "Platelet": {"CD41", "CD36", "CD31"},
}
SUBTYPE_MARKERS = {
    "naive": {"CCR7", "CD45RA", "CD27", "CD28"},
    "memory": {"CD45RO", "CD27", "CD28"},
    "cytotoxic": {"GZB", "CD57", "KLRG1", "CD16"},
    "activated": {"HLADR", "CD38", "CD39", "CD69", "CD25", "CTLA4", "PD1"},
    "treg": {"FOXP3", "CD25", "CTLA4"},
    "gdT": {"VD2", "VA72"},
}


def coarse_cell_type_label(value) -> str:
    """Map platform-specific truth names to broad diagnostic cell types."""
    if pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown"}:
        return "Unknown"
    key = re.sub(r"[^A-Za-z0-9]+", "", text).upper()
    if "BASOPHIL" in key:
        return "Basophil"
    if "PLASMA" in key or key in {"PB"}:
        return "Plasmablast"
    if "CMONO" in key or "CLASSICALMONO" in key:
        return "cMono"
    if "NCMONO" in key or "NONCLASSICALMONO" in key:
        return "ncMono"
    if "MONO" in key:
        return "Mono"
    if key == "PDC" or "PLASMACYTOIDDENDRITIC" in key:
        return "pDC"
    if key.startswith("B") or key in {"BCELLS", "NAIVEBCELLS"} or "MEMORYB" in key or "SWITCHEDMEMORYB" in key:
        return "B"
    if "NK" in key or "CD56" in key:
        return "NK"
    if "CD8" in key or "CYTOTOXICT" in key or "MEMORYCD8T" in key:
        return "CD8_T"
    if "CD4" in key or "TREG" in key or "MEMORYCD4T" in key or "NAIVECD4T" in key:
        return "CD4_T"
    if "TCRGD" in key or "GAMMADELTA" in key:
        return "gdT"
    if key in {"DN", "TOTHER"} or "TCELLS" in key:
        return "T"
    return text


def add_coarse_truth_labels(cells: pd.DataFrame) -> pd.DataFrame:
    """Attach broad truth labels for cross-platform diagnostics only."""
    out = cells.copy()
    if "platform" not in out:
        return out
    major_values = []
    minor_values = []
    for row in out.itertuples(index=False):
        platform = str(getattr(row, "platform", ""))
        if platform == "citeseq":
            major = getattr(row, "subtype", np.nan)
            minor = getattr(row, "minor_subset", major)
        else:
            major = getattr(row, "major_cell_type", np.nan)
            minor = getattr(row, "fine_cluster_id", major)
        major_values.append(coarse_cell_type_label(major))
        minor_values.append(coarse_cell_type_label(minor))
    out["coarse_truth_label"] = major_values
    out["coarse_fine_truth_label"] = minor_values
    return out


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    result_dir: Path
    batches: tuple[str, ...]
    major_key: str
    minor_key: str


def normalize_marker_name(marker: str) -> str:
    """Return a platform-invariant marker name used only for matching."""
    name = re.sub(r"[^A-Za-z0-9]+", "", str(marker)).upper()
    aliases = {
        "KLGR1": "KLRG1",
        "KLRG1": "KLRG1",
        "HLADR": "HLADR",
        "BCL2": "BCL2",
        "VD2": "VD2",
        "VA72": "VA72",
    }
    return aliases.get(name, name)


def parse_list(value) -> list[str]:
    """Parse list-like CSV fields saved by the cluster summaries."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    text = str(value).strip().strip("[]")
    if not text:
        return []
    return [x.strip().strip("'\"") for x in text.split(",") if x.strip()]


def _marker_set(markers) -> set[str]:
    return {normalize_marker_name(marker) for marker in parse_list(markers)}


def infer_marker_configuration(high_markers, low_markers=None) -> dict:
    """Infer a readable cell-type configuration and resolution from marker evidence."""
    high = _marker_set(high_markers)
    low = _marker_set(low_markers) if low_markers is not None else set()
    evidence = high
    lineage_scores = {
        lineage: len(evidence & markers)
        for lineage, markers in LINEAGE_MARKERS.items()
    }
    lineage, score = max(lineage_scores.items(), key=lambda item: item[1]) if lineage_scores else ("Unknown", 0)
    if score == 0:
        lineage = "Unknown"

    subtype_hits = {
        name: sorted(evidence & markers)
        for name, markers in SUBTYPE_MARKERS.items()
        if evidence & markers
    }
    high_lineage = sorted(high & LINEAGE_MARKERS.get(lineage, set()))
    high_subtype = sorted(set().union(*[set(v) for v in subtype_hits.values()]) & high) if subtype_hits else []
    n_lineage = len(high_lineage)
    n_subtype = len(high_subtype)
    n_high = len(high)

    if lineage == "Unknown" or n_high <= 1 or n_lineage <= 1:
        resolution = "insufficient"
        configuration = "Unknown"
        rationale = "insufficient marker support"
    elif n_subtype == 0 or n_high <= 3:
        resolution = "broad"
        configuration = lineage
        rationale = "lineage markers only"
    elif n_subtype <= 2:
        resolution = "intermediate"
        subtype = "+".join(sorted(subtype_hits)[:2])
        configuration = f"{lineage}:{subtype}"
        rationale = "lineage plus limited subtype markers"
    else:
        resolution = "fine"
        subtype = "+".join(sorted(subtype_hits)[:3])
        configuration = f"{lineage}:{subtype}"
        rationale = "lineage plus multiple subtype/state markers"

    return {
        "configuration_label": configuration,
        "resolution_level": resolution,
        "resolution_rationale": rationale,
        "inferred_lineage": lineage,
        "n_high_markers": n_high,
        "n_lineage_markers": n_lineage,
        "n_subtype_markers": n_subtype,
        "lineage_marker_support": "; ".join(high_lineage),
        "subtype_marker_support": "; ".join(high_subtype),
        "high_marker_support": "; ".join(sorted(high)),
        "low_marker_support": "; ".join(sorted(low)),
    }


def load_batch_result(spec: PlatformSpec, batch: str) -> dict:
    """Load one saved EM batch result."""
    result_dir = Path(spec.result_dir)
    cells = pd.read_csv(result_dir / f"{batch}_cell_clusters.csv")
    library = pd.read_csv(result_dir / f"{batch}_cluster_marker_library_valid.csv")
    markers = pd.read_csv(result_dir / f"{batch}_marker_params.csv")
    prob = np.load(result_dir / f"{batch}_protein_probabilities.npz", allow_pickle=True)

    marker_names = prob["marker_names"].astype(str)
    high = np.asarray(prob["high_prob"], dtype=float)
    low = np.asarray(prob["low_prob"], dtype=float)

    cells = cells.copy()
    cells["platform"] = spec.name
    cells["source_batch"] = batch
    cells["node_id"] = cells["platform"] + "|" + cells["source_batch"] + "|" + cells["protein_phenotype"].astype(str)

    library = library.copy()
    library["platform"] = spec.name
    library["source_batch"] = batch
    library["node_id"] = library["platform"] + "|" + library["source_batch"] + "|" + library["cluster"].astype(str)

    markers = markers.copy()
    markers["platform"] = spec.name
    markers["source_batch"] = batch

    return {
        "spec": spec,
        "batch": batch,
        "cells": cells,
        "library": library,
        "markers": markers,
        "marker_names": marker_names,
        "low_prob": low,
        "high_prob": high,
    }


def load_platform_results(spec: PlatformSpec) -> list[dict]:
    """Load all batches for one platform."""
    return [load_batch_result(spec, batch) for batch in spec.batches]


def build_marker_mapping(cite_records: list[dict], cytof_records: list[dict]) -> pd.DataFrame:
    """Build normalized and curated alias marker matches between CITE-seq and CyTOF."""
    cite_markers = {}
    cytof_markers = {}
    for rec in cite_records:
        for marker in rec["marker_names"]:
            cite_markers.setdefault(normalize_marker_name(marker), marker)
    for rec in cytof_records:
        for marker in rec["marker_names"]:
            cytof_markers.setdefault(normalize_marker_name(marker), marker)

    rows = []
    used_pairs = set()
    for common in sorted(set(cite_markers) & set(cytof_markers)):
        rows.append(
            {
                "common_marker": common,
                "citeseq_marker": cite_markers[common],
                "cytof_marker": cytof_markers[common],
            }
        )
        used_pairs.add((cite_markers[common], cytof_markers[common]))

    for cytof_norm, cite_norm in MARKER_ALIASES.items():
        if cytof_norm not in cytof_markers or cite_norm not in cite_markers:
            continue
        pair = (cite_markers[cite_norm], cytof_markers[cytof_norm])
        if pair in used_pairs:
            continue
        rows.append(
            {
                "common_marker": cytof_norm,
                "citeseq_marker": pair[0],
                "cytof_marker": pair[1],
            }
        )
        used_pairs.add(pair)
    return pd.DataFrame(rows)


def build_within_platform_marker_mapping(records: list[dict], platform: str) -> pd.DataFrame:
    """Build a marker mapping for one platform across many batches."""
    marker_sets = []
    marker_lookup = {}
    for rec in records:
        rec_map = {normalize_marker_name(marker): str(marker) for marker in rec["marker_names"]}
        marker_sets.append(set(rec_map))
        marker_lookup.update(rec_map)
    common = sorted(set.intersection(*marker_sets)) if marker_sets else []
    return pd.DataFrame(
        {
            "common_marker": common,
            f"{platform}_marker": [marker_lookup[m] for m in common],
        }
    )


def build_marker_comparison(cite_records: list[dict], cytof_records: list[dict]) -> pd.DataFrame:
    """Return a diagnostic table of shared and platform-specific markers."""
    cite_markers = {}
    cytof_markers = {}
    for rec in cite_records:
        for marker in rec["marker_names"]:
            cite_markers.setdefault(normalize_marker_name(marker), marker)
    for rec in cytof_records:
        for marker in rec["marker_names"]:
            cytof_markers.setdefault(normalize_marker_name(marker), marker)

    rows = []
    alias_cite_norms = set()
    alias_cytof_norms = set()
    for cytof_norm, cite_norm in MARKER_ALIASES.items():
        if cytof_norm not in cytof_markers or cite_norm not in cite_markers:
            continue
        rows.append(
            {
                "common_marker": cytof_norm,
                "citeseq_marker": cite_markers[cite_norm],
                "cytof_marker": cytof_markers[cytof_norm],
                "match_type": "shared_alias",
            }
        )
        alias_cite_norms.add(cite_norm)
        alias_cytof_norms.add(cytof_norm)

    all_norm = sorted((set(cite_markers) | set(cytof_markers)) - alias_cite_norms - alias_cytof_norms)
    for marker_norm in all_norm:
        cite_marker = cite_markers.get(marker_norm)
        cytof_marker = cytof_markers.get(marker_norm)
        if cite_marker is not None and cytof_marker is not None:
            rows.append(
                {
                    "common_marker": marker_norm,
                    "citeseq_marker": cite_marker,
                    "cytof_marker": cytof_marker,
                    "match_type": "exact_or_normalized",
                }
            )
            continue

        if cite_marker is not None:
            rows.append(
                {
                    "common_marker": marker_norm,
                    "citeseq_marker": cite_marker,
                    "cytof_marker": "",
                    "match_type": "citeseq_only",
                }
            )
        else:
            rows.append(
                {
                    "common_marker": marker_norm,
                    "citeseq_marker": "",
                    "cytof_marker": cytof_marker,
                    "match_type": "cytof_only",
                }
            )
    return pd.DataFrame(rows).sort_values(["match_type", "common_marker"]).reset_index(drop=True)


def load_batch_metrics(metrics_csv: str | Path, status: str = "complete") -> pd.DataFrame:
    """Load and lightly normalize saved per-batch phenotype metrics."""
    metrics = pd.read_csv(metrics_csv)
    if "status" in metrics.columns:
        metrics = metrics[metrics["status"] == status].copy()
    return metrics.reset_index(drop=True)


def _marker_weight_table(rec: dict, platform: str, marker_mapping: pd.DataFrame) -> pd.DataFrame:
    platform_col = f"{platform}_marker"
    mapping = marker_mapping[["common_marker", platform_col]].rename(columns={platform_col: "marker"})
    marker_params = rec["markers"].copy()
    if "phenotype_weight" not in marker_params.columns:
        marker_params["phenotype_weight"] = marker_params.get("marker_weight", 1.0)
    marker_params = marker_params[["marker", "phenotype_weight", "status"]].copy()
    marker_params["marker"] = marker_params["marker"].astype(str)
    out = mapping.merge(marker_params, on="marker", how="left")
    out["phenotype_weight"] = out["phenotype_weight"].fillna(0.0).clip(0.0, 1.0)
    out["status"] = out["status"].fillna("missing")
    return out


def build_cluster_signatures(records: list[dict], marker_mapping: pd.DataFrame) -> pd.DataFrame:
    """Summarize each cluster by mean posterior state probabilities per shared marker."""
    rows = []
    for rec in records:
        spec: PlatformSpec = rec["spec"]
        platform_col = f"{spec.name}_marker"
        mapping = marker_mapping[["common_marker", platform_col]].rename(columns={platform_col: "marker"})
        marker_index = {marker: i for i, marker in enumerate(rec["marker_names"])}
        weights = _marker_weight_table(rec, spec.name, marker_mapping)
        weight_map = weights.set_index("common_marker")["phenotype_weight"].to_dict()
        status_map = weights.set_index("common_marker")["status"].to_dict()

        cells = rec["cells"]
        cluster_values = sorted(cells["protein_phenotype"].dropna().unique())
        for cluster_id in cluster_values:
            mask = cells["protein_phenotype"].to_numpy() == cluster_id
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue
            for row in mapping.itertuples(index=False):
                marker_pos = marker_index.get(str(row.marker))
                if marker_pos is None:
                    continue
                low = float(rec["low_prob"][mask, marker_pos].mean())
                high = float(rec["high_prob"][mask, marker_pos].mean())
                probs = np.array([low, high], dtype=float)
                state_idx = int(np.argmax(probs))
                rows.append(
                    {
                        "platform": spec.name,
                        "batch": rec["batch"],
                        "cluster": int(cluster_id),
                        "node_id": f"{spec.name}|{rec['batch']}|{cluster_id}",
                        "n_cells": n_cells,
                        "common_marker": row.common_marker,
                        "platform_marker": row.marker,
                        "low_prob": low,
                        "high_prob": high,
                        "dominant_state": ("low", "high")[state_idx],
                        "state_confidence": float(probs[state_idx]),
                        "marker_weight": float(weight_map.get(row.common_marker, 0.0)),
                        "marker_status": status_map.get(row.common_marker, "missing"),
                    }
                )

    signatures = pd.DataFrame(rows)
    if signatures.empty:
        return signatures

    signatures["activation_score"] = signatures["high_prob"] - signatures["low_prob"]
    signatures["specificity"] = 0.0
    keys = ["platform", "common_marker"]
    for _, idx in signatures.groupby(keys).groups.items():
        sub = signatures.loc[idx]
        total = np.maximum(float(sub["n_cells"].sum()), 1.0)
        weighted_sum = float((sub["activation_score"] * sub["n_cells"]).sum())
        for row_idx, row in sub.iterrows():
            other_n = max(total - float(row["n_cells"]), 1.0)
            other_mean = (weighted_sum - float(row["activation_score"]) * float(row["n_cells"])) / other_n
            signatures.loc[row_idx, "specificity"] = abs(float(row["activation_score"]) - other_mean)
    signatures["specificity"] = signatures["specificity"].clip(0.0, 1.0)
    return signatures


def add_truth_summaries(signatures: pd.DataFrame, records: list[dict]) -> pd.DataFrame:
    """Attach diagnostic truth-majority labels if present in source outputs."""
    if signatures.empty:
        return signatures
    summary_rows = []
    for rec in records:
        spec: PlatformSpec = rec["spec"]
        cells = rec["cells"]
        for cluster_id, sub in cells.groupby("protein_phenotype"):
            item = {
                "node_id": f"{spec.name}|{rec['batch']}|{cluster_id}",
                "major_truth": "",
                "major_truth_fraction": np.nan,
                "minor_truth": "",
                "minor_truth_fraction": np.nan,
            }
            for source, frac_col, key in [
                ("major_truth", "major_truth_fraction", spec.major_key),
                ("minor_truth", "minor_truth_fraction", spec.minor_key),
            ]:
                if key in sub.columns:
                    vals = sub[key].fillna("Unknown").astype(str)
                    counts = vals.value_counts()
                    if len(counts):
                        item[source] = counts.index[0]
                        item[frac_col] = float(counts.iloc[0] / max(len(vals), 1))
            summary_rows.append(item)
    truth = pd.DataFrame(summary_rows).drop_duplicates("node_id")
    return signatures.merge(truth, on="node_id", how="left")


def _score_pair(cite: pd.DataFrame, cytof: pd.DataFrame) -> dict:
    """Compare independently fitted clusters through qualitative states."""

    def observed(configuration: pd.DataFrame) -> pd.DataFrame:
        separation = (
            configuration["high_prob"].astype(float)
            - configuration["low_prob"].astype(float)
        ).abs()
        return configuration.loc[
            configuration["marker_status"].astype(str).isin(
                {"valid_bimodal", "weak_bimodal", "unknown"}
            )
            & (configuration["marker_weight"].astype(float) > 0)
            & (configuration["state_confidence"].astype(float) >= 0.60)
            & (separation >= 0.15)
        ].copy()

    cite_observed = observed(cite)
    cytof_observed = observed(cytof)
    merged = cite_observed.merge(
        cytof_observed,
        on="common_marker",
        suffixes=("_cite", "_cytof"),
        how="inner",
    )
    n_cite = len(cite_observed)
    n_cytof = len(cytof_observed)
    if merged.empty:
        return {
            "score": 0.0,
            "n_common_markers": 0,
            "n_observed_citeseq": n_cite,
            "n_observed_cytof": n_cytof,
            "n_state_matches": 0,
            "n_state_conflicts": 0,
            "configuration_coverage": 0.0,
            "mean_prob_similarity": 0.0,
            "mean_state_agreement": 0.0,
            "mean_direction_agreement": 0.0,
            "mean_specificity": 0.0,
            "top_markers": "",
        }

    state_agree = (
        merged["dominant_state_cite"].to_numpy()
        == merged["dominant_state_cytof"].to_numpy()
    )
    n_matches = int(state_agree.sum())
    n_conflicts = int((~state_agree).sum())
    agreement = n_matches / max(len(merged), 1)
    coverage = len(merged) / max(n_cite, n_cytof, 1)
    score = float(agreement * coverage)
    top = merged.assign(_match=state_agree).sort_values(
        ["_match", "common_marker"],
        ascending=[False, True],
    ).head(12)
    return {
        "score": score,
        "n_common_markers": int(len(merged)),
        "n_observed_citeseq": int(n_cite),
        "n_observed_cytof": int(n_cytof),
        "n_state_matches": n_matches,
        "n_state_conflicts": n_conflicts,
        "configuration_coverage": float(coverage),
        "mean_prob_similarity": float(agreement),
        "mean_state_agreement": float(agreement),
        "mean_direction_agreement": float(agreement),
        "mean_specificity": float(coverage),
        "top_markers": "; ".join(
            f"{row.common_marker}:{row.dominant_state_cite}/"
            f"{row.dominant_state_cytof}"
            for row in top.itertuples(index=False)
        ),
    }


def build_score_table(signatures: pd.DataFrame) -> pd.DataFrame:
    """Score every CITE-seq cluster against every CyTOF cluster."""
    cite_groups = {node: sub for node, sub in signatures[signatures["platform"] == "citeseq"].groupby("node_id")}
    cytof_groups = {node: sub for node, sub in signatures[signatures["platform"] == "cytof"].groupby("node_id")}
    rows = []
    for cite_node, cite_sig in cite_groups.items():
        _, cite_batch, cite_cluster = cite_node.split("|")
        for cytof_node, cytof_sig in cytof_groups.items():
            _, cytof_batch, cytof_cluster = cytof_node.split("|")
            score = _score_pair(cite_sig, cytof_sig)
            rows.append(
                {
                    "cite_node_id": cite_node,
                    "cite_batch": cite_batch,
                    "cite_cluster": int(cite_cluster),
                    "cytof_node_id": cytof_node,
                    "cytof_batch": cytof_batch,
                    "cytof_cluster": int(cytof_cluster),
                    **score,
                }
            )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def call_matches(
    scores: pd.DataFrame,
    min_score: float = 0.78,
    relaxed_score: float = 0.66,
    ambiguous_score: float = 0.62,
    min_margin: float = 0.015,
    min_common_markers: int = 3,
    min_configuration_coverage: float = 0.60,
) -> pd.DataFrame:
    """Call confident and ambiguous cross-platform cluster matches."""
    if scores.empty:
        return scores
    out = scores.copy()
    out["cite_rank"] = out.groupby("cite_node_id")["score"].rank(method="first", ascending=False)
    out["cytof_rank"] = out.groupby("cytof_node_id")["score"].rank(method="first", ascending=False)
    cite_second = out[out["cite_rank"] == 2].set_index("cite_node_id")["score"]
    cytof_second = out[out["cytof_rank"] == 2].set_index("cytof_node_id")["score"]
    out["cite_margin"] = out["score"] - out["cite_node_id"].map(cite_second).fillna(0.0)
    out["cytof_margin"] = out["score"] - out["cytof_node_id"].map(cytof_second).fillna(0.0)
    mutual_best = (out["cite_rank"] == 1) & (out["cytof_rank"] == 1)
    one_sided_best = (out["cite_rank"] == 1) | (out["cytof_rank"] == 1)
    enough_margin = (out["cite_margin"] >= min_margin) | (out["cytof_margin"] >= min_margin)
    state_consistent = (
        (out["n_common_markers"] >= min_common_markers)
        & (out["n_state_conflicts"] == 0)
        & (out["configuration_coverage"] >= min_configuration_coverage)
    )
    out["match_status"] = "low_similarity"
    out.loc[
        (out["score"] >= ambiguous_score)
        & one_sided_best
        & state_consistent,
        "match_status",
    ] = "ambiguous_candidate"
    out.loc[
        (out["score"] >= relaxed_score) & one_sided_best & state_consistent,
        "match_status",
    ] = "matched_relaxed"
    out.loc[
        (out["score"] >= min_score)
        & one_sided_best
        & enough_margin
        & state_consistent,
        "match_status",
    ] = "matched"
    out.loc[
        (out["score"] >= min_score) & mutual_best & state_consistent,
        "match_status",
    ] = "matched_mutual_best"
    return out.sort_values(["match_status", "score"], ascending=[True, False]).reset_index(drop=True)


def build_within_platform_consensus(
    result_dir: str | Path,
    batch_ids: Iterable[str],
    platform: str,
    major_key: str,
    minor_key: str,
    max_markers: int | None = None,
    min_marker_weight: float = 0.25,
    min_score: float | None = None,
    secondary_score: float | None = None,
    strong_score: float | None = None,
    min_margin: float = 0.02,
    absorb_score: float | None = None,
    top_k_edges: int = 8,
    chunk_size: int = 128,
    weak_conflict_tolerance: int = 0,
    strong_conflict_reliability: float = 0.18,
    tolerant_score_delta: float = 0.08,
    min_observed_markers: int = 2,
    min_state_confidence: float = 0.60,
    min_state_separation: float = 0.15,
    min_containment: float | None = None,
    min_jaccard: float | None = None,
    min_configuration_coverage: float = 0.60,
) -> dict[str, pd.DataFrame]:
    """Build within-platform consensus phenotypes from batch-local signatures.

    CITE-seq and CyTOF batches within the same platform share the same marker
    panel, so this step performs normal cross-batch alignment. Qualitative
    low/high marker states are converted into reliable marker-state sets.
    Missing or weak states provide no evidence. Reciprocal configuration
    coverage is required, so a coarse local cluster is not forced into a finer
    phenotype merely because its small observed subset is contained there.
    """
    result_dir = Path(result_dir)
    library_frames = []
    cells_frames = []
    marker_frames = []
    batch_ids = [str(x) for x in batch_ids]
    if platform.lower() == "cytof":
        min_score = 0.78 if min_score is None else min_score
        secondary_score = 0.86 if secondary_score is None else secondary_score
        strong_score = 0.92 if strong_score is None else strong_score
        absorb_score = 0.86 if absorb_score is None else absorb_score
    else:
        min_score = 0.78 if min_score is None else min_score
        secondary_score = 0.86 if secondary_score is None else secondary_score
        strong_score = 0.92 if strong_score is None else strong_score
        absorb_score = 0.86 if absorb_score is None else absorb_score
    min_containment = min_score if min_containment is None else float(min_containment)
    min_jaccard = 0.62 if min_jaccard is None else float(min_jaccard)

    for batch_id in batch_ids:
        library = pd.read_csv(result_dir / f"{batch_id}_cluster_marker_library_valid.csv", low_memory=False)
        library["batch"] = str(batch_id)
        library["platform"] = platform
        library["node_id"] = platform + "|" + library["batch"].astype(str) + "|" + library["cluster"].astype(str)
        for col in ["positive_markers", "negative_markers", "selected_markers"]:
            if col in library.columns:
                library[col] = library[col].apply(parse_list)
            else:
                library[col] = [[] for _ in range(len(library))]
        library_frames.append(library)

        cells = pd.read_csv(result_dir / f"{batch_id}_cell_clusters.csv", low_memory=False)
        cells["cell_barcode"] = cells["cell_barcode"].astype(str)
        cells["source_batch"] = str(batch_id)
        cells["node_id"] = platform + "|" + cells["source_batch"].astype(str) + "|" + cells["protein_phenotype"].astype(str)
        cells_frames.append(cells)

        markers = pd.read_csv(result_dir / f"{batch_id}_marker_params.csv", low_memory=False)
        markers["batch"] = str(batch_id)
        for col in ["phenotype_weight", "marker_weight"]:
            if col in markers.columns:
                markers[col] = pd.to_numeric(markers[col], errors="coerce")
        if "phenotype_weight" not in markers.columns and "marker_weight" in markers.columns:
            markers["phenotype_weight"] = markers["marker_weight"]
        if "status" not in markers.columns:
            markers["status"] = "unknown"
        marker_frames.append(markers[["batch", "marker", "phenotype_weight", "status"]])

    library_df = pd.concat(library_frames, ignore_index=True)
    cells_df = pd.concat(cells_frames, ignore_index=True)
    marker_df = pd.concat(marker_frames, ignore_index=True)
    marker_df["phenotype_weight"] = pd.to_numeric(marker_df["phenotype_weight"], errors="coerce").fillna(0.0).clip(0, 1)

    marker_score = (
        marker_df.groupby("marker")["phenotype_weight"]
        .max()
        .sort_values(ascending=False)
    )
    eligible_markers = marker_score[marker_score >= min_marker_weight]
    marker_rank = (
        eligible_markers.head(max_markers).index.tolist()
        if max_markers is not None
        else eligible_markers.index.tolist()
    )
    if len(marker_rank) < min(8, len(marker_score)):
        marker_rank = (
            marker_score.head(max_markers).index.tolist()
            if max_markers is not None
            else marker_score.index.tolist()
        )

    meta = library_df[["node_id", "platform", "batch", "cluster", "n_cells"]].drop_duplicates("node_id").copy()
    truth_rows = []
    for node_id, sub in cells_df.groupby("node_id", sort=False):
        item = {"node_id": node_id, "major_truth": "", "minor_truth": ""}
        for out_key, obs_key in [("major_truth", major_key), ("minor_truth", minor_key)]:
            if obs_key in sub.columns:
                values = sub[obs_key].fillna("Unknown").astype(str)
                item[out_key] = values.value_counts().index[0] if len(values) else ""
        truth_rows.append(item)
    truth_df = pd.DataFrame(truth_rows)
    node_meta = meta.merge(truth_df, on="node_id", how="left")

    sig_rows = []
    marker_pos = {marker: i for i, marker in enumerate(marker_rank)}
    for batch_id in batch_ids:
        prob = np.load(result_dir / f"{batch_id}_protein_probabilities.npz", allow_pickle=True)
        marker_names = prob["marker_names"].astype(str).tolist()
        idx = [marker_names.index(marker) for marker in marker_rank if marker in marker_names]
        used_markers = [marker for marker in marker_rank if marker in marker_names]
        if not used_markers:
            continue
        low = np.asarray(prob["low_prob"][:, idx], dtype=float)
        high = np.asarray(prob["high_prob"][:, idx], dtype=float)
        cells = cells_df[cells_df["source_batch"] == str(batch_id)].reset_index(drop=True)
        params = marker_df[marker_df["batch"] == str(batch_id)].set_index("marker")["phenotype_weight"].to_dict()
        statuses = marker_df[marker_df["batch"] == str(batch_id)].set_index("marker")["status"].to_dict()
        for cluster_id, group_idx in cells.groupby("protein_phenotype", sort=False).groups.items():
            rows = np.asarray(list(group_idx), dtype=int)
            if rows.size == 0:
                continue
            node_id = f"{platform}|{batch_id}|{cluster_id}"
            mean_low = low[rows].mean(axis=0)
            mean_high = high[rows].mean(axis=0)
            for local_pos, marker in enumerate(used_markers):
                probs = np.array([mean_low[local_pos], mean_high[local_pos]], dtype=float)
                state_idx = int(np.argmax(probs))
                sig_rows.append(
                    {
                        "node_id": node_id,
                        "platform": platform,
                        "batch": str(batch_id),
                        "cluster": cluster_id,
                        "n_cells": int(rows.size),
                        "common_marker": marker,
                        "low_prob": float(probs[0]),
                        "high_prob": float(probs[1]),
                        "dominant_state": ("low", "high")[state_idx],
                        "state_confidence": float(probs[state_idx]),
                        "marker_weight": float(params.get(marker, 0.0)),
                        "marker_status": str(statuses.get(marker, "unknown")),
                    }
                )
    state_df = pd.DataFrame(sig_rows)
    if state_df.empty:
        node_meta["consensus_phenotype"] = [f"{platform.upper()}_CP_{i + 1:03d}" for i in range(len(node_meta))]
        node_meta["is_multi_batch"] = False
        cells = cells_df.merge(node_meta[["node_id", "consensus_phenotype"]], on="node_id", how="left")
        cells["batch"] = cells["source_batch"].astype(str)
        return {
            "marker_mapping": pd.DataFrame({"common_marker": marker_rank, f"{platform}_marker": marker_rank}),
            "cluster_signatures": state_df,
            "scores": pd.DataFrame(),
            "matches": pd.DataFrame(),
            "node_meta": node_meta,
            "cells": cells,
            "marker_rank": pd.DataFrame({"common_marker": marker_rank}),
        }

    state_df["activation_score"] = state_df["high_prob"] - state_df["low_prob"]
    state_df["specificity"] = 0.0
    for marker, idx in state_df.groupby("common_marker").groups.items():
        sub = state_df.loc[idx]
        weights = np.maximum(sub["n_cells"].to_numpy(float), 1.0)
        total_w = weights.sum()
        values = sub["activation_score"].to_numpy(float)
        weighted_total = float((values * weights).sum())
        other_mean = (weighted_total - values * weights) / np.maximum(total_w - weights, 1.0)
        state_df.loc[idx, "specificity"] = np.abs(values - other_mean)
    state_df["specificity"] = state_df["specificity"].clip(0.0, 1.0)

    nodes = node_meta["node_id"].astype(str).tolist()
    node_index = {node: i for i, node in enumerate(nodes)}
    marker_index = {marker: i for i, marker in enumerate(marker_rank)}
    n_nodes = len(nodes)
    n_markers = len(marker_rank)
    probs = np.zeros((n_nodes, n_markers, 2), dtype=np.float32)
    weights = np.zeros((n_nodes, n_markers), dtype=np.float32)
    specificity = np.zeros((n_nodes, n_markers), dtype=np.float32)
    available = np.zeros((n_nodes, n_markers), dtype=bool)
    state_confidence = np.zeros((n_nodes, n_markers), dtype=np.float32)
    marker_status = np.full((n_nodes, n_markers), "", dtype=object)
    for row in state_df.itertuples(index=False):
        i = node_index.get(str(row.node_id))
        j = marker_index.get(str(row.common_marker))
        if i is None or j is None:
            continue
        probs[i, j, :] = [row.low_prob, row.high_prob]
        weights[i, j] = row.marker_weight
        specificity[i, j] = row.specificity
        state_confidence[i, j] = row.state_confidence
        marker_status[i, j] = str(getattr(row, "marker_status", "unknown"))
        available[i, j] = True
    dominant = probs.argmax(axis=2)
    direction = np.sign(probs[:, :, 1] - probs[:, :, 0])
    state_separation = np.abs(probs[:, :, 1] - probs[:, :, 0])
    marker_reliability = weights * specificity * state_separation
    reliable_status = np.isin(marker_status, ["valid_bimodal", "weak_bimodal", "unknown"])
    observed = (
        available
        & reliable_status
        & (weights > 0)
        & (state_confidence >= min_state_confidence)
        & (state_separation >= min_state_separation)
    )
    batch_values = node_meta["batch"].astype(str).to_numpy()

    norm_to_idx = {normalize_marker_name(marker): idx for idx, marker in enumerate(marker_rank)}

    def max_activation(row_idx: int, names: tuple[str, ...]) -> float:
        vals = []
        for name in names:
            marker_idx = norm_to_idx.get(normalize_marker_name(name))
            if marker_idx is not None and available[row_idx, marker_idx]:
                vals.append(float(probs[row_idx, marker_idx, 1] - probs[row_idx, marker_idx, 0]))
        return max(vals) if vals else 0.0

    coarse_labels = []
    for row_idx in range(n_nodes):
        t_score = max_activation(row_idx, ("CD3", "CD5", "CD7", "ABTCR"))
        cd4_score = max_activation(row_idx, ("CD4",))
        cd8_score = max_activation(row_idx, ("CD8",))
        b_score = max_activation(row_idx, ("CD19", "CD20", "IGM", "IGD"))
        myeloid_score = max_activation(row_idx, ("CD14", "CD33", "CD11C", "CLEC12A", "CD64"))
        nk_score = max_activation(row_idx, ("CD56", "CD16", "KLRG1", "CD57"))
        pdc_score = max_activation(row_idx, ("CD123", "HLADR"))
        if b_score > 0.25:
            label = "B"
        elif t_score > 0.25:
            if cd4_score > 0.20 and cd8_score <= 0.20:
                label = "T_CD4"
            elif cd8_score > 0.20 and cd4_score <= 0.20:
                label = "T_CD8"
            elif cd4_score > 0.20 and cd8_score > 0.20:
                label = "T_DP"
            else:
                label = "T_OTHER"
        elif myeloid_score > 0.25:
            label = "MYELOID"
        elif nk_score > 0.25:
            label = "NK"
        elif pdc_score > 0.25:
            label = "PDC"
        else:
            label = "UNKNOWN"
        coarse_labels.append(label)
    node_meta["coarse_lineage"] = coarse_labels

    scores = np.full((n_nodes, n_nodes), -np.inf, dtype=np.float32)
    jaccard_scores = np.full((n_nodes, n_nodes), -np.inf, dtype=np.float32)
    containment_scores = np.full((n_nodes, n_nodes), -np.inf, dtype=np.float32)
    weak_conflict_counts = np.zeros((n_nodes, n_nodes), dtype=np.int16)
    strong_conflict_counts = np.zeros((n_nodes, n_nodes), dtype=np.int16)
    state_conflict_counts = np.zeros((n_nodes, n_nodes), dtype=np.int16)
    state_match_counts = np.zeros((n_nodes, n_nodes), dtype=np.int16)
    union_state_counts = np.zeros((n_nodes, n_nodes), dtype=np.int16)
    min_observed_counts = np.zeros((n_nodes, n_nodes), dtype=np.int16)
    configuration_coverage = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for start in range(0, n_nodes, chunk_size):
        stop = min(start + chunk_size, n_nodes)
        state_agree = (dominant[start:stop, None, :] == dominant[None, :, :]).astype(np.float32)
        shared = observed[start:stop, None, :] & observed[None, :, :]
        observed_union = observed[start:stop, None, :] | observed[None, :, :]
        state_conflict = shared & (state_agree < 0.5)
        pair_reliability = 0.5 * (
            marker_reliability[start:stop, None, :]
            + marker_reliability[None, :, :]
        )
        weak_conflict = state_conflict & (pair_reliability < strong_conflict_reliability)
        strong_conflict = state_conflict & (pair_reliability >= strong_conflict_reliability)
        state_match = shared & (state_agree > 0.5)
        match_counts = state_match.sum(axis=2).astype(np.float32)
        union_counts = observed_union.sum(axis=2).astype(np.float32)
        observed_a = observed[start:stop].sum(axis=1).astype(np.float32)[:, None]
        observed_b = observed.sum(axis=1).astype(np.float32)[None, :]
        min_counts = np.minimum(observed_a, observed_b)
        max_counts = np.maximum(observed_a, observed_b)
        chunk_coverage = shared.sum(axis=2) / np.maximum(max_counts, 1.0)
        chunk_jaccard = match_counts / np.maximum(union_counts, 1.0)
        chunk_containment = match_counts / np.maximum(min_counts, 1.0)
        chunk_scores = np.maximum(chunk_containment, chunk_jaccard)
        shared_counts = shared.sum(axis=2)
        invalid = batch_values[start:stop, None] == batch_values[None, :]
        invalid |= shared_counts < max(1, int(min_observed_markers))
        invalid |= min_counts < max(1, int(min_observed_markers))
        invalid |= chunk_coverage < float(min_configuration_coverage)
        coarse_a = node_meta["coarse_lineage"].to_numpy()[start:stop, None]
        coarse_b = node_meta["coarse_lineage"].to_numpy()[None, :]
        coarse_ok = (
            (coarse_a == coarse_b)
            | (coarse_a == "UNKNOWN")
            | (coarse_b == "UNKNOWN")
        )
        invalid |= ~coarse_ok
        chunk_scores[invalid] = -np.inf
        chunk_jaccard[invalid] = -np.inf
        chunk_containment[invalid] = -np.inf
        scores[start:stop] = chunk_scores.astype(np.float32)
        jaccard_scores[start:stop] = chunk_jaccard.astype(np.float32)
        containment_scores[start:stop] = chunk_containment.astype(np.float32)
        weak_conflict_counts[start:stop] = weak_conflict.sum(axis=2).astype(np.int16)
        strong_conflict_counts[start:stop] = strong_conflict.sum(axis=2).astype(np.int16)
        state_conflict_counts[start:stop] = state_conflict.sum(axis=2).astype(np.int16)
        state_match_counts[start:stop] = match_counts.astype(np.int16)
        union_state_counts[start:stop] = union_counts.astype(np.int16)
        min_observed_counts[start:stop] = min_counts.astype(np.int16)
        configuration_coverage[start:stop] = chunk_coverage.astype(np.float32)

    shared_marker_counts = observed.astype(np.int16) @ observed.astype(np.int16).T

    best_idx = np.argmax(scores, axis=1)
    best_score = scores[np.arange(n_nodes), best_idx]
    safe_scores = scores.copy()
    safe_scores[np.arange(n_nodes), best_idx] = -np.inf
    second_score = np.max(safe_scores, axis=1)
    edges = []
    match_rows = []
    for i in range(n_nodes):
        if not np.isfinite(best_score[i]):
            continue
        ranked = np.argsort(scores[i])[::-1]
        ranked = ranked[np.isfinite(scores[i, ranked])]
        candidates = ranked[:top_k_edges]
        candidates = np.unique(np.r_[candidates, np.flatnonzero(scores[i] >= secondary_score), best_idx[i]])
        for j in candidates:
            if i >= j or not np.isfinite(scores[i, j]):
                continue
            score = float(max(scores[i, j], scores[j, i]))
            jaccard = float(max(jaccard_scores[i, j], jaccard_scores[j, i]))
            containment = float(max(containment_scores[i, j], containment_scores[j, i]))
            weak_conflicts = int(max(weak_conflict_counts[i, j], weak_conflict_counts[j, i]))
            strong_conflicts = int(max(strong_conflict_counts[i, j], strong_conflict_counts[j, i]))
            state_conflicts = int(max(state_conflict_counts[i, j], state_conflict_counts[j, i]))
            reciprocal_coverage = float(
                max(configuration_coverage[i, j], configuration_coverage[j, i])
            )
            conflict_tolerant = (
                state_conflicts == 0
                and weak_conflicts <= weak_conflict_tolerance
            )
            set_compatible = (
                conflict_tolerant
                and (
                    containment >= min_containment
                    or jaccard >= min_jaccard
                    or score >= max(min_score - tolerant_score_delta, 0.0)
                )
            )
            if not set_compatible:
                continue
            row_best = int(best_idx[i]) == int(j)
            col_best = int(best_idx[j]) == int(i)
            row_margin = float(scores[i, j] - second_score[i]) if row_best else float(scores[i, j] - best_score[i])
            col_margin = float(scores[j, i] - second_score[j]) if col_best else float(scores[j, i] - best_score[j])
            mutual = row_best and col_best
            strong = containment >= secondary_score and (row_best or col_best) and max(row_margin, col_margin) >= min_margin
            very_high = score >= strong_score
            tolerant = conflict_tolerant and (row_best or col_best or containment >= secondary_score or jaccard >= min_jaccard)
            if mutual or strong or very_high or tolerant:
                edges.append((nodes[i], nodes[j]))
                match_rows.append(
                    {
                        "node_a": nodes[i],
                        "node_b": nodes[j],
                        "score": score,
                        "state_jaccard": jaccard,
                        "state_containment": containment,
                        "margin_a": row_margin,
                        "margin_b": col_margin,
                        "row_best": row_best,
                        "col_best": col_best,
                        "mutual_best": mutual,
                        "n_common_markers": int(shared_marker_counts[i, j]),
                        "n_state_matches": int(max(state_match_counts[i, j], state_match_counts[j, i])),
                        "n_state_union": int(max(union_state_counts[i, j], union_state_counts[j, i])),
                        "n_min_observed_states": int(max(min_observed_counts[i, j], min_observed_counts[j, i])),
                        "n_state_conflicts": state_conflicts,
                        "configuration_coverage": reciprocal_coverage,
                        "n_weak_state_conflicts": weak_conflicts,
                        "n_strong_state_conflicts": strong_conflicts,
                        "match_status": (
                            "matched_mutual_best"
                            if mutual
                            else "matched_set_containment"
                            if containment >= min_containment
                            else "matched_tolerant_weak_conflict"
                            if tolerant
                            else "matched_many_to_one"
                        ),
                    }
                )

    comp_map = _connected_components(edges)
    next_comp = max(comp_map.values(), default=0) + 1
    for node in nodes:
        if node not in comp_map:
            comp_map[node] = next_comp
            next_comp += 1
    node_meta["component_id"] = node_meta["node_id"].map(comp_map).astype(int)

    support = node_meta.groupby("component_id")["batch"].nunique()
    multi_components = set(support[support > 1].index)
    for idx, row in node_meta.iterrows():
        comp = int(row["component_id"])
        if comp in multi_components:
            continue
        i = node_index[row["node_id"]]
        ranked = np.argsort(scores[i])[::-1]
        for j in ranked[:20]:
            if not np.isfinite(scores[i, j]) or scores[i, j] < absorb_score:
                break
            if state_conflict_counts[i, j] != 0:
                continue
            if configuration_coverage[i, j] < min_configuration_coverage:
                continue
            target_comp = int(node_meta.iloc[j]["component_id"])
            if target_comp in multi_components:
                node_meta.at[idx, "component_id"] = target_comp
                break

    comp_order = (
        node_meta.groupby("component_id")["n_cells"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    comp_names = {comp: f"{platform.upper()}_CP_{pos + 1:03d}" for pos, comp in enumerate(comp_order)}
    node_meta["consensus_phenotype"] = node_meta["component_id"].map(comp_names)
    node_meta["is_multi_batch"] = node_meta["consensus_phenotype"].map(
        node_meta.groupby("consensus_phenotype")["batch"].nunique()
    ) > 1
    node_meta["best_match_score"] = best_score

    cells = cells_df.merge(node_meta[["node_id", "consensus_phenotype"]], on="node_id", how="left")
    cells["batch"] = cells["source_batch"].astype(str)
    matches = pd.DataFrame(match_rows).sort_values("score", ascending=False).reset_index(drop=True) if match_rows else pd.DataFrame()

    return {
        "marker_mapping": pd.DataFrame({"common_marker": marker_rank, f"{platform}_marker": marker_rank}),
        "cluster_signatures": state_df,
        "scores": matches,
        "matches": matches,
        "node_meta": node_meta,
        "cells": cells,
        "marker_rank": pd.DataFrame({"common_marker": marker_rank}),
    }


def _connected_components(edges: Iterable[tuple[str, str]]) -> dict[str, int]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    roots = {node: find(node) for node in parent}
    root_ids = {root: i + 1 for i, root in enumerate(sorted(set(roots.values())))}
    return {node: root_ids[root] for node, root in roots.items()}


def build_consensus(
    signatures: pd.DataFrame,
    matches: pd.DataFrame,
    min_edge_score: float = 0.66,
    edge_statuses: Iterable[str] = ("matched", "matched_mutual_best", "matched_relaxed"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create cross-platform phenotype components and marker summaries."""
    edge_statuses = set(edge_statuses)
    edges = matches[
        (matches["score"] >= min_edge_score)
        & matches["match_status"].isin(edge_statuses)
    ][["cite_node_id", "cytof_node_id"]]
    comp_map = _connected_components(edges.itertuples(index=False, name=None))

    all_nodes = sorted(signatures["node_id"].unique())
    next_id = max(comp_map.values(), default=0) + 1
    for node in all_nodes:
        if node not in comp_map:
            comp_map[node] = next_id
            next_id += 1

    node_meta = (
        signatures[
            [
                "node_id",
                "platform",
                "batch",
                "cluster",
                "n_cells",
                "major_truth",
                "major_truth_fraction",
                "minor_truth",
                "minor_truth_fraction",
            ]
        ]
        .drop_duplicates("node_id")
        .copy()
    )
    node_meta["cross_platform_phenotype"] = node_meta["node_id"].map(lambda x: f"CP_{comp_map[x]:03d}")
    node_meta["is_cross_platform"] = node_meta["cross_platform_phenotype"].duplicated(keep=False)

    marker_rows = []
    for phenotype, nodes in node_meta.groupby("cross_platform_phenotype"):
        sub_sig = signatures[signatures["node_id"].isin(nodes["node_id"])]
        for marker, marker_sig in sub_sig.groupby("common_marker"):
            w = marker_sig["n_cells"].to_numpy(float)
            w = np.maximum(w, 1.0)
            low = float(np.average(marker_sig["low_prob"], weights=w))
            high = float(np.average(marker_sig["high_prob"], weights=w))
            probs = np.array([low, high])
            state = ("low", "high")[int(np.argmax(probs))]
            marker_rows.append(
                {
                    "cross_platform_phenotype": phenotype,
                    "common_marker": marker,
                    "low_prob": low,
                    "high_prob": high,
                    "dominant_state": state,
                    "state_confidence": float(probs.max()),
                    "mean_specificity": float(np.average(marker_sig["specificity"], weights=w)),
                    "mean_marker_weight": float(np.average(marker_sig["marker_weight"], weights=w)),
                }
            )
    marker_summary = pd.DataFrame(marker_rows)

    library_rows = []
    for phenotype, nodes in node_meta.groupby("cross_platform_phenotype"):
        markers = marker_summary[marker_summary["cross_platform_phenotype"] == phenotype].copy()
        informative = markers[
            (markers["state_confidence"] >= 0.65)
            & (markers["mean_marker_weight"] >= 0.10)
            & (markers["mean_specificity"] >= 0.08)
        ]
        high = informative[informative["dominant_state"] == "high"].sort_values(
            ["mean_specificity", "state_confidence"], ascending=False
        )["common_marker"].head(12).tolist()
        low = informative[informative["dominant_state"] == "low"].sort_values(
            ["mean_specificity", "state_confidence"], ascending=False
        )["common_marker"].head(12).tolist()
        library_rows.append(
            {
                "cross_platform_phenotype": phenotype,
                "platforms": ",".join(sorted(nodes["platform"].unique())),
                "n_member_clusters": int(len(nodes)),
                "n_cells": int(nodes["n_cells"].sum()),
                "members": "; ".join(nodes["node_id"].astype(str)),
                "high_markers": "; ".join(high),
                "low_markers": "; ".join(low),
                "diagnostic_major_truth": "; ".join(
                    nodes.sort_values("n_cells", ascending=False)["major_truth"].fillna("").astype(str).head(6)
                ),
                "diagnostic_minor_truth": "; ".join(
                    nodes.sort_values("n_cells", ascending=False)["minor_truth"].fillna("").astype(str).head(6)
                ),
            }
        )
    library = pd.DataFrame(library_rows).sort_values(["platforms", "n_cells"], ascending=[True, False])
    return library.reset_index(drop=True), marker_summary.reset_index(drop=True), node_meta.reset_index(drop=True)


def annotate_cells(records: list[dict], node_meta: pd.DataFrame) -> pd.DataFrame:
    """Attach cross-platform phenotype IDs to cells."""
    mapping = node_meta.set_index("node_id")["cross_platform_phenotype"].to_dict()
    frames = []
    for rec in records:
        cells = rec["cells"].copy()
        cells["cross_platform_phenotype"] = cells["node_id"].map(mapping).fillna("unmatched")
        frames.append(cells)
    return pd.concat(frames, ignore_index=True)


def run_integration(
    cite_dir: str | Path,
    cytof_dir: str | Path,
    cite_batches: Iterable[str] = ("S00052", "N00023"),
    cytof_batches: Iterable[str] = ("A_1", "B_1"),
    output_dir: str | Path | None = None,
    match_min_score: float = 0.78,
    match_relaxed_score: float = 0.66,
    match_ambiguous_score: float = 0.62,
    match_min_margin: float = 0.015,
    consensus_min_edge_score: float = 0.66,
    include_ambiguous_edges: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the cross-platform integration and optionally save CSV outputs."""
    cite_spec = PlatformSpec("citeseq", Path(cite_dir), tuple(cite_batches), "subtype", "minor_subset")
    cytof_spec = PlatformSpec("cytof", Path(cytof_dir), tuple(cytof_batches), "major_cell_type", "fine_cluster_id")
    cite_records = load_platform_results(cite_spec)
    cytof_records = load_platform_results(cytof_spec)
    records = cite_records + cytof_records

    marker_mapping = build_marker_mapping(cite_records, cytof_records)
    marker_comparison = build_marker_comparison(cite_records, cytof_records)
    signatures = build_cluster_signatures(records, marker_mapping)
    signatures = add_truth_summaries(signatures, records)
    align_signatures = signatures.copy()
    scores = build_score_table(align_signatures)
    matches = call_matches(
        scores,
        min_score=match_min_score,
        relaxed_score=match_relaxed_score,
        ambiguous_score=match_ambiguous_score,
        min_margin=match_min_margin,
    )
    edge_statuses = ["matched", "matched_mutual_best", "matched_relaxed"]
    if include_ambiguous_edges:
        edge_statuses.append("ambiguous_candidate")
    library, marker_summary, node_meta = build_consensus(
        align_signatures,
        matches,
        min_edge_score=consensus_min_edge_score,
        edge_statuses=edge_statuses,
    )
    cells = annotate_cells(records, node_meta)
    cells = add_coarse_truth_labels(cells)
    atlas_outputs = build_marker_state_atlas(
        align_signatures,
        source_col="platform",
        min_shared_markers=3,
        compare_within_source=False,
    )
    atlas_outputs.pop("cluster_signatures", None)

    outputs = {
        "marker_mapping": marker_mapping,
        "marker_comparison": marker_comparison,
        "cluster_signatures": align_signatures,
        "scores": scores,
        "matches": matches,
        "consensus_library": library,
        "consensus_marker_summary": marker_summary,
        "cluster_to_consensus": node_meta,
        "cells": cells,
        **atlas_outputs,
    }
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, df in outputs.items():
            df.to_csv(out / f"{name}.csv", index=False)
    return outputs


def plot_score_heatmap(matches: pd.DataFrame, figsize: tuple[float, float] = (14, 8), title: str | None = None):
    """Plot CITE-seq by CyTOF cluster matching scores."""
    pivot = matches.pivot_table(index="cite_node_id", columns="cytof_node_id", values="score", aggfunc="max")
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]
    pivot = pivot[pivot.max(axis=0).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("CyTOF cluster")
    ax.set_ylabel("CITE-seq cluster")
    ax.set_title(title or "Cross-platform cluster similarity")
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    return fig, ax


def plot_consensus_marker_heatmap(
    marker_summary: pd.DataFrame,
    library: pd.DataFrame,
    max_phenotypes: int = 30,
    max_markers: int = 30,
    figsize: tuple[float, float] = (14, 8),
):
    """Plot activation scores for the most informative consensus markers."""
    keep_pheno = library.sort_values("n_cells", ascending=False)["cross_platform_phenotype"].head(max_phenotypes)
    sub = marker_summary[marker_summary["cross_platform_phenotype"].isin(keep_pheno)].copy()
    sub["activation_score"] = sub["high_prob"] - sub["low_prob"]
    marker_rank = (
        sub.assign(info=sub["mean_specificity"] * sub["mean_marker_weight"])
        .groupby("common_marker")["info"]
        .mean()
        .sort_values(ascending=False)
        .head(max_markers)
        .index
    )
    mat = sub[sub["common_marker"].isin(marker_rank)].pivot_table(
        index="cross_platform_phenotype",
        columns="common_marker",
        values="activation_score",
        aggfunc="mean",
    )
    mat = mat.reindex(keep_pheno).dropna(how="all")
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.fillna(0.0).to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xlabel("shared marker")
    ax.set_ylabel("cross-platform phenotype")
    ax.set_title("Consensus phenotype marker activation")
    fig.colorbar(im, ax=ax, label="high posterior - low posterior")
    fig.tight_layout()
    return fig, ax


def build_cross_platform_configuration(
    library: pd.DataFrame,
    marker_summary: pd.DataFrame,
    min_marker_confidence: float = 0.65,
    min_marker_weight: float = 0.10,
    min_marker_specificity: float = 0.08,
) -> pd.DataFrame:
    """Build phenotype configuration table with marker evidence and resolution."""
    rows = []
    for phenotype, markers in marker_summary.groupby("cross_platform_phenotype"):
        info = library[library["cross_platform_phenotype"].astype(str).eq(str(phenotype))]
        if info.empty:
            continue
        info_row = info.iloc[0]
        selected = markers[
            (markers["state_confidence"] >= min_marker_confidence)
            & (markers["mean_marker_weight"] >= min_marker_weight)
            & (markers["mean_specificity"] >= min_marker_specificity)
        ].copy()
        high = selected[selected["dominant_state"].eq("high")]["common_marker"].tolist()
        low = selected[selected["dominant_state"].eq("low")]["common_marker"].tolist()
        config = infer_marker_configuration(high, low)
        rows.append(
            {
                "phenotype": phenotype,
                "platforms": info_row.get("platforms", ""),
                "n_member_clusters": info_row.get("n_member_clusters", np.nan),
                "n_cells": info_row.get("n_cells", np.nan),
                "diagnostic_major_truth": info_row.get("diagnostic_major_truth", ""),
                "diagnostic_minor_truth": info_row.get("diagnostic_minor_truth", ""),
                **config,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["resolution_level", "n_cells"], ascending=[True, False]).reset_index(drop=True)
    return out


def build_platform_configuration(result_dir: str | Path, platform: str) -> pd.DataFrame:
    """Build configuration table for within-platform consensus phenotypes."""
    result_dir = Path(result_dir)
    nodes = pd.read_csv(result_dir / "platform_consensus_nodes.csv", low_memory=False)
    rows = []
    for phenotype, group in nodes.groupby("consensus_phenotype"):
        high_counts: dict[str, int] = {}
        low_counts: dict[str, int] = {}
        n_cells = int(group["n_cells"].sum())
        for row in group.itertuples(index=False):
            path = result_dir / f"{row.batch}_cluster_marker_library_valid.csv"
            if not path.exists():
                continue
            lib = pd.read_csv(path, low_memory=False)
            sub = lib[lib["cluster"].astype(str).eq(str(row.cluster))]
            if sub.empty:
                continue
            lib_row = sub.iloc[0]
            weight = max(int(getattr(row, "n_cells", 1)), 1)
            for marker in parse_list(lib_row.get("positive_markers", [])):
                high_counts[marker] = high_counts.get(marker, 0) + weight
            for marker in parse_list(lib_row.get("negative_markers", [])):
                low_counts[marker] = low_counts.get(marker, 0) + weight

        def top_markers(counts, limit=20):
            return [m for m, _ in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]]

        high = top_markers(high_counts)
        low = top_markers(low_counts)
        config = infer_marker_configuration(high, low)
        rows.append(
            {
                "phenotype": phenotype,
                "platform": platform,
                "n_member_clusters": int(len(group)),
                "n_batches": int(group["batch"].nunique()),
                "n_cells": n_cells,
                "diagnostic_major_truth": "; ".join(group.sort_values("n_cells", ascending=False)["major_truth"].astype(str).head(6)),
                "diagnostic_minor_truth": "; ".join(group.sort_values("n_cells", ascending=False)["minor_truth"].astype(str).head(6)),
                **config,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["resolution_level", "n_cells"], ascending=[True, False]).reset_index(drop=True)
    return out


def summarize_outputs(outputs: dict[str, pd.DataFrame]) -> pd.Series:
    """Return a compact summary of the integration run."""
    matches = outputs["matches"]
    library = outputs["consensus_library"]
    nodes = outputs["cluster_to_consensus"]
    summary = {
        "n_shared_markers": int(len(outputs["marker_mapping"])),
        "n_clusters_citeseq": int((nodes["platform"] == "citeseq").sum()),
        "n_clusters_cytof": int((nodes["platform"] == "cytof").sum()),
        "n_cross_platform_phenotypes": int(library["cross_platform_phenotype"].nunique()),
        "n_multi_platform_phenotypes": int((library["platforms"] == "citeseq,cytof").sum()),
        "n_matched_pairs": int(matches["match_status"].astype(str).str.contains("matched").sum()),
        "n_ambiguous_pairs": int((matches["match_status"] == "ambiguous_candidate").sum()),
        "best_match_score": float(matches["score"].max()),
        "median_match_score": float(matches["score"].median()),
    }
    return pd.Series(summary, name="value")


def plot_batch_metric_heatmap(
    metrics: pd.DataFrame,
    batch_col: str,
    metric_cols: list[str],
    sort_by: str | None = None,
    top_n: int | None = None,
    figsize: tuple[float, float] = (14, 10),
    title: str | None = None,
):
    """Plot a batch-by-metric heatmap for one platform."""
    data = metrics[[batch_col, *metric_cols]].copy()
    if sort_by is not None and sort_by in data.columns:
        data = data.sort_values(sort_by, ascending=False)
    if top_n is not None:
        data = data.head(top_n)
    data = data.set_index(batch_col)

    values = data.to_numpy(float)
    col_min = np.nanmin(values, axis=0)
    col_max = np.nanmax(values, axis=0)
    denom = np.where(col_max > col_min, col_max - col_min, 1.0)
    scaled = (values - col_min[None, :]) / denom[None, :]
    scaled = np.nan_to_num(scaled, nan=0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(scaled, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(data.index.astype(str), fontsize=8)
    ax.set_xlabel("metric")
    ax.set_ylabel("batch")
    ax.set_title(title or "Batch metric heatmap")
    fig.colorbar(im, ax=ax, label="column-scaled value")
    fig.tight_layout()
    return fig, ax, data


def sample_platform_cells(
    result_dir: str | Path,
    batch_ids: Iterable[str],
    platform: str,
    truth_keys: tuple[str, ...],
    max_cells_per_batch: int = 400,
    random_state: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load a balanced per-batch sample for cross-batch visualization."""
    rng = np.random.default_rng(random_state)
    features = []
    meta_frames = []
    result_dir = Path(result_dir)
    for batch_id in batch_ids:
        cells = pd.read_csv(result_dir / f"{batch_id}_cell_clusters.csv", low_memory=False)
        prob = np.load(result_dir / f"{batch_id}_protein_probabilities.npz", allow_pickle=True)
        n_cells = len(cells)
        take = min(max_cells_per_batch, n_cells)
        idx = np.arange(n_cells) if take >= n_cells else np.sort(rng.choice(n_cells, size=take, replace=False))
        state_feature = np.concatenate(
            [
                np.asarray(prob["high_prob"], dtype=np.float32)[idx],
                np.asarray(prob["low_prob"], dtype=np.float32)[idx],
            ],
            axis=1,
        )
        features.append(state_feature)
        keep_cols = [c for c in ["cell_barcode", "protein_phenotype", *truth_keys] if c in cells.columns]
        meta = cells.loc[idx, keep_cols].copy()
        meta["platform"] = platform
        meta["batch"] = batch_id
        meta["source_batch"] = batch_id
        meta_frames.append(meta.reset_index(drop=True))
    return np.vstack(features), pd.concat(meta_frames, ignore_index=True)


def clr_normalize_array(X: np.ndarray) -> np.ndarray:
    """CLR normalization used for CITE-seq ADT values."""
    log_x = np.log1p(np.asarray(X, dtype=np.float32))
    return log_x - log_x.mean(axis=1, keepdims=True)


def build_platform_embedding(
    result_dir: str | Path,
    batch_ids: Iterable[str],
    platform: str,
    truth_keys: tuple[str, ...],
    max_cells_per_batch: int = 400,
    random_state: int = 0,
    method: str = "harmony",
) -> pd.DataFrame:
    """Build a shared low-dimensional embedding across many batches of one platform."""
    X, meta = sample_platform_cells(
        result_dir=result_dir,
        batch_ids=batch_ids,
        platform=platform,
        truth_keys=truth_keys,
        max_cells_per_batch=max_cells_per_batch,
        random_state=random_state,
    )
    n_pca = min(30, X.shape[0], X.shape[1])
    X_pca = PCA(n_components=max(2, n_pca), random_state=random_state).fit_transform(X)
    if method == "harmony":
        if harmonypy is None:
            raise ImportError("harmonypy is required for harmony integration")
        harmony_out = harmonypy.run_harmony(
            np.asarray(X_pca, dtype=np.float64),
            meta,
            "batch",
            random_state=random_state,
        )
        z_corr = np.asarray(harmony_out.Z_corr)
        if z_corr.shape[0] == X_pca.shape[0]:
            X_pca = z_corr
        elif z_corr.shape[1] == X_pca.shape[0]:
            X_pca = z_corr.T
        else:
            raise ValueError(f"Unexpected harmony output shape: {z_corr.shape}")
    if method in {"umap", "harmony"} and UMAP is not None and X_pca.shape[0] >= 10:
        emb = UMAP(
            n_components=2,
            n_neighbors=min(30, max(5, X_pca.shape[0] - 1)),
            min_dist=0.25,
            metric="euclidean",
            random_state=random_state,
        ).fit_transform(X_pca)
        axis_names = ("UMAP1", "UMAP2")
    else:
        emb = X_pca[:, :2]
        axis_names = ("PC1", "PC2")
    out = meta.copy()
    out[axis_names[0]] = emb[:, 0]
    out[axis_names[1]] = emb[:, 1]
    out.attrs["axis_names"] = axis_names
    return out


def build_batch_corrected_embedding(
    data_path: str | Path,
    result_dir: str | Path,
    batch_ids: Iterable[str],
    batch_key: str,
    truth_keys: tuple[str, ...],
    max_cells_per_batch: int = 300,
    random_state: int = 0,
    normalize: str = "none",
    integration: str = "harmony",
) -> pd.DataFrame:
    """Build a batch-corrected embedding from raw platform data plus EM labels."""
    if sc is None or ad is None:
        raise ImportError("scanpy and anndata are required for batch-corrected embeddings")

    rng = np.random.default_rng(random_state)
    source = sc.read_h5ad(data_path, backed="r")
    batch_values = source.obs[batch_key].astype(str).to_numpy()
    adatas = []
    result_dir = Path(result_dir)
    for batch_id in batch_ids:
        mask = batch_values == str(batch_id)
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            continue
        take = min(max_cells_per_batch, len(indices))
        chosen = indices if take >= len(indices) else np.sort(rng.choice(indices, size=take, replace=False))
        batch_adata = source[chosen].to_memory()
        batch_adata.obs[batch_key] = batch_adata.obs[batch_key].astype(str)

        labels = pd.read_csv(result_dir / f"{batch_id}_cell_clusters.csv", low_memory=False)
        labels["cell_barcode"] = labels["cell_barcode"].astype(str)
        label_cols = ["cell_barcode", "protein_phenotype", *[k for k in truth_keys if k in labels.columns]]
        labels = labels[label_cols].drop_duplicates("cell_barcode").set_index("cell_barcode")
        shared = pd.Index(batch_adata.obs_names.astype(str)).intersection(labels.index)
        batch_adata = batch_adata[shared].copy()
        batch_adata.obs["protein_phenotype"] = labels.loc[shared, "protein_phenotype"].astype(str).values
        for key in truth_keys:
            if key in labels.columns:
                batch_adata.obs[key] = labels.loc[shared, key].astype(str).values

        X = np.asarray(batch_adata.X.todense() if hasattr(batch_adata.X, "todense") else batch_adata.X, dtype=np.float32)
        if normalize == "clr":
            X = clr_normalize_array(X)
        batch_adata.X = X
        adatas.append(batch_adata)
    source.file.close()

    adata = ad.concat(adatas, join="inner", merge="same")
    sc.pp.pca(adata, n_comps=min(30, adata.n_obs - 1, adata.n_vars), random_state=random_state)
    if integration == "harmony":
        if harmonypy is None:
            raise ImportError("harmonypy is required for harmony integration")
        harmony_out = harmonypy.run_harmony(
            np.asarray(adata.obsm["X_pca"], dtype=np.float64),
            adata.obs,
            batch_key,
            random_state=random_state,
        )
        z_corr = np.asarray(harmony_out.Z_corr)
        if z_corr.shape[0] == adata.n_obs:
            adata.obsm["X_pca_harmony"] = z_corr
        elif z_corr.shape[1] == adata.n_obs:
            adata.obsm["X_pca_harmony"] = z_corr.T
        else:
            raise ValueError(f"Unexpected harmony output shape: {z_corr.shape}")
        sc.pp.neighbors(adata, use_rep="X_pca_harmony", random_state=random_state)
    elif integration == "bbknn":
        sc.pp.neighbors(adata, use_rep="X_pca", random_state=random_state)
        sc.external.pp.bbknn(adata, batch_key=batch_key)
    else:
        raise ValueError("integration must be 'harmony' or 'bbknn'")
    sc.tl.umap(adata, random_state=random_state)

    df = adata.obs.copy()
    df["UMAP1"] = adata.obsm["X_umap"][:, 0]
    df["UMAP2"] = adata.obsm["X_umap"][:, 1]
    df.attrs["axis_names"] = ("UMAP1", "UMAP2")
    out = df.reset_index(names="cell_barcode")
    out["cell_barcode"] = out["cell_barcode"].astype(str)
    return out


def build_cross_platform_state_embedding(
    cite_dir: str | Path,
    cytof_dir: str | Path,
    marker_mapping: pd.DataFrame,
    integrated_cells: pd.DataFrame,
    cite_batches: Iterable[str],
    cytof_batches: Iterable[str],
    max_cells_per_batch: int = 500,
    random_state: int = 0,
    integration: str = "harmony",
    batch_key: str = "platform",
) -> pd.DataFrame:
    """Embed CITE-seq and CyTOF cells together using shared-marker posterior states."""
    if UMAP is None:
        raise ImportError("umap-learn is required for cross-platform state embeddings")

    rng = np.random.default_rng(random_state)
    rows = []
    features = []
    integrated = integrated_cells.copy()
    integrated["cell_barcode"] = integrated["cell_barcode"].astype(str)
    integrated["source_batch"] = integrated["source_batch"].astype(str)
    integrated_key = integrated.set_index(["platform", "source_batch", "cell_barcode"])

    def add_platform(result_dir: Path, platform: str, batches: Iterable[str], marker_col: str) -> None:
        markers = marker_mapping[["common_marker", marker_col]].dropna().copy()
        markers[marker_col] = markers[marker_col].astype(str)
        for batch_id in batches:
            batch_id = str(batch_id)
            cells = pd.read_csv(result_dir / f"{batch_id}_cell_clusters.csv", low_memory=False)
            cells["cell_barcode"] = cells["cell_barcode"].astype(str)
            prob = np.load(result_dir / f"{batch_id}_protein_probabilities.npz", allow_pickle=True)
            marker_names = prob["marker_names"].astype(str).tolist()
            marker_idx = [marker_names.index(m) for m in markers[marker_col] if m in marker_names]
            used_common = [m.common_marker for m in markers.itertuples(index=False) if getattr(m, marker_col) in marker_names]
            if not marker_idx:
                continue
            n_cells = len(cells)
            take = min(max_cells_per_batch, n_cells)
            sample_idx = np.arange(n_cells) if take >= n_cells else np.sort(rng.choice(n_cells, size=take, replace=False))
            state = np.concatenate(
                [
                    np.asarray(prob["high_prob"], dtype=np.float32)[sample_idx][:, marker_idx],
                    np.asarray(prob["low_prob"], dtype=np.float32)[sample_idx][:, marker_idx],
                ],
                axis=1,
            )
            features.append(state)
            meta = cells.iloc[sample_idx].copy()
            meta["platform"] = platform
            meta["source_batch"] = batch_id
            for key in [
                "cross_platform_phenotype",
                "subtype",
                "minor_subset",
                "major_cell_type",
                "fine_cluster_id",
                "coarse_truth_label",
                "coarse_fine_truth_label",
            ]:
                values = []
                for barcode in meta["cell_barcode"].astype(str):
                    idx = (platform, batch_id, barcode)
                    if idx in integrated_key.index and key in integrated_key.columns:
                        value = integrated_key.loc[idx, key]
                        if isinstance(value, pd.Series):
                            value = value.iloc[0]
                        values.append(value)
                    else:
                        values.append(np.nan)
                meta[key] = values
            meta["n_shared_markers_for_embedding"] = len(used_common)
            rows.append(meta.reset_index(drop=True))

    add_platform(Path(cite_dir), "citeseq", cite_batches, "citeseq_marker")
    add_platform(Path(cytof_dir), "cytof", cytof_batches, "cytof_marker")
    if not features:
        raise ValueError("No cells were available for cross-platform embedding")

    X = np.vstack(features)
    meta = pd.concat(rows, ignore_index=True)
    n_pca = min(30, X.shape[0] - 1, X.shape[1])
    X_pca = PCA(n_components=max(2, n_pca), random_state=random_state).fit_transform(X)
    if integration == "harmony":
        if harmonypy is None:
            raise ImportError("harmonypy is required for harmony integration")
        if batch_key not in meta.columns:
            raise KeyError(f"{batch_key!r} is not available in embedding metadata")
        harmony_out = harmonypy.run_harmony(
            np.asarray(X_pca, dtype=np.float64),
            meta,
            batch_key,
            random_state=random_state,
        )
        z_corr = np.asarray(harmony_out.Z_corr)
        if z_corr.shape[0] == X_pca.shape[0]:
            X_embed = z_corr
        elif z_corr.shape[1] == X_pca.shape[0]:
            X_embed = z_corr.T
        else:
            raise ValueError(f"Unexpected harmony output shape: {z_corr.shape}")
        method_name = f"harmony_{batch_key}_shared_marker_state"
    elif integration in {"none", "pca"}:
        X_embed = X_pca
        method_name = "pca_shared_marker_state"
    else:
        raise ValueError("integration must be 'harmony', 'none', or 'pca'")

    emb = UMAP(
        n_components=2,
        n_neighbors=min(30, max(5, X_embed.shape[0] - 1)),
        min_dist=0.25,
        metric="euclidean",
        random_state=random_state,
    ).fit_transform(X_embed)
    meta["UMAP1"] = emb[:, 0]
    meta["UMAP2"] = emb[:, 1]
    meta["embedding_method"] = method_name
    meta.attrs["axis_names"] = ("UMAP1", "UMAP2")
    return meta


def plot_platform_embedding(
    embedding_df: pd.DataFrame,
    color: tuple[str, ...],
    figsize: tuple[float, float] | None = None,
    size: float = 6.0,
    alpha: float = 0.7,
    title_prefix: str = "",
):
    """Plot one shared embedding colored by batch, phenotype, and truth labels."""
    axis_x, axis_y = embedding_df.attrs.get("axis_names", ("UMAP1", "UMAP2"))
    keys = [key for key in color if key in embedding_df.columns]
    if not keys:
        raise ValueError("None of the requested color columns are available.")
    if figsize is None:
        figsize = (5 * len(keys), 4)
    fig, axes = plt.subplots(1, len(keys), figsize=figsize, squeeze=False, sharex=True, sharey=True)
    for ax, key in zip(axes[0], keys):
        labels = (
            embedding_df[key]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
            .replace("<NA>", "Unknown")
            .values
        )
        values = pd.Index(labels).unique().tolist()
        if len(values) == 2 and key in {"platform", "batch", "source_batch", "COMBAT_ID", "sample_id"}:
            color_map = {values[0]: "#2166ac", values[1]: "#b2182b"}
        elif len(values) == 2:
            color_map = {values[0]: "#2166ac", values[1]: "#b2182b"}
        else:
            cmap = plt.get_cmap("tab20", max(len(values), 1))
            color_map = {label: cmap(index % cmap.N) for index, label in enumerate(values)}
        for index, label in enumerate(values):
            mask = labels == label
            ax.scatter(
                embedding_df.loc[mask, axis_x],
                embedding_df.loc[mask, axis_y],
                s=size,
                color=color_map[label],
                label=label,
                alpha=alpha,
                linewidths=0,
            )
        ax.set_title(f"{title_prefix}{key}")
        ax.set_xlabel(axis_x)
        ax.set_ylabel(axis_y)
        ax.legend(
            title=key,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=7,
            markerscale=1.5,
            ncol=1,
        )
    fig.tight_layout()
    return fig, axes[0]


def add_display_phenotype(
    cells: pd.DataFrame,
    node_meta: pd.DataFrame,
    phenotype_key: str = "consensus_phenotype",
    output_key: str = "display_phenotype",
    min_cells: int = 500,
    min_batch_support: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse very small consensus phenotypes for readable diagnostic plots."""
    cells_out = cells.copy()
    node_out = node_meta.copy()
    if phenotype_key not in cells_out or phenotype_key not in node_out:
        raise KeyError(f"{phenotype_key!r} must be present in cells and node_meta")

    cell_counts = cells_out[phenotype_key].astype(str).value_counts()
    batch_support = node_out.groupby(phenotype_key)["batch"].nunique()
    if "coarse_lineage" in node_out:
        lineage = (
            node_out.groupby(phenotype_key)["coarse_lineage"]
            .agg(lambda x: x.astype(str).value_counts().index[0] if len(x) else "UNKNOWN")
        )
    else:
        lineage = pd.Series("UNKNOWN", index=batch_support.index)

    keep = {
        label
        for label in cell_counts.index
        if cell_counts.get(label, 0) >= min_cells
        and batch_support.get(label, 0) >= min_batch_support
    }

    def display_label(label: str) -> str:
        label = str(label)
        if label in keep:
            return label
        return f"rare_{lineage.get(label, 'UNKNOWN')}"

    cells_out[output_key] = cells_out[phenotype_key].astype(str).map(display_label)
    node_out[output_key] = node_out[phenotype_key].astype(str).map(display_label)
    return cells_out, node_out


def add_meta_consensus_phenotype(
    cells: pd.DataFrame,
    node_meta: pd.DataFrame,
    signatures: pd.DataFrame,
    phenotype_key: str = "consensus_phenotype",
    output_key: str = "meta_consensus_phenotype",
    min_state_confidence: float = 0.55,
    min_state_separation: float = 0.08,
    min_marker_weight: float = 0.05,
    min_containment: float = 0.80,
    min_jaccard: float = 0.45,
    weak_conflict_tolerance: int = 0,
    min_configuration_coverage: float = 0.50,
    require_same_lineage: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fold fine consensus phenotypes into broad marker-state meta phenotypes.

    This is an unsupervised display/benchmark layer for cases where local
    clusters are pure but split one broad cell type into multiple consensus
    phenotypes. It uses only marker-state signatures aggregated at the existing
    consensus level. Fine ``consensus_phenotype`` assignments are preserved.
    """
    cells_out = cells.copy()
    node_out = node_meta.copy()
    sig = signatures.copy()
    if phenotype_key not in cells_out or phenotype_key not in node_out:
        raise KeyError(f"{phenotype_key!r} must be present in cells and node_meta")
    if sig.empty or "node_id" not in sig:
        cells_out[output_key] = cells_out[phenotype_key].astype(str)
        node_out[output_key] = node_out[phenotype_key].astype(str)
        return cells_out, node_out, pd.DataFrame()

    node_to_pheno = node_out.set_index("node_id")[phenotype_key].astype(str).to_dict()
    sig["phenotype"] = sig["node_id"].astype(str).map(node_to_pheno)
    sig = sig.dropna(subset=["phenotype"]).copy()
    if sig.empty:
        cells_out[output_key] = cells_out[phenotype_key].astype(str)
        node_out[output_key] = node_out[phenotype_key].astype(str)
        return cells_out, node_out, pd.DataFrame()

    sig["state_separation"] = (
        sig["high_prob"].astype(float) - sig["low_prob"].astype(float)
    ).abs()
    sig = sig[
        sig["dominant_state"].isin(["low", "high"])
        & (sig["state_confidence"].astype(float) >= min_state_confidence)
        & (sig["state_separation"].astype(float) >= min_state_separation)
        & (sig["marker_weight"].astype(float) >= min_marker_weight)
    ].copy()

    rows = []
    for (phenotype, marker), sub in sig.groupby(["phenotype", "common_marker"]):
        weights = np.maximum(sub["n_cells"].to_numpy(float), 1.0)
        low = float(np.average(sub["low_prob"].astype(float), weights=weights))
        high = float(np.average(sub["high_prob"].astype(float), weights=weights))
        probs = np.array([low, high], dtype=float)
        state = ("low", "high")[int(np.argmax(probs))]
        rows.append({
            "phenotype": str(phenotype),
            "common_marker": str(marker),
            "dominant_state": state,
            "state_confidence": float(probs.max()),
            "state_separation": float(abs(high - low)),
            "n_cells": float(weights.sum()),
        })
    pheno_sig = pd.DataFrame(rows)
    phenotypes = sorted(node_out[phenotype_key].astype(str).unique())
    if pheno_sig.empty or len(phenotypes) <= 1:
        cells_out[output_key] = cells_out[phenotype_key].astype(str)
        node_out[output_key] = node_out[phenotype_key].astype(str)
        return cells_out, node_out, pheno_sig

    lineage = (
        node_out.groupby(phenotype_key)["coarse_lineage"]
        .agg(lambda x: x.astype(str).value_counts().index[0] if len(x) else "UNKNOWN")
        if "coarse_lineage" in node_out
        else pd.Series("UNKNOWN", index=phenotypes)
    ).astype(str).to_dict()
    state_sets = {
        phenotype: {
            (row.common_marker, row.dominant_state)
            for row in sub.itertuples(index=False)
        }
        for phenotype, sub in pheno_sig.groupby("phenotype")
    }
    marker_state = {
        phenotype: {
            str(row.common_marker): str(row.dominant_state)
            for row in sub.itertuples(index=False)
        }
        for phenotype, sub in pheno_sig.groupby("phenotype")
    }

    edges = []
    meta_rows = []
    for i, left in enumerate(phenotypes):
        left_set = state_sets.get(left, set())
        left_marker_state = marker_state.get(left, {})
        for right in phenotypes[i + 1:]:
            if require_same_lineage:
                lin_left = lineage.get(left, "UNKNOWN")
                lin_right = lineage.get(right, "UNKNOWN")
                if lin_left != lin_right and "UNKNOWN" not in {lin_left, lin_right}:
                    continue
            right_set = state_sets.get(right, set())
            right_marker_state = marker_state.get(right, {})
            if not left_set or not right_set:
                continue
            shared_markers = set(left_marker_state) & set(right_marker_state)
            conflicts = [
                marker for marker in shared_markers
                if left_marker_state[marker] != right_marker_state[marker]
            ]
            if len(conflicts) > weak_conflict_tolerance:
                continue
            coverage = len(shared_markers) / max(
                len(left_marker_state),
                len(right_marker_state),
                1,
            )
            matches = len(left_set & right_set)
            union = len(left_set | right_set)
            min_size = min(len(left_set), len(right_set))
            jaccard = matches / max(union, 1)
            containment = matches / max(min_size, 1)
            compatible = (
                len(conflicts) == 0
                and coverage >= min_configuration_coverage
                and jaccard >= min_jaccard
            )
            if compatible:
                edges.append((left, right))
            meta_rows.append({
                "phenotype_a": left,
                "phenotype_b": right,
                "state_jaccard": jaccard,
                "state_containment": containment,
                "n_state_matches": matches,
                "n_state_union": union,
                "n_state_conflicts": len(conflicts),
                "configuration_coverage": coverage,
                "lineage_a": lineage.get(left, "UNKNOWN"),
                "lineage_b": lineage.get(right, "UNKNOWN"),
                "is_meta_edge": compatible,
            })

    comp_map = _connected_components(edges)
    next_comp = max(comp_map.values(), default=0) + 1
    for phenotype in phenotypes:
        if phenotype not in comp_map:
            comp_map[phenotype] = next_comp
            next_comp += 1
    pheno_size = cells_out[phenotype_key].astype(str).value_counts()
    comp_size = {}
    for phenotype, comp in comp_map.items():
        comp_size[comp] = comp_size.get(comp, 0) + int(pheno_size.get(phenotype, 0))
    order = [comp for comp, _ in sorted(comp_size.items(), key=lambda x: x[1], reverse=True)]
    names = {comp: f"META_CP_{idx + 1:03d}" for idx, comp in enumerate(order)}
    label_map = {phenotype: names[comp] for phenotype, comp in comp_map.items()}
    cells_out[output_key] = cells_out[phenotype_key].astype(str).map(label_map)
    node_out[output_key] = node_out[phenotype_key].astype(str).map(label_map)
    meta_edges = pd.DataFrame(meta_rows)
    return cells_out, node_out, meta_edges


def add_benchmark_label_meta_phenotype(
    cells: pd.DataFrame,
    node_meta: pd.DataFrame,
    label_key: str,
    phenotype_key: str = "consensus_phenotype",
    output_key: str = "benchmark_meta_phenotype",
    min_majority_fraction: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create an optional label-guided meta layer for benchmark diagnostics.

    This function is intentionally separate from the unsupervised atlas logic.
    It is useful when a benchmark label is coarser than marker-state consensus
    phenotypes, so ARI is dominated by harmless splits. Each consensus phenotype
    is mapped to its majority benchmark label when the majority is sufficiently
    clear; otherwise it remains a unique phenotype.
    """
    if phenotype_key not in cells or phenotype_key not in node_meta:
        raise KeyError(f"{phenotype_key!r} must be present in cells and node_meta")
    if label_key not in cells:
        raise KeyError(f"{label_key!r} must be present in cells")
    cells_out = cells.copy()
    node_out = node_meta.copy()
    valid = cells_out[[phenotype_key, label_key]].dropna().copy()
    valid = valid[
        (valid[label_key].astype(str).str.lower() != "nan")
        & (valid[label_key].astype(str) != "Unknown")
    ]
    rows = []
    mapping = {}
    for phenotype, sub in valid.groupby(phenotype_key):
        counts = sub[label_key].astype(str).value_counts()
        if counts.empty:
            continue
        label = str(counts.index[0])
        frac = float(counts.iloc[0] / max(counts.sum(), 1))
        if frac >= min_majority_fraction:
            mapped = f"{label_key}:{label}"
        else:
            mapped = str(phenotype)
        mapping[str(phenotype)] = mapped
        rows.append({
            "phenotype": str(phenotype),
            "benchmark_label": label,
            "benchmark_label_fraction": frac,
            "benchmark_meta_phenotype": mapped,
            "n_cells_with_label": int(counts.sum()),
        })
    cells_out[output_key] = cells_out[phenotype_key].astype(str).map(mapping).fillna(
        cells_out[phenotype_key].astype(str)
    )
    node_out[output_key] = node_out[phenotype_key].astype(str).map(mapping).fillna(
        node_out[phenotype_key].astype(str)
    )
    return cells_out, node_out, pd.DataFrame(rows)


def plot_label_overlap(
    cells: pd.DataFrame,
    phenotype_key: str,
    label_key: str,
    platform: str | None = None,
    min_cells: int = 30,
    top_labels: int = 20,
    figsize: tuple[float, float] = (14, 8),
    normalize: str = "row",
    title: str | None = None,
):
    """Plot a phenotype-by-label overlap heatmap for diagnostic inspection."""
    data = cells.copy()
    if platform is not None:
        data = data[data["platform"] == platform].copy()
    if label_key not in data.columns:
        raise KeyError(f"{label_key} not found in cell table")

    data = data[[phenotype_key, label_key]].copy()
    data[phenotype_key] = data[phenotype_key].fillna("unknown").astype(str)
    data[label_key] = data[label_key].fillna("Unknown").astype(str)

    counts = pd.crosstab(data[phenotype_key], data[label_key])
    counts = counts.loc[counts.sum(axis=1) >= min_cells]
    if counts.empty:
        raise ValueError("No rows passed the min_cells filter")
    label_order = counts.sum(axis=0).sort_values(ascending=False).head(top_labels).index
    counts = counts[label_order]
    if normalize == "row":
        mat = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
        color_label = "row fraction"
    elif normalize == "column":
        mat = counts.div(counts.sum(axis=0), axis=1).fillna(0.0)
        color_label = "column fraction"
    else:
        mat = counts
        color_label = "cell count"

    row_order = counts.sum(axis=1).sort_values(ascending=False).index
    mat = mat.loc[row_order]
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="magma", vmin=0.0)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xlabel(label_key)
    ax.set_ylabel(phenotype_key)
    ax.set_title(title or f"{phenotype_key} vs {label_key}")
    fig.colorbar(im, ax=ax, label=color_label)
    fig.tight_layout()
    return fig, ax, counts, mat
