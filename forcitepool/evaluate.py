#!/usr/bin/env python3
"""Ground-truth evaluation and figures for the protein taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "outputs" / "run"
DEFAULT_TRUTH_KEYS = ("cell_type", "major_subset", "subtype", "minor_subset")
UNKNOWN_VALUES = {"", "nan", "NaN", "None", "Unknown", "unknown"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forcitepool evaluate",
        description="Evaluate a completed protein taxonomy against annotations."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--input-h5ad", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--truth-keys",
        nargs="+",
        default=list(DEFAULT_TRUTH_KEYS),
        help="Annotation columns to evaluate when present.",
    )
    parser.add_argument(
        "--tree-truth-key",
        default=None,
        help="Annotation used only to color the post-hoc tree plot.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_input_h5ad(
    run_dir: Path,
    input_h5ad: Path | None,
) -> Path:
    """Resolve evaluation input from an explicit path or the run manifest."""

    if input_h5ad is not None:
        path = Path(input_h5ad).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input h5ad does not exist: {path}")
        return path

    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            "cannot infer input h5ad: missing run_config.json; "
            "pass --input-h5ad explicitly"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidates = []
    materialized = config.get("materialized_input")
    if materialized:
        candidates.append(Path(materialized))
        candidates.append(run_dir / "cache" / Path(materialized).name)
    original = config.get("input")
    if original and not str(original).endswith(".gz"):
        candidates.append(Path(original))
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if path.exists():
            return path
    raise FileNotFoundError(
        "the h5ad used by this run is unavailable; pass --input-h5ad explicitly"
    )


def safe_filename_token(value: str) -> str:
    """Return a deterministic, path-safe filename component."""

    value = str(value)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")[:80]
    slug = slug or "annotation"
    if slug != value:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug}_{digest}"
    return slug


def purity_score(truth: pd.Series, prediction: pd.Series) -> float:
    table = pd.crosstab(prediction, truth)
    return float(table.max(axis=1).sum() / table.to_numpy().sum())


def metric_record(
    truth: pd.Series,
    prediction: pd.Series,
    *,
    truth_key: str,
    stage: str,
    stage_kind: str,
    round_id: int | None,
) -> dict:
    truth = truth.astype(str)
    prediction = prediction.astype(str)
    keep = ~truth.isin(UNKNOWN_VALUES) & ~prediction.isin(UNKNOWN_VALUES)
    y_true = truth[keep]
    y_pred = prediction[keep]
    return {
        "truth_key": truth_key,
        "stage": stage,
        "stage_kind": stage_kind,
        "round": round_id,
        "n_cells": int(keep.sum()),
        "n_truth_classes": int(y_true.nunique()),
        "n_predicted_groups": int(y_pred.nunique()),
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "purity": purity_score(y_true, y_pred),
        "homogeneity": homogeneity_score(y_true, y_pred),
        "completeness": completeness_score(y_true, y_pred),
        "v_measure": v_measure_score(y_true, y_pred),
    }


def attach_ground_truth(
    cells: pd.DataFrame,
    input_h5ad: Path,
    truth_keys: Sequence[str] = DEFAULT_TRUTH_KEYS,
) -> pd.DataFrame:
    """Attach all available annotations without using them in construction."""

    output = cells.copy()
    missing = [key for key in truth_keys if key not in output]
    if not missing:
        return output
    adata = ad.read_h5ad(input_h5ad, backed="r")
    available = [key for key in missing if key in adata.obs]
    truth = adata.obs[available].copy()
    truth.index = truth.index.astype(str)
    truth = truth.reset_index(names="cell_barcode")
    for key in available:
        truth[key] = truth[key].astype(str)
    output["cell_barcode"] = output["cell_barcode"].astype(str)
    output = output.merge(truth, on="cell_barcode", how="left", validate="one_to_one")
    adata.file.close()
    return output


def taxonomy_cut_labels(
    cells: pd.DataFrame,
    merges: pd.DataFrame,
) -> dict[int, pd.Series]:
    """Return cell labels after every parallel bottom-up merge round."""

    leaf_to_group = {
        leaf: leaf for leaf in sorted(cells["aligned_leaf_id"].astype(str).unique())
    }
    cuts = {
        0: cells["aligned_leaf_id"].astype(str).copy()
    }
    if merges.empty or "round" not in merges:
        return cuts
    for round_id in sorted(pd.to_numeric(merges["round"]).astype(int).unique()):
        round_merges = merges[pd.to_numeric(merges["round"]).astype(int) == round_id]
        for row in round_merges.itertuples(index=False):
            children = {str(row.child_a), str(row.child_b)}
            for leaf, current in list(leaf_to_group.items()):
                if current in children:
                    leaf_to_group[leaf] = str(row.parent)
        cuts[int(round_id)] = cells["aligned_leaf_id"].astype(str).map(leaf_to_group)
    return cuts


def evaluate_levels(
    cells: pd.DataFrame,
    merges: pd.DataFrame,
    truth_keys: Sequence[str] = DEFAULT_TRUTH_KEYS,
) -> tuple[pd.DataFrame, dict[int, pd.Series]]:
    cuts = taxonomy_cut_labels(cells, merges)
    predictions: list[tuple[str, str, int | None, pd.Series]] = [
        (
            "batch_local_cluster",
            "local",
            None,
            cells["batch"].astype(str) + "|" + cells["protein_phenotype"].astype(str),
        ),
        ("aligned_leaf", "aligned", 0, cuts[0]),
    ]
    predictions.extend(
        (
            f"taxonomy_round_{round_id}",
            "taxonomy",
            round_id,
            labels,
        )
        for round_id, labels in cuts.items()
        if round_id > 0
    )
    records = []
    for truth_key in truth_keys:
        if truth_key not in cells:
            continue
        valid = ~cells[truth_key].astype(str).isin(UNKNOWN_VALUES)
        if not valid.any():
            continue
        for stage, kind, round_id, prediction in predictions:
            records.append(
                metric_record(
                    cells[truth_key],
                    prediction,
                    truth_key=truth_key,
                    stage=stage,
                    stage_kind=kind,
                    round_id=round_id,
                )
            )
    return pd.DataFrame(records), cuts


def majority_label(values: pd.Series) -> str:
    values = values.astype(str)
    values = values[~values.isin(UNKNOWN_VALUES)]
    return str(values.value_counts().index[0]) if len(values) else "Unknown"


def alignment_truth_audit(
    cells: pd.DataFrame,
    membership: pd.DataFrame,
    truth_keys: Sequence[str] = DEFAULT_TRUTH_KEYS,
) -> pd.DataFrame:
    local_truth_rows = []
    for local_node, sub in cells.groupby("local_node_id", sort=True):
        row = {
            "local_node_id": str(local_node),
            "n_cells_observed": len(sub),
        }
        for truth_key in truth_keys:
            if truth_key in sub:
                counts = sub[truth_key].astype(str)
                valid = counts[~counts.isin(UNKNOWN_VALUES)]
                row[f"{truth_key}_majority"] = majority_label(counts)
                row[f"{truth_key}_purity"] = (
                    float(valid.value_counts().iloc[0] / len(valid))
                    if len(valid)
                    else np.nan
                )
        local_truth_rows.append(row)
    local_truth = pd.DataFrame(local_truth_rows)
    output = membership.merge(local_truth, on="local_node_id", how="left")
    for truth_key in truth_keys:
        column = f"{truth_key}_majority"
        if column not in output:
            continue
        output[f"{truth_key}_group_concordant"] = output.groupby(
            "aligned_node_id"
        )[column].transform(
            lambda x: len(set(x.astype(str)) - UNKNOWN_VALUES) <= 1
        )
    return output


def best_cuts(metrics: pd.DataFrame) -> pd.DataFrame:
    taxonomy = metrics[metrics["stage_kind"].isin(["aligned", "taxonomy"])].copy()
    rows = []
    for truth_key, sub in taxonomy.groupby("truth_key", sort=False):
        for metric in ["ari", "nmi", "purity", "v_measure"]:
            best = sub.sort_values(
                [metric, "n_predicted_groups"],
                ascending=[False, True],
            ).iloc[0]
            rows.append(
                {
                    "truth_key": truth_key,
                    "metric": metric,
                    "best_stage": best["stage"],
                    "round": best["round"],
                    "n_predicted_groups": int(best["n_predicted_groups"]),
                    "value": float(best[metric]),
                }
            )
    return pd.DataFrame(rows)


def _plot_grid(n_panels: int, *, width: float, height: float):
    n_columns = min(2, max(n_panels, 1))
    n_rows = int(np.ceil(max(n_panels, 1) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(width * n_columns, height * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    return fig, axes.flat


def plot_metric_curves(metrics: pd.DataFrame, output_path: Path) -> None:
    truth_keys = list(metrics["truth_key"].unique())
    fig, axes = _plot_grid(len(truth_keys), width=7.5, height=5.5)
    axes = list(axes)
    for ax, truth_key in zip(axes, truth_keys):
        sub = metrics[metrics["truth_key"] == truth_key]
        hierarchy = sub[sub["stage_kind"].isin(["aligned", "taxonomy"])].sort_values(
            "n_predicted_groups"
        )
        for metric, color, marker in [
            ("ari", "#1f77b4", "o"),
            ("nmi", "#ff7f0e", "s"),
            ("purity", "#2ca02c", "^"),
            ("v_measure", "#9467bd", "D"),
        ]:
            ax.plot(
                hierarchy["n_predicted_groups"],
                hierarchy[metric],
                marker=marker,
                markersize=5,
                linewidth=1.8,
                color=color,
                label=metric.upper(),
            )
            local = sub[sub["stage_kind"] == "local"]
            if not local.empty:
                ax.scatter(
                    local["n_predicted_groups"],
                    local[metric],
                    marker="x",
                    s=55,
                    color=color,
                )
        ax.set_title(truth_key)
        ax.set_xlabel("Number of predicted groups")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=9)
    for ax in axes[len(truth_keys):]:
        ax.set_visible(False)
    fig.suptitle(
        "Ground-truth agreement across taxonomy cuts\n"
        "(x markers show batch-local clustering)",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_confusions(
    cells: pd.DataFrame,
    metrics: pd.DataFrame,
    cuts: dict[int, pd.Series],
    output_path: Path,
) -> None:
    truth_keys = list(metrics["truth_key"].unique())
    fig, axes = _plot_grid(len(truth_keys), width=9.5, height=7.5)
    axes = list(axes)
    for ax, truth_key in zip(axes, truth_keys):
        sub = metrics[
            (metrics["truth_key"] == truth_key)
            & metrics["stage_kind"].isin(["aligned", "taxonomy"])
        ]
        best = sub.sort_values(["ari", "n_predicted_groups"], ascending=[False, True]).iloc[0]
        round_id = int(best["round"])
        prediction = cuts[round_id]
        truth = cells[truth_key].astype(str)
        keep = ~truth.isin(UNKNOWN_VALUES)
        table = pd.crosstab(prediction[keep], truth[keep])
        row_order = (
            table.div(table.sum(axis=1), axis=0).max(axis=1).sort_values(ascending=False).index
        )
        table = table.loc[row_order]
        normalized = table.div(table.sum(axis=1), axis=0)
        sns.heatmap(
            normalized,
            ax=ax,
            cmap="mako",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Fraction within predicted group"},
            xticklabels=True,
            yticklabels=table.shape[0] <= 30,
        )
        ax.set_title(
            f"{truth_key}: best ARI cut, round {round_id}, "
            f"K={int(best['n_predicted_groups'])}, ARI={best['ari']:.3f}"
        )
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Taxonomy group")
        ax.tick_params(axis="x", rotation=75, labelsize=8)
    for ax in axes[len(truth_keys):]:
        ax.set_visible(False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_leaf_heatmap(
    cells: pd.DataFrame,
    output_path: Path,
    *,
    truth_key: str,
) -> None:
    truth = cells[truth_key].astype(str)
    keep = ~truth.isin(UNKNOWN_VALUES)
    table = pd.crosstab(cells.loc[keep, "aligned_leaf_id"], truth[keep])
    normalized = table.div(table.sum(axis=1), axis=0)
    order = pd.DataFrame(
        {
            "major": normalized.idxmax(axis=1),
            "purity": normalized.max(axis=1),
        }
    ).sort_values(["major", "purity"], ascending=[True, False]).index
    normalized = normalized.loc[order]
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(
        normalized,
        cmap="viridis",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Fraction within aligned leaf"},
    )
    ax.set_title(f"Aligned protein leaves versus {truth_key}")
    ax.set_xlabel(f"Ground truth {truth_key}")
    ax.set_ylabel("Aligned leaf")
    ax.tick_params(axis="x", rotation=65)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _tree_positions(edges: pd.DataFrame, roots: list[str]) -> tuple[dict, dict, dict]:
    children: dict[str, list[str]] = {}
    for row in edges.itertuples(index=False):
        children.setdefault(str(row.parent), []).append(str(row.child))
    for node in children:
        children[node] = sorted(children[node])
    x_pos: dict[str, float] = {}
    depth: dict[str, int] = {}
    next_leaf = [0]

    def visit(node: str, level: int) -> float:
        depth[node] = level
        node_children = children.get(node, [])
        if not node_children:
            x_pos[node] = float(next_leaf[0])
            next_leaf[0] += 1
        else:
            child_x = [visit(child, level + 1) for child in node_children]
            x_pos[node] = float(np.mean(child_x))
        return x_pos[node]

    for root in roots:
        visit(root, 0)
        next_leaf[0] += 1
    return x_pos, depth, children


def plot_truth_colored_tree(
    cells: pd.DataFrame,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_path: Path,
    *,
    truth_key: str = "subtype",
) -> None:
    if truth_key not in cells:
        raise KeyError(f"missing tree coloring annotation: {truth_key}")
    roots = nodes.loc[nodes["is_root"].astype(bool), "tree_node_id"].astype(str).tolist()
    x_pos, depth, children = _tree_positions(edges, roots)
    leaf_truth = (
        cells.groupby("aligned_leaf_id")[truth_key].agg(majority_label).to_dict()
    )
    node_truth = {}
    for row in nodes.itertuples(index=False):
        leaves = str(row.aligned_leaf_descendants).split(";")
        values = []
        for leaf in leaves:
            values.extend(
                cells.loc[
                    cells["aligned_leaf_id"].astype(str) == leaf,
                    truth_key,
                ].astype(str).tolist()
            )
        node_truth[str(row.tree_node_id)] = majority_label(pd.Series(values, dtype=str))

    categories = sorted(set(node_truth.values()) - UNKNOWN_VALUES)
    palette = sns.color_palette("tab20", n_colors=max(len(categories), 1))
    colors = dict(zip(categories, palette))
    colors["Unknown"] = (0.75, 0.75, 0.75)
    max_depth = max(depth.values(), default=0)
    fig, ax = plt.subplots(figsize=(25, 17))
    for row in edges.itertuples(index=False):
        parent, child = str(row.parent), str(row.child)
        ax.plot(
            [x_pos[parent], x_pos[child]],
            [max_depth - depth[parent], max_depth - depth[child]],
            color="#777777",
            linewidth=2.4,
            linestyle="-",
            zorder=1,
        )
    for node, x in x_pos.items():
        truth = node_truth.get(node, "Unknown")
        is_leaf = node not in children
        ax.scatter(
            x,
            max_depth - depth[node],
            s=320 if is_leaf else 180,
            color=colors.get(truth, colors["Unknown"]),
            edgecolor="black",
            linewidth=1.1 if is_leaf else 0.7,
            alpha=1.0,
            zorder=2,
        )
        if is_leaf:
            ax.text(
                x,
                max_depth - depth[node] - 0.28,
                f"{node.replace('ALIGNED_', 'A')}\n{leaf_truth.get(node, 'Unknown')}",
                ha="right",
                va="top",
                rotation=58,
                fontsize=10,
                linespacing=1.15,
            )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            label=category,
            markerfacecolor=colors[category],
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=13,
        )
        for category in categories
    ]
    ax.legend(
        handles=legend,
        title=f"Majority {truth_key}",
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=10,
        title_fontsize=11,
        ncol=2 if len(categories) > 12 else 1,
    )
    subtitle = "(ground truth was not used to construct the tree)"
    ax.set_title(
        f"Protein taxonomy colored post hoc by majority {truth_key}\n{subtitle}",
        fontsize=19,
    )
    ax.set_xlabel("Aligned leaves")
    ax.set_ylabel("Tree depth", fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    metrics: pd.DataFrame,
    best: pd.DataFrame,
    alignment_audit: pd.DataFrame,
    output_path: Path,
    *,
    truth_keys: Sequence[str],
    tree_truth_key: str,
    tree_truth_file_token: str,
) -> None:
    aligned = metrics[metrics["stage"] == "aligned_leaf"].set_index("truth_key")
    best_ari = best[best["metric"] == "ari"].set_index("truth_key")
    multi = alignment_audit.groupby("aligned_node_id")["batch"].nunique()
    multi_ids = set(multi[multi > 1].index)
    multi_rows = alignment_audit[
        alignment_audit["aligned_node_id"].isin(multi_ids)
    ]
    concordance = {}
    for truth_key in truth_keys:
        col = f"{truth_key}_group_concordant"
        if col in multi_rows:
            concordance[truth_key] = float(
                multi_rows.groupby("aligned_node_id")[col].first().mean()
            )
    summary = {
        "truth_keys": list(truth_keys),
        "tree_truth_key": tree_truth_key,
        "tree_truth_file_token": tree_truth_file_token,
        "n_cells": int(aligned["n_cells"].max()),
        "n_aligned_leaves": int(aligned["n_predicted_groups"].max()),
        "n_multi_batch_aligned_leaves": len(multi_ids),
        "aligned_leaf_metrics": {
            truth: {
                key: float(aligned.loc[truth, key])
                for key in ["ari", "nmi", "purity", "v_measure"]
            }
            for truth in aligned.index
        },
        "best_ari_taxonomy_cut": {
            truth: {
                "stage": str(best_ari.loc[truth, "best_stage"]),
                "round": int(best_ari.loc[truth, "round"]),
                "n_predicted_groups": int(best_ari.loc[truth, "n_predicted_groups"]),
                "ari": float(best_ari.loc[truth, "value"]),
            }
            for truth in best_ari.index
        },
        "multi_batch_alignment_majority_concordance": concordance,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run_evaluation(
    run_dir: Path = DEFAULT_RUN,
    input_h5ad: Path | None = None,
    output_dir: Path | None = None,
    truth_keys: Sequence[str] = DEFAULT_TRUTH_KEYS,
    tree_truth_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    run_dir = Path(run_dir).expanduser().resolve()
    input_h5ad = resolve_input_h5ad(run_dir, input_h5ad)
    tables = run_dir / "tables"
    output_dir = Path(output_dir) if output_dir else run_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(tables / "cell_taxonomy_assignments.csv", low_memory=False)
    truth_keys = tuple(dict.fromkeys(str(key) for key in truth_keys))
    if not truth_keys:
        raise ValueError("at least one truth key is required for evaluation")
    cells = attach_ground_truth(cells, Path(input_h5ad), truth_keys)
    available_truth_keys = [
        key
        for key in truth_keys
        if key in cells
        and (~cells[key].astype(str).isin(UNKNOWN_VALUES)).any()
    ]
    if not available_truth_keys:
        raise ValueError(
            "none of the requested ground-truth columns are available: "
            f"{list(truth_keys)}"
        )
    if tree_truth_key is None:
        tree_truth_key = (
            "subtype"
            if "subtype" in available_truth_keys
            else available_truth_keys[-1]
        )
    if tree_truth_key not in available_truth_keys:
        raise ValueError(
            f"tree truth key {tree_truth_key!r} is not an available evaluated column"
        )
    tree_truth_file_token = safe_filename_token(tree_truth_key)
    membership = pd.read_csv(tables / "alignment_membership.csv")
    nodes = pd.read_csv(tables / "taxonomy_nodes.csv")
    edges = pd.read_csv(tables / "taxonomy_edges.csv")
    merges = pd.read_csv(tables / "taxonomy_merges.csv")

    metrics, cuts = evaluate_levels(cells, merges, available_truth_keys)
    best = best_cuts(metrics)
    alignment_audit = alignment_truth_audit(
        cells,
        membership,
        available_truth_keys,
    )
    metrics.to_csv(output_dir / "groundtruth_metrics_by_level.csv", index=False)
    best.to_csv(output_dir / "best_taxonomy_cuts.csv", index=False)
    alignment_audit.to_csv(
        output_dir / "alignment_groundtruth_audit.csv",
        index=False,
    )
    cells.to_csv(output_dir / "cells_with_all_groundtruth.csv", index=False)
    plot_metric_curves(metrics, output_dir / "metrics_across_taxonomy.png")
    plot_best_confusions(
        cells,
        metrics,
        cuts,
        output_dir / "best_cut_confusion_heatmaps.png",
    )
    plot_aligned_leaf_heatmap(
        cells,
        output_dir / f"aligned_leaf_{tree_truth_file_token}_heatmap.png",
        truth_key=tree_truth_key,
    )
    plot_truth_colored_tree(
        cells,
        nodes,
        edges,
        output_dir / f"taxonomy_{tree_truth_file_token}_tree.png",
        truth_key=tree_truth_key,
    )
    write_summary(
        metrics,
        best,
        alignment_audit,
        output_dir / "evaluation_summary.json",
        truth_keys=available_truth_keys,
        tree_truth_key=tree_truth_key,
        tree_truth_file_token=tree_truth_file_token,
    )
    return {
        "cells": cells,
        "metrics": metrics,
        "best_cuts": best,
        "alignment_audit": alignment_audit,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_evaluation(
        args.run_dir,
        args.input_h5ad,
        args.output,
        args.truth_keys,
        args.tree_truth_key,
    )
    print(result["best_cuts"].to_string(index=False))
    output = args.output or args.run_dir / "evaluation"
    print(f"\nEvaluation written to: {output}")


if __name__ == "__main__":
    main()
