#!/usr/bin/env python3
"""Bootstrap support for the standalone protein phenotype taxonomy."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forcitepool.tree import (
    StateThresholds,
    align_clusters,
    build_cluster_marker_states,
    build_taxonomy,
)


DEFAULT_RUN = HERE / "outputs" / "run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forcitepool bootstrap",
        description="Estimate exact-clade bootstrap support without refitting clusters."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--support-threshold", type=float, default=0.80)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def thresholds_from_config(config: dict) -> StateThresholds:
    names = {field.name for field in fields(StateThresholds)}
    values = {
        name: config[name]
        for name in names
        if name in config
    }
    return StateThresholds(**values)


def resampled_states(
    model_dir: Path,
    batches: list[str],
    batch_file_ids: dict[str, str],
    thresholds: StateThresholds,
    rng: np.random.Generator,
) -> pd.DataFrame:
    frames = []
    for batch in batches:
        batch_file_id = batch_file_ids[batch]
        cells = pd.read_csv(
            model_dir / f"{batch_file_id}_cell_clusters.csv",
            low_memory=False,
        )
        params = pd.read_csv(
            model_dir / f"{batch_file_id}_marker_params.csv",
            low_memory=False,
        )
        probabilities = np.load(
            model_dir / f"{batch_file_id}_protein_probabilities.npz",
            allow_pickle=True,
        )
        labels = cells["protein_phenotype"].astype(str).to_numpy()
        low = probabilities["low_prob"]
        high = probabilities["high_prob"]
        if len(labels) != len(low) or low.shape != high.shape:
            raise ValueError(f"inconsistent cached model arrays for {batch}")

        sampled_parts = []
        for cluster in sorted(np.unique(labels)):
            indices = np.flatnonzero(labels == cluster)
            sampled_parts.append(rng.choice(indices, size=len(indices), replace=True))
        sampled = np.concatenate(sampled_parts)
        frames.append(
            build_cluster_marker_states(
                batch_id=batch,
                cluster_labels=labels[sampled],
                marker_names=probabilities["marker_names"].astype(str),
                low_prob=low[sampled],
                high_prob=high[sampled],
                marker_params=params,
                thresholds=thresholds,
            )
        )
    return pd.concat(frames, ignore_index=True)


def node_clades(nodes: pd.DataFrame) -> set[frozenset[str]]:
    return {
        frozenset(str(value).split(";"))
        for value in nodes["local_cluster_descendants"]
        if str(value)
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.replicates < 1:
        raise ValueError("--replicates must be positive")
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    thresholds = thresholds_from_config(config)
    batches = [str(batch) for batch in config["batches"]]
    batch_file_ids = {
        str(batch): str(file_id)
        for batch, file_id in config.get(
            "batch_file_ids",
            {batch: batch for batch in batches},
        ).items()
    }
    model_dir = run_dir / "batch_models"
    tables = run_dir / "tables"
    baseline_nodes = pd.read_csv(tables / "taxonomy_nodes.csv")
    baseline_edges = pd.read_csv(tables / "taxonomy_edges.csv")
    baseline_merges = pd.read_csv(tables / "taxonomy_merges.csv")
    baseline_clades = [
        frozenset(str(value).split(";"))
        for value in baseline_nodes["local_cluster_descendants"]
    ]
    support_counts = {clade: 0 for clade in baseline_clades}
    replicate_rows = []
    root_counts = []
    leaf_counts = []

    seed_sequence = np.random.SeedSequence(args.seed)
    for replicate, child_seed in enumerate(
        seed_sequence.spawn(args.replicates),
        start=1,
    ):
        states = resampled_states(
            model_dir,
            batches,
            batch_file_ids,
            thresholds,
            np.random.default_rng(child_seed),
        )
        alignment = align_clusters(
            states,
            min_shared_observed=int(config["min_shared_observed"]),
            use_weighted_scores=not bool(config.get("unweighted_scores", False)),
            minimum_state_specificity=float(
                config.get("minimum_state_specificity", 0.10)
            ),
            score_absolute_tolerance=float(
                config.get("score_absolute_tolerance", 0.05)
            ),
            score_relative_tolerance=float(
                config.get("score_relative_tolerance", 0.02)
            ),
            min_alignment_agreement=float(
                config.get("min_alignment_agreement", 0.80)
            ),
            min_prototype_batches=int(
                config.get("min_prototype_batches", 2)
            ),
            max_candidate_neighbors=config.get("max_alignment_neighbors"),
        )
        taxonomy = build_taxonomy(
            alignment["membership"],
            alignment["states"],
            min_tree_shared_observed=int(config["min_tree_shared_observed"]),
            min_tree_weighted_shared=float(
                config.get("min_tree_weighted_shared", 0.50)
            ),
            max_tree_conflicts=config.get("max_tree_conflicts"),
            minimum_state_specificity=float(
                config.get("minimum_state_specificity", 0.10)
            ),
            score_absolute_tolerance=float(
                config.get("score_absolute_tolerance", 0.05)
            ),
            score_relative_tolerance=float(
                config.get("score_relative_tolerance", 0.02)
            ),
            use_weighted_scores=not bool(config.get("unweighted_scores", False)),
        )
        observed = node_clades(taxonomy["nodes"])
        for clade in support_counts:
            support_counts[clade] += int(clade in observed)
        n_leaves = alignment["membership"]["aligned_node_id"].nunique()
        n_roots = len(taxonomy["forest"]["roots"])
        leaf_counts.append(n_leaves)
        root_counts.append(n_roots)
        replicate_rows.append(
            {
                "replicate": replicate,
                "n_aligned_leaves": n_leaves,
                "n_roots": n_roots,
                "n_merges": len(taxonomy["merges"]),
            }
        )
        print(
            f"bootstrap {replicate}/{args.replicates}: "
            f"{n_leaves} leaves, {n_roots} roots",
            flush=True,
        )

    output = baseline_nodes[
        [
            "tree_node_id",
            "node_kind",
            "is_root",
            "n_aligned_leaves",
            "n_local_clusters",
            "local_cluster_descendants",
        ]
    ].copy()
    output["bootstrap_support"] = [
        support_counts[clade] / args.replicates for clade in baseline_clades
    ]
    output["n_bootstrap"] = args.replicates
    evaluation = run_dir / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    support_path = evaluation / "taxonomy_branch_stability.csv"
    replicate_path = evaluation / "bootstrap_replicate_summary.csv"
    stable_merges_path = evaluation / "high_confidence_taxonomy_merges.csv"
    stable_edges_path = evaluation / "high_confidence_taxonomy_edges.csv"
    support_tmp = support_path.with_suffix(".csv.partial")
    replicate_tmp = replicate_path.with_suffix(".csv.partial")
    output.to_csv(support_tmp, index=False)
    pd.DataFrame(replicate_rows).to_csv(replicate_tmp, index=False)
    support_tmp.replace(support_path)
    replicate_tmp.replace(replicate_path)
    parent_support = output.set_index("tree_node_id")["bootstrap_support"]
    stable_parents = set(
        parent_support[parent_support >= args.support_threshold].index
    )
    stable_merges = baseline_merges[
        baseline_merges["parent"].astype(str).isin(stable_parents)
    ].copy()
    stable_merges["bootstrap_support"] = stable_merges["parent"].map(
        parent_support
    )
    stable_edges = baseline_edges[
        baseline_edges["parent"].astype(str).isin(stable_parents)
    ].copy()
    stable_edges["parent_bootstrap_support"] = stable_edges["parent"].map(
        parent_support
    )
    stable_merges.to_csv(stable_merges_path, index=False)
    stable_edges.to_csv(stable_edges_path, index=False)
    cells_path = evaluation / "cells_with_all_groundtruth.csv"
    if cells_path.exists():
        from forcitepool.evaluate import plot_truth_colored_tree

        cells = pd.read_csv(cells_path, low_memory=False)
        summary_path = evaluation / "evaluation_summary.json"
        evaluation_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else {}
        )
        truth_key = evaluation_summary.get("tree_truth_key")
        truth_file_token = evaluation_summary.get(
            "tree_truth_file_token",
            truth_key,
        )
        if truth_key in cells:
            plot_truth_colored_tree(
                cells,
                baseline_nodes,
                baseline_edges,
                evaluation / f"taxonomy_{truth_file_token}_tree_bootstrap.png",
                truth_key=truth_key,
                node_support=parent_support.to_dict(),
                support_threshold=args.support_threshold,
            )
    print(
        "Bootstrap complete: "
        f"median leaves={np.median(leaf_counts):.1f}, "
        f"median roots={np.median(root_counts):.1f}"
    )
    print(f"Support table: {support_path}")
    print(
        f"High-confidence merges (>={args.support_threshold:.2f}): "
        f"{len(stable_merges)}/{len(baseline_merges)}"
    )


if __name__ == "__main__":
    main()
