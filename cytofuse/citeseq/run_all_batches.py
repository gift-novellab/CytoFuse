#!/usr/bin/env python3
"""Fit protein phenotypes for every batch and save evaluation summaries."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
PACKAGE_ROOT = MODULE_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "outputs"
CITESEQ_DATA = PROJECT_ROOT / "data" / "citeseq_example.h5ad"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytofuse.citeseq.workflow import CiteSeqWorkflow
from cytofuse.crosstech.cross_platform import add_display_phenotype, build_within_platform_consensus
from cytofuse.methods.cluster import evaluate_phenotypes, summarize_phenotypes
from cytofuse.methods.paramshared import fit_phenotypes
from cytofuse.methods.base import PlatformConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=CITESEQ_DATA)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "citeseq")
    parser.add_argument("--batch-key", default="batch")
    parser.add_argument("--batches", nargs="*", default=None)
    parser.add_argument("--min-cells", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-init", type=int, default=40)
    parser.add_argument("--min-component-weight", type=float, default=0.025)
    parser.add_argument("--min-bic-gain", type=float, default=10.0)
    parser.add_argument("--min-separation", type=float, default=1.0)
    parser.add_argument("--min-bhattacharyya", type=float, default=0.08)
    parser.add_argument("--min-posterior-confidence", type=float, default=0.70)
    parser.add_argument("--min-marker-component-weight", type=float, default=0.01)
    parser.add_argument("--min-marker-posterior-confidence", type=float, default=0.60)
    parser.add_argument("--min-configuration-bhattacharyya", type=float, default=1.0)
    parser.add_argument("--shrinkage", type=float, default=8.0)
    parser.add_argument("--alpha-pi", type=float, default=0.08)
    parser.add_argument("--min-component-cells", type=int, default=40)
    parser.add_argument("--merge-threshold", type=float, default=0.012)
    parser.add_argument("--redundant-merge-threshold", type=float, default=0.025)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--build-consensus", action="store_true")
    parser.add_argument("--weak-conflict-tolerance", type=int, default=0)
    parser.add_argument("--strong-conflict-reliability", type=float, default=0.18)
    parser.add_argument("--tolerant-score-delta", type=float, default=0.08)
    parser.add_argument("--min-observed-markers", type=int, default=2)
    parser.add_argument("--min-state-confidence", type=float, default=0.60)
    parser.add_argument("--min-state-separation", type=float, default=0.15)
    parser.add_argument("--min-containment", type=float, default=None)
    parser.add_argument("--min-jaccard", type=float, default=None)
    parser.add_argument("--min-configuration-coverage", type=float, default=0.60)
    return parser.parse_args()


def setup_logging(output_dir):
    log_path = output_dir / "run_all_batches.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def dense_matrix(adata):
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def clr_normalize(X):
    log_x = np.log1p(X)
    return log_x - log_x.mean(axis=1, keepdims=True)


def evaluate(cluster_result, truth_key):
    obs = cluster_result["adata_cluster"].obs
    if truth_key not in obs:
        return {}
    metrics, _, _ = evaluate_phenotypes(
        cluster_result["adata_cluster"],
        truth_key=truth_key,
        ignore_unknown=True,
        display_table=False,
    )
    return {f"{truth_key}_{key.lower()}": value for key, value in metrics.items()}


def fit_batch(adata_backed, batch_key, batch_id, output_dir, args):
    mask = np.asarray(adata_backed.obs[batch_key].astype(str) == str(batch_id))
    indices = np.flatnonzero(mask)
    adata_batch = adata_backed[indices].to_memory()
    X_norm = clr_normalize(dense_matrix(adata_batch))
    marker_names = np.asarray(adata_batch.var_names, dtype=str)

    model_result = fit_phenotypes(
        X_norm,
        marker_names=marker_names,
        K_init=args.k_init,
        min_component_weight=args.min_component_weight,
        min_component_weight_3=0.04,
        min_bic_gain=args.min_bic_gain,
        min_bhattacharyya=args.min_bhattacharyya,
        jitter_noise_sd=0.0,
        jitter_mode="none",
        shrinkage=args.shrinkage,
        alpha_theta=0.2,
        alpha_pi=args.alpha_pi,
        warmup_iter=5,
        min_component_cells=args.min_component_cells,
        min_component_fraction=0.0005,
        merge_threshold=args.merge_threshold,
        redundant_merge_threshold=args.redundant_merge_threshold,
        min_marker_component_weight=args.min_marker_component_weight,
        min_marker_posterior_confidence=args.min_marker_posterior_confidence,
        min_configuration_bhattacharyya=args.min_configuration_bhattacharyya,
        random_state=args.seed,
    )
    cluster_result = summarize_phenotypes(
        adata_batch=adata_batch,
        result=model_result,
        marker_names=marker_names,
        batch_id=str(batch_id),
        subtype_key="subtype",
        minor_subset_key="minor_subset",
        batch_key=batch_key,
        min_cluster_fraction=0.3,
        min_marker_effect=0.10,
        max_markers_per_cluster=25,
        save_csv=True,
        output_dir=output_dir,
    )

    marker_params = model_result["marker_params"]
    confidence = np.asarray(model_result["phenotype_confidence"])
    metrics = {
        "batch": str(batch_id),
        "status": "complete",
        "n_cells": int(adata_batch.n_obs),
        "n_markers": int(adata_batch.n_vars),
        "n_phenotypes": int(len(model_result["pi"])),
        "n_unimodal": int((marker_params["n_states"] == 1).sum()),
        "n_bimodal": int((marker_params["n_states"] == 2).sum()),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "min_phenotype_cells": int(np.min(model_result["phenotype_counts"])),
        "max_phenotype_cells": int(np.max(model_result["phenotype_counts"])),
        "n_iterations": int(model_result["n_iter"]),
    }
    metrics.update(evaluate(cluster_result, "subtype"))
    metrics.update(evaluate(cluster_result, "minor_subset"))
    return metrics


def update_summary(output_dir, records):
    summary = pd.DataFrame(records)
    if not summary.empty and "batch" in summary:
        summary = summary.sort_values("batch")
    summary.to_csv(output_dir / "batch_metrics.csv", index=False)


def save_platform_consensus(output_dir, args=None):
    metrics = pd.read_csv(output_dir / "batch_metrics.csv")
    batches = metrics.loc[metrics["status"].eq("complete"), "batch"].astype(str).tolist()
    if not batches:
        logging.warning("Skip platform consensus: no complete CITE-seq batches")
        return
    logging.info("Build CITE-seq platform consensus for %d batches", len(batches))
    consensus = build_within_platform_consensus(
        result_dir=output_dir,
        batch_ids=batches,
        platform="citeseq",
        major_key="subtype",
        minor_key="minor_subset",
        weak_conflict_tolerance=getattr(args, "weak_conflict_tolerance", 0),
        strong_conflict_reliability=getattr(args, "strong_conflict_reliability", 0.18),
        tolerant_score_delta=getattr(args, "tolerant_score_delta", 0.08),
        min_observed_markers=getattr(args, "min_observed_markers", 2),
        min_state_confidence=getattr(args, "min_state_confidence", 0.60),
        min_state_separation=getattr(args, "min_state_separation", 0.15),
        min_containment=getattr(args, "min_containment", None),
        min_jaccard=getattr(args, "min_jaccard", None),
        min_configuration_coverage=getattr(
            args, "min_configuration_coverage", 0.60
        ),
    )
    consensus["cells"], consensus["node_meta"] = add_display_phenotype(
        consensus["cells"],
        consensus["node_meta"],
        min_cells=500,
        min_batch_support=2,
    )
    consensus["node_meta"].to_csv(output_dir / "platform_consensus_nodes.csv", index=False)
    consensus["cells"].to_csv(output_dir / "platform_consensus_cells.csv", index=False)
    consensus["matches"].to_csv(output_dir / "platform_consensus_matches.csv", index=False)
    consensus["marker_rank"].to_csv(output_dir / "platform_consensus_marker_rank.csv", index=False)
    logging.info(
        "Saved CITE-seq platform consensus: %d phenotypes, %d cells",
        consensus["node_meta"]["consensus_phenotype"].nunique(),
        len(consensus["cells"]),
    )


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workflow = CiteSeqWorkflow(PlatformConfig("citeseq", args.output, args.seed))
    log_path = setup_logging(args.output)
    logging.info("Input: %s", args.input)
    logging.info("Output: %s", args.output)

    adata = sc.read_h5ad(args.input, backed="r")
    if args.batch_key not in adata.obs:
        raise KeyError(f"{args.batch_key!r} is not available in adata.obs")

    counts = adata.obs[args.batch_key].astype(str).value_counts().sort_index()
    selected = list(counts.index) if args.batches is None else [str(x) for x in args.batches]
    missing = sorted(set(selected) - set(counts.index))
    if missing:
        raise ValueError(f"Unknown batches: {missing}")

    config = vars(args).copy()
    config["input"] = str(args.input)
    config["output"] = str(args.output)
    config["selected_batches"] = selected
    with open(args.output / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    previous_path = args.output / "batch_metrics.csv"
    records = []
    if previous_path.exists():
        records = pd.read_csv(previous_path).to_dict("records")
    records_by_batch = {str(row["batch"]): row for row in records}

    total = len(selected)
    for position, batch_id in enumerate(selected, start=1):
        n_cells = int(counts[batch_id])
        if n_cells < args.min_cells:
            logging.warning("[%d/%d] Skip %s: only %d cells", position, total, batch_id, n_cells)
            records_by_batch[batch_id] = {
                "batch": batch_id,
                "status": "skipped_too_few_cells",
                "n_cells": n_cells,
            }
            update_summary(args.output, records_by_batch.values())
            continue
        if not args.force and workflow.is_complete(batch_id):
            logging.info("[%d/%d] Skip completed %s", position, total, batch_id)
            continue

        logging.info("[%d/%d] Start %s (%d cells)", position, total, batch_id, n_cells)
        started = time.monotonic()
        try:
            metrics = fit_batch(
                adata,
                args.batch_key,
                batch_id,
                args.output,
                args,
            )
            metrics["runtime_minutes"] = (time.monotonic() - started) / 60.0
            records_by_batch[batch_id] = metrics
            logging.info(
                "Finished %s: K=%d, subtype ARI=%.4f, %.1f min",
                batch_id,
                metrics["n_phenotypes"],
                metrics.get("subtype_ari", np.nan),
                metrics["runtime_minutes"],
            )
        except Exception as error:
            runtime = (time.monotonic() - started) / 60.0
            records_by_batch[batch_id] = {
                "batch": batch_id,
                "status": "failed",
                "n_cells": n_cells,
                "runtime_minutes": runtime,
                "error": repr(error),
            }
            logging.error("Failed %s: %s", batch_id, error)
            logging.error(traceback.format_exc())
            if args.stop_on_error:
                update_summary(args.output, records_by_batch.values())
                raise
        update_summary(args.output, records_by_batch.values())

    adata.file.close()
    update_summary(args.output, records_by_batch.values())
    if args.build_consensus:
        save_platform_consensus(args.output, args)
    logging.info("All requested batches processed. Log: %s", log_path)


if __name__ == "__main__":
    main()
