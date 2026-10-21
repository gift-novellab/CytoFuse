#!/usr/bin/env python3
"""Fit every CyTOF sample and save resumable result summaries."""

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

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
PACKAGE_ROOT = MODULE_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "outputs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytofuse.cytof.cytof_pipeline import DEFAULT_DATA, fit_sample
from cytofuse.cytof.workflow import CytofWorkflow
from cytofuse.crosstech.cross_platform import add_display_phenotype, build_within_platform_consensus
from cytofuse.methods.base import PlatformConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "cytof_example")
    parser.add_argument("--sample-key", default="batch")
    parser.add_argument("--samples", nargs="*", default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--min-cells", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-init", type=int, default=40)
    parser.add_argument("--min-component-weight", type=float, default=0.025)
    parser.add_argument("--min-bic-gain", type=float, default=10.0)
    parser.add_argument("--min-separation", type=float, default=1.2)
    parser.add_argument("--min-bhattacharyya", type=float, default=0.08)
    parser.add_argument("--min-posterior-confidence", type=float, default=0.75)
    parser.add_argument("--min-marker-component-weight", type=float, default=0.01)
    parser.add_argument("--min-marker-posterior-confidence", type=float, default=0.60)
    parser.add_argument("--shrinkage", type=float, default=8.0)
    parser.add_argument("--alpha-pi", type=float, default=0.08)
    parser.add_argument("--min-component-cells", type=int, default=80)
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


def scalar_mode(values):
    values = pd.Series(values).dropna().astype(str)
    return values.mode().iat[0] if not values.empty else "Unknown"


def fit_one(args, source, sample_id, sample_indices):
    selected = np.asarray(sample_indices, dtype=int)
    if args.max_cells is not None and selected.size > args.max_cells:
        rng = np.random.default_rng(args.seed)
        selected = np.sort(
            rng.choice(selected, size=args.max_cells, replace=False)
        )
    adata = source[selected].to_memory()
    source_obs = source.obs.iloc[sample_indices]
    fitted = fit_sample(
        adata,
        sample_id=sample_id,
        output_dir=args.output,
        random_state=args.seed,
        save=True,
        k_init=args.k_init,
        min_component_weight=args.min_component_weight,
        min_bic_gain=args.min_bic_gain,
        min_separation=args.min_separation,
        min_bhattacharyya=args.min_bhattacharyya,
        min_mean_posterior_conf=args.min_posterior_confidence,
        shrinkage=args.shrinkage,
        alpha_pi=args.alpha_pi,
        min_component_cells=args.min_component_cells,
        merge_threshold=args.merge_threshold,
        redundant_merge_threshold=args.redundant_merge_threshold,
        min_marker_component_weight=args.min_marker_component_weight,
        min_marker_posterior_confidence=args.min_marker_posterior_confidence,
    )
    result = fitted["model_result"]
    params = result["marker_params"]
    confidence = np.asarray(result["phenotype_confidence"])
    row = {
        "sample_id": sample_id,
        "status": "complete",
        "source_n_cells": int(len(source_obs)),
        "n_cells_used": int(adata.n_obs),
        "technical_batch": scalar_mode(source_obs.get("batch")),
        "condition": scalar_mode(source_obs.get("condition")),
        "n_markers": int(len(fitted["marker_names"])),
        "n_phenotypes": int(len(result["pi"])),
        "n_unimodal": int((params["n_states"] == 1).sum()),
        "n_bimodal": int((params["n_states"] == 2).sum()),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "min_phenotype_cells": int(np.min(result["phenotype_counts"])),
        "max_phenotype_cells": int(np.max(result["phenotype_counts"])),
        "n_iterations": int(result["n_iter"]),
    }
    for truth_key in ("major_cell_type", "fine_cluster_id"):
        if truth_key not in fitted["metrics"]:
            continue
        row.update({
            f"{truth_key}_{key.lower()}": value
            for key, value in fitted["metrics"][truth_key].items()
        })
    return row


def update_summary(output_dir, records):
    summary = pd.DataFrame(records)
    if not summary.empty and "sample_id" in summary:
        summary = summary.sort_values("sample_id")
    summary.to_csv(output_dir / "batch_metrics.csv", index=False)


def save_platform_consensus(output_dir, args=None):
    metrics = pd.read_csv(output_dir / "batch_metrics.csv")
    samples = metrics.loc[metrics["status"].eq("complete"), "sample_id"].astype(str).tolist()
    if not samples:
        logging.warning("Skip platform consensus: no complete CyTOF samples")
        return
    first_cells = pd.read_csv(output_dir / f"{samples[0]}_cell_clusters.csv", nrows=1)
    minor_key = "fine_cluster_id" if "fine_cluster_id" in first_cells.columns else "major_cell_type"
    logging.info("Build CyTOF platform consensus for %d samples", len(samples))
    consensus = build_within_platform_consensus(
        result_dir=output_dir,
        batch_ids=samples,
        platform="cytof",
        major_key="major_cell_type",
        minor_key=minor_key,
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
        min_cells=2000,
        min_batch_support=3,
    )
    consensus["node_meta"].to_csv(output_dir / "platform_consensus_nodes.csv", index=False)
    consensus["cells"].to_csv(output_dir / "platform_consensus_cells.csv", index=False)
    consensus["matches"].to_csv(output_dir / "platform_consensus_matches.csv", index=False)
    consensus["marker_rank"].to_csv(output_dir / "platform_consensus_marker_rank.csv", index=False)
    logging.info(
        "Saved CyTOF platform consensus: %d phenotypes, %d cells",
        consensus["node_meta"]["consensus_phenotype"].nunique(),
        len(consensus["cells"]),
    )


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    workflow = CytofWorkflow(PlatformConfig("cytof", args.output, args.seed))
    log_path = setup_logging(args.output)
    logging.info("Input: %s", args.input)
    logging.info("Output: %s", args.output)

    source = sc.read_h5ad(args.input, backed="r")
    if args.sample_key not in source.obs:
        raise KeyError(f"{args.sample_key!r} is not available in adata.obs")
    sample_values = source.obs[args.sample_key].astype(str)
    counts = sample_values.value_counts().sort_index()
    selected = list(counts.index) if args.samples is None else [str(x) for x in args.samples]
    missing = sorted(set(selected) - set(counts.index))
    if missing:
        source.file.close()
        raise ValueError(f"Unknown samples: {missing}")

    config = vars(args).copy()
    config["input"] = str(args.input)
    config["output"] = str(args.output)
    config["selected_samples"] = selected
    with open(args.output / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    metrics_path = args.output / "batch_metrics.csv"
    previous = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    records = {str(row["sample_id"]): row for row in previous}

    total = len(selected)
    for position, sample_id in enumerate(selected, start=1):
        n_cells = int(counts[sample_id])
        if n_cells < args.min_cells:
            logging.warning(
                "[%d/%d] Skip %s: only %d cells",
                position,
                total,
                sample_id,
                n_cells,
            )
            records[sample_id] = {
                "sample_id": sample_id,
                "status": "skipped_too_few_cells",
                "source_n_cells": n_cells,
            }
            update_summary(args.output, records.values())
            continue
        if not args.force and workflow.is_complete(sample_id):
            logging.info("[%d/%d] Skip completed %s", position, total, sample_id)
            continue

        logging.info(
            "[%d/%d] Start %s (%d source cells, max %s)",
            position,
            total,
            sample_id,
            n_cells,
            "all" if args.max_cells is None else str(args.max_cells),
        )
        started = time.monotonic()
        try:
            sample_indices = np.flatnonzero(sample_values.values == sample_id)
            row = fit_one(args, source, sample_id, sample_indices)
            row["runtime_minutes"] = (time.monotonic() - started) / 60.0
            records[sample_id] = row
            logging.info(
                "Finished %s: K=%d, major ARI=%.4f, fine ARI=%.4f, %.1f min",
                sample_id,
                row["n_phenotypes"],
                row.get("major_cell_type_ari", np.nan),
                row.get("fine_cluster_id_ari", np.nan),
                row["runtime_minutes"],
            )
        except Exception as error:
            records[sample_id] = {
                "sample_id": sample_id,
                "status": "failed",
                "source_n_cells": n_cells,
                "runtime_minutes": (time.monotonic() - started) / 60.0,
                "error": repr(error),
            }
            logging.error("Failed %s: %s", sample_id, error)
            logging.error(traceback.format_exc())
            if args.stop_on_error:
                update_summary(args.output, records.values())
                source.file.close()
                raise
        update_summary(args.output, records.values())

    source.file.close()
    update_summary(args.output, records.values())
    if args.build_consensus:
        save_platform_consensus(args.output, args)
    logging.info("All requested samples processed. Log: %s", log_path)


if __name__ == "__main__":
    main()
