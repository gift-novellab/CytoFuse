#!/usr/bin/env python3
"""Run the standalone CytoFuse CITE-seq protein taxonomy workflow."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytofuse.methods.cluster import summarize_phenotypes
from cytofuse.methods.paramshared import fit_phenotypes
from forcitepool.tree import (
    StateThresholds,
    align_clusters,
    build_cell_tree_assignments,
    build_cluster_marker_states,
    build_taxonomy,
)
DEFAULT_OUTPUT = HERE / "outputs" / "run"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m forcitepool run",
        description="Fit protein clusters and build an ambiguity-safe taxonomy."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Optional batch subset. By default every batch in --batch-key is used.",
    )
    parser.add_argument("--batch-key", default="batch")
    protein_source = parser.add_mutually_exclusive_group()
    protein_source.add_argument(
        "--protein-layer",
        default=None,
        help="Use this AnnData layer instead of X for protein measurements.",
    )
    protein_source.add_argument(
        "--protein-obsm-key",
        default=None,
        help="Use this AnnData obsm matrix for protein measurements.",
    )
    parser.add_argument(
        "--protein-names-key",
        default=None,
        help="AnnData uns key containing marker names for an ndarray obsm matrix.",
    )
    parser.add_argument(
        "--normalization",
        choices=("clr", "none"),
        default="clr",
        help="Apply per-cell log1p CLR to nonnegative counts, or use preprocessed values.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-fit", action="store_true")
    parser.add_argument("--max-cells-per-batch", type=int, default=None)
    parser.add_argument("--k-init", type=int, default=40)
    parser.add_argument("--min-component-cells", type=int, default=40)
    parser.add_argument("--min-marker-weight", type=float, default=0.25)
    parser.add_argument(
        "--min-configuration-bhattacharyya",
        type=float,
        default=1.0,
        help="Minimum final Bhattacharyya distance for a marker to emit +/- calls.",
    )
    parser.add_argument("--min-state-confidence", type=float, default=0.60)
    parser.add_argument("--min-state-separation", type=float, default=0.15)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.60)
    parser.add_argument("--min-marker-effect", type=float, default=0.10)
    parser.add_argument("--min-cell-state-probability", type=float, default=0.60)
    parser.add_argument("--max-markers-per-cluster", type=int, default=40)
    parser.add_argument("--max-markers-per-state", type=int, default=20)
    parser.add_argument("--min-shared-observed", type=int, default=3)
    parser.add_argument("--min-alignment-agreement", type=float, default=0.80)
    parser.add_argument("--min-configuration-coverage", type=float, default=0.60)
    parser.add_argument("--min-prototype-batches", type=int, default=2)
    parser.add_argument("--max-alignment-neighbors", type=int, default=None)
    parser.add_argument(
        "--max-batch-conflicts",
        type=int,
        default=2,
        help="Maximum weak +/- disagreements allowed for a cross-batch match.",
    )
    parser.add_argument(
        "--allow-strong-batch-conflicts",
        action="store_true",
        help="Dummy/display mode: permit strong conflicts within the conflict budget.",
    )
    parser.add_argument("--min-tree-shared-observed", type=int, default=1)
    parser.add_argument("--min-tree-weighted-shared", type=float, default=0.50)
    parser.add_argument("--minimum-state-specificity", type=float, default=0.10)
    parser.add_argument("--score-absolute-tolerance", type=float, default=0.05)
    parser.add_argument("--score-relative-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--unweighted-scores",
        action="store_true",
        help="Ablation mode: reproduce integer-count alignment and tree scores.",
    )
    parser.add_argument(
        "--weighted-alignment",
        action="store_true",
        help="Deprecated compatibility flag; alignment is confidence-weighted by default.",
    )
    parser.add_argument(
        "--max-tree-conflicts",
        type=int,
        default=None,
        help="Hard robustness gate; eligible pairs still use the requested positive/negative/conflict priority.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def materialize_input(input_path: Path, cache_dir: Path) -> Path:
    """Return an h5ad path, expanding gzip inputs inside the workflow directory."""

    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if input_path.suffix != ".gz":
        return input_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / input_path.with_suffix("").name
    if output.exists() and output.stat().st_mtime >= input_path.stat().st_mtime:
        return output
    temporary = output.with_suffix(output.suffix + ".partial")
    with gzip.open(input_path, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(output)
    return output


def resolve_batches(
    batch_values: pd.Series,
    requested: Sequence[str] | None,
) -> list[str]:
    """Resolve all or selected batches while preserving input order."""

    if batch_values.isna().any():
        raise ValueError("batch column contains missing values")
    values = batch_values.astype(str)
    if values.str.strip().eq("").any():
        raise ValueError("batch column contains empty labels")
    available = list(pd.unique(values))
    if not available:
        raise ValueError("batch column contains no batches")
    if requested is None:
        return available
    selected = [str(batch) for batch in requested]
    if len(selected) != len(set(selected)):
        raise ValueError("--batches contains duplicate labels")
    missing = [batch for batch in selected if batch not in set(available)]
    if missing:
        raise ValueError(f"unknown batches: {missing}")
    return selected


def make_batch_file_ids(batches: Sequence[str]) -> dict[str, str]:
    """Map arbitrary batch labels to unique, path-safe cache identifiers."""

    output: dict[str, str] = {}
    used: set[str] = set()
    for batch in batches:
        label = str(batch)
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("._-")
        slug = slug[:80] or "batch"
        candidate = slug
        if candidate != label or candidate in used:
            digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:8]
            candidate = f"{slug}_{digest}"
        while candidate in used:
            digest = hashlib.sha256(
                f"{label}:{len(used)}".encode("utf-8")
            ).hexdigest()[:8]
            candidate = f"{slug}_{digest}"
        output[label] = candidate
        used.add(candidate)
    return output


def protein_matrix_and_names(
    adata_batch,
    *,
    protein_layer: str | None = None,
    protein_obsm_key: str | None = None,
    protein_names_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a validated cells × proteins matrix and marker names."""

    if protein_obsm_key is not None:
        if protein_obsm_key not in adata_batch.obsm:
            raise KeyError(f"missing protein obsm key: {protein_obsm_key}")
        source = adata_batch.obsm[protein_obsm_key]
        if isinstance(source, pd.DataFrame):
            marker_names = np.asarray(source.columns, dtype=str)
            matrix = source.to_numpy()
        else:
            matrix = source
            if protein_names_key is None:
                raise ValueError(
                    "an ndarray protein obsm requires --protein-names-key "
                    "pointing to marker names in adata.uns"
                )
            if protein_names_key not in adata_batch.uns:
                raise KeyError(f"missing protein names uns key: {protein_names_key}")
            marker_names = np.asarray(
                adata_batch.uns[protein_names_key],
                dtype=str,
            )
    else:
        if protein_names_key is not None:
            raise ValueError(
                "--protein-names-key is only valid with --protein-obsm-key"
            )
        if protein_layer is not None:
            if protein_layer not in adata_batch.layers:
                raise KeyError(f"missing protein layer: {protein_layer}")
            matrix = adata_batch.layers[protein_layer]
        else:
            matrix = adata_batch.X
        marker_names = np.asarray(adata_batch.var_names, dtype=str)

    if sp.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != adata_batch.n_obs:
        raise ValueError("protein data must be a two-dimensional cells × markers matrix")
    if matrix.shape[1] != len(marker_names):
        raise ValueError("protein marker names do not match matrix columns")
    if len(marker_names) == 0 or len(set(marker_names)) != len(marker_names):
        raise ValueError("protein marker names must be non-empty and unique")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("protein matrix contains non-finite values")
    return matrix, marker_names


def protein_data(adata_batch, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    return protein_matrix_and_names(
        adata_batch,
        protein_layer=args.protein_layer,
        protein_obsm_key=args.protein_obsm_key,
        protein_names_key=args.protein_names_key,
    )


def normalize_proteins(matrix: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return np.asarray(matrix, dtype=np.float64)
    if method != "clr":
        raise ValueError(f"unknown protein normalization: {method}")
    if np.any(matrix < 0):
        raise ValueError(
            "CLR normalization requires nonnegative protein values; "
            "use --normalization none for already transformed data"
        )
    logged = np.log1p(matrix)
    return logged - logged.mean(axis=1, keepdims=True)


def model_paths(model_dir: Path, batch: str) -> list[Path]:
    suffixes = [
        "cell_clusters.csv",
        "cluster_marker_library_valid.csv",
        "marker_params.csv",
        "protein_probabilities.npz",
    ]
    return [model_dir / f"{batch}_{suffix}" for suffix in suffixes]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_fit_fingerprint(
    args: argparse.Namespace,
    *,
    batch: str,
    batch_file_id: str,
    input_h5ad: Path,
) -> dict:
    stat = input_h5ad.stat()
    return {
        "model_version": "leiden_structural_em_v2",
        "batch": str(batch),
        "batch_file_id": str(batch_file_id),
        "input": str(input_h5ad.resolve()),
        "input_size": int(stat.st_size),
        "input_mtime_ns": int(stat.st_mtime_ns),
        "seed": int(args.seed),
        "max_cells_per_batch": args.max_cells_per_batch,
        "k_init": int(args.k_init),
        "protein_layer": args.protein_layer,
        "protein_obsm_key": args.protein_obsm_key,
        "protein_names_key": args.protein_names_key,
        "normalization": args.normalization,
        "min_configuration_bhattacharyya": float(
            args.min_configuration_bhattacharyya
        ),
        "min_component_cells": int(args.min_component_cells),
        "min_component_fraction": 0.0005,
        "merge_threshold": 0.012,
        "redundant_merge_threshold": 0.025,
        "paramshared_source_sha256": file_sha256(
            PROJECT_ROOT / "cytofuse" / "methods" / "paramshared.py"
        ),
        "cluster_source_sha256": file_sha256(
            PROJECT_ROOT / "cytofuse" / "methods" / "cluster.py"
        ),
        "tree_source_sha256": file_sha256(HERE / "tree.py"),
    }


def batch_fit_metadata_path(model_dir: Path, batch: str) -> Path:
    return model_dir / f"{batch}_fit_config.json"


def batch_fit_is_reusable(
    model_dir: Path,
    batch: str,
    fingerprint: dict,
) -> bool:
    paths = model_paths(model_dir, batch)
    metadata_path = batch_fit_metadata_path(model_dir, batch)
    if not all(path.exists() and path.stat().st_size > 0 for path in paths):
        return False
    if not metadata_path.exists() or metadata_path.stat().st_size == 0:
        return False
    try:
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return previous == fingerprint


def save_batch_fit_metadata(
    model_dir: Path,
    batch: str,
    fingerprint: dict,
) -> None:
    path = batch_fit_metadata_path(model_dir, batch)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def fit_one_batch(
    adata,
    batch: str,
    args: argparse.Namespace,
    model_dir: Path,
    batch_file_id: str,
) -> None:
    mask = np.asarray(adata.obs[args.batch_key].astype(str) == batch)
    indices = np.flatnonzero(mask)
    if args.max_cells_per_batch and len(indices) > args.max_cells_per_batch:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(
            rng.choice(indices, size=args.max_cells_per_batch, replace=False)
        )
    batch_data = adata[indices].to_memory()
    matrix, marker_names = protein_data(batch_data, args)
    normalized = normalize_proteins(matrix, args.normalization)
    result = fit_phenotypes(
        normalized,
        marker_names=marker_names,
        K_init=args.k_init,
        min_component_weight=0.025,
        min_component_weight_3=0.04,
        min_bic_gain=10.0,
        min_bhattacharyya=0.08,
        jitter_noise_sd=0.0,
        jitter_mode="none",
        shrinkage=8.0,
        alpha_theta=0.2,
        alpha_pi=0.08,
        warmup_iter=5,
        min_component_cells=min(args.min_component_cells, len(indices)),
        min_component_fraction=0.0005,
        merge_threshold=0.012,
        redundant_merge_threshold=0.025,
        min_marker_component_weight=0.01,
        min_marker_posterior_confidence=0.60,
        min_configuration_bhattacharyya=args.min_configuration_bhattacharyya,
        random_state=args.seed,
    )
    summarize_phenotypes(
        adata_batch=batch_data,
        result=result,
        marker_names=marker_names,
        batch_id=batch_file_id,
        subtype_key="subtype",
        minor_subset_key="minor_subset",
        batch_key=args.batch_key,
        save_csv=True,
        output_dir=model_dir,
    )
    for suffix in ("cell_clusters.csv", "cluster_marker_library_valid.csv"):
        path = model_dir / f"{batch_file_id}_{suffix}"
        frame = pd.read_csv(path, low_memory=False)
        frame["batch"] = str(batch)
        frame.to_csv(path, index=False)


def build_states_for_batch(
    model_dir: Path,
    batch: str,
    batch_file_id: str,
    thresholds: StateThresholds,
) -> pd.DataFrame:
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
    return build_cluster_marker_states(
        batch_id=batch,
        cluster_labels=cells["protein_phenotype"].astype(str).to_numpy(),
        marker_names=probabilities["marker_names"].astype(str),
        low_prob=probabilities["low_prob"],
        high_prob=probabilities["high_prob"],
        marker_params=params,
        thresholds=thresholds,
    )


def save_outputs(
    output_dir: Path,
    state_table: pd.DataFrame,
    alignment: dict[str, pd.DataFrame],
    taxonomy: dict,
    cell_assignments: pd.DataFrame,
) -> None:
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    state_table.to_csv(tables / "cluster_marker_states_long.csv", index=False)
    wide = state_table.pivot(
        index=["batch", "cluster", "local_node_id", "n_cells"],
        columns="marker",
        values="state",
    ).reset_index()
    wide.to_csv(tables / "cluster_marker_states_wide.csv", index=False)
    for key, frame in alignment.items():
        frame.to_csv(tables / f"alignment_{key}.csv", index=False)
    for key in ["nodes", "edges", "merges", "states", "pair_audit"]:
        taxonomy[key].to_csv(tables / f"taxonomy_{key}.csv", index=False)
    cell_assignments.to_csv(tables / "cell_taxonomy_assignments.csv", index=False)
    with (tables / "taxonomy_forest.json").open("w", encoding="utf-8") as handle:
        json.dump(taxonomy["forest"], handle, indent=2, ensure_ascii=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    model_dir = args.output / "batch_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    input_h5ad = materialize_input(args.input, args.output / "cache")
    logging.info("Reading %s", input_h5ad)
    adata = ad.read_h5ad(input_h5ad, backed="r")
    if args.batch_key not in adata.obs:
        raise KeyError(f"missing batch column: {args.batch_key}")
    args.batches = resolve_batches(adata.obs[args.batch_key], args.batches)
    batch_file_ids = make_batch_file_ids(args.batches)
    logging.info(
        "Using %d batches from %s: %s",
        len(args.batches),
        args.batch_key,
        ", ".join(args.batches),
    )

    config = vars(args).copy()
    config.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "materialized_input": str(input_h5ad),
            "batch_file_ids": batch_file_ids,
        }
    )
    with (args.output / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, default=str)

    for batch in args.batches:
        batch_file_id = batch_file_ids[batch]
        fingerprint = batch_fit_fingerprint(
            args,
            batch=batch,
            batch_file_id=batch_file_id,
            input_h5ad=input_h5ad,
        )
        if (
            not args.force_fit
            and batch_fit_is_reusable(
                model_dir,
                batch_file_id,
                fingerprint,
            )
        ):
            logging.info("Reusing completed model for %s", batch)
            continue
        started = time.monotonic()
        n_cells = int((adata.obs[args.batch_key].astype(str) == batch).sum())
        logging.info("Fitting %s (%d cells)", batch, n_cells)
        fit_one_batch(
            adata,
            batch,
            args,
            model_dir,
            batch_file_id,
        )
        save_batch_fit_metadata(
            model_dir,
            batch_file_id,
            fingerprint,
        )
        logging.info("Finished %s in %.1f min", batch, (time.monotonic() - started) / 60)

    thresholds = StateThresholds(
        min_marker_weight=args.min_marker_weight,
        min_state_confidence=args.min_state_confidence,
        min_state_separation=args.min_state_separation,
        min_cluster_fraction=args.min_cluster_fraction,
        min_marker_effect=args.min_marker_effect,
        min_cell_state_probability=args.min_cell_state_probability,
        max_markers_per_cluster=args.max_markers_per_cluster,
        max_markers_per_state=args.max_markers_per_state,
    )
    state_table = pd.concat(
        [
            build_states_for_batch(
                model_dir,
                batch,
                batch_file_ids[batch],
                thresholds,
            )
            for batch in args.batches
        ],
        ignore_index=True,
    )
    alignment = align_clusters(
        state_table,
        min_shared_observed=args.min_shared_observed,
        minimum_state_specificity=args.minimum_state_specificity,
        score_absolute_tolerance=args.score_absolute_tolerance,
        score_relative_tolerance=args.score_relative_tolerance,
        use_weighted_scores=not args.unweighted_scores,
        min_alignment_agreement=args.min_alignment_agreement,
        min_configuration_coverage=args.min_configuration_coverage,
        min_prototype_batches=args.min_prototype_batches,
        max_candidate_neighbors=args.max_alignment_neighbors,
        max_batch_conflicts=args.max_batch_conflicts,
        allow_strong_conflicts=args.allow_strong_batch_conflicts,
    )
    taxonomy = build_taxonomy(
        alignment["membership"],
        alignment["states"],
        min_tree_shared_observed=args.min_tree_shared_observed,
        min_tree_weighted_shared=args.min_tree_weighted_shared,
        max_tree_conflicts=args.max_tree_conflicts,
        minimum_state_specificity=args.minimum_state_specificity,
        score_absolute_tolerance=args.score_absolute_tolerance,
        score_relative_tolerance=args.score_relative_tolerance,
        use_weighted_scores=not args.unweighted_scores,
    )
    all_cells = pd.concat(
        [
            pd.read_csv(
                model_dir / f"{batch_file_ids[batch]}_cell_clusters.csv",
                low_memory=False,
            )
            for batch in args.batches
        ],
        ignore_index=True,
    )
    cell_assignments = build_cell_tree_assignments(
        all_cells,
        alignment["membership"],
        taxonomy["edges"],
    )
    save_outputs(
        args.output,
        state_table,
        alignment,
        taxonomy,
        cell_assignments,
    )
    adata.file.close()
    logging.info(
        "Complete: %d local clusters -> %d aligned leaves -> %d roots",
        state_table["local_node_id"].nunique(),
        alignment["membership"]["aligned_node_id"].nunique(),
        len(taxonomy["forest"]["roots"]),
    )
    logging.info("Outputs: %s", args.output / "tables")


if __name__ == "__main__":
    main()
