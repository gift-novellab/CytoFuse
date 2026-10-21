#!/usr/bin/env python3
"""Fit FACS panels and build a panel-aware fingerprint atlas."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
PACKAGE_ROOT = MODULE_DIR.parent
RESULTS_ROOT = PROJECT_ROOT / "outputs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cytofuse.facs.atlas import build_facs_atlas, save_facs_atlas
from cytofuse.facs.facs_pipeline import DEFAULT_PANELS, fit_panel
from cytofuse.facs.workflow import FacsWorkflow
from cytofuse.methods.base import PlatformConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "facs_example")
    parser.add_argument("--panels", nargs="*", default=list(DEFAULT_PANELS))
    parser.add_argument("--panel-path", action="append", default=[], help="Override panel path as PANEL=/path/file.h5ad")
    parser.add_argument("--max-cells", type=int, default=300_000, help="Maximum cells per panel; use 0 for all cells.")
    parser.add_argument("--sample-key", default="batch")
    parser.add_argument("--k-init", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--min-bhattacharyya", type=float, default=0.08)
    parser.add_argument("--min-marker-component-weight", type=float, default=0.01)
    parser.add_argument("--min-marker-posterior-confidence", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--atlas-min-edge-score", type=float, default=0.78)
    parser.add_argument("--atlas-min-shared-markers", type=int, default=3)
    return parser.parse_args()


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_all_panels.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def panel_paths(overrides: list[str]) -> dict[str, Path]:
    paths = dict(DEFAULT_PANELS)
    for item in overrides:
        if "=" not in item:
            raise ValueError("--panel-path must be formatted as PANEL=/path/file.h5ad")
        key, value = item.split("=", 1)
        paths[key] = Path(value)
    return paths


def main():
    args = parse_args()
    workflow = FacsWorkflow(PlatformConfig("facs", args.output, args.seed))
    log_path = setup_logging(args.output)
    paths = panel_paths(args.panel_path)
    selected = [str(x) for x in args.panels]
    missing = sorted(set(selected) - set(paths))
    if missing:
        raise ValueError(f"Unknown panels: {missing}")

    config = vars(args).copy()
    config["output"] = str(args.output)
    config["panel_paths"] = {panel: str(paths[panel]) for panel in selected}
    with open(args.output / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    metrics_path = args.output / "panel_metrics.csv"
    previous = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    records = {str(row["panel"]): row for row in previous}

    for pos, panel in enumerate(selected, start=1):
        if not args.force and workflow.is_complete(panel):
            logging.info("[%d/%d] Skip completed %s", pos, len(selected), panel)
            continue
        logging.info("[%d/%d] Fit FACS panel %s from %s", pos, len(selected), panel, paths[panel])
        started = time.monotonic()
        try:
            row = fit_panel(
                input_path=paths[panel],
                panel_id=panel,
                output_dir=args.output,
                max_cells=args.max_cells,
                sample_key=args.sample_key,
                k_init=args.k_init,
                max_iter=args.max_iter,
                min_bhattacharyya=args.min_bhattacharyya,
                min_marker_component_weight=args.min_marker_component_weight,
                min_marker_posterior_confidence=args.min_marker_posterior_confidence,
                random_state=args.seed,
            )
            row["runtime_minutes"] = (time.monotonic() - started) / 60.0
            records[panel] = row
            logging.info(
                "Finished %s: K=%d, cells=%d, %.1f min",
                panel,
                row["n_phenotypes"],
                row["n_cells_used"],
                row["runtime_minutes"],
            )
        except Exception as error:
            records[panel] = {
                "panel": panel,
                "input": str(paths[panel]),
                "status": "failed",
                "runtime_minutes": (time.monotonic() - started) / 60.0,
                "error": repr(error),
            }
            logging.error("Failed %s: %s", panel, error)
            logging.error(traceback.format_exc())
            if args.stop_on_error:
                pd.DataFrame(records.values()).to_csv(metrics_path, index=False)
                raise
        pd.DataFrame(records.values()).sort_values("panel").to_csv(metrics_path, index=False)

    if not args.skip_atlas:
        complete_panels = [panel for panel in selected if workflow.is_complete(panel)]
        logging.info("Build FACS atlas from panels: %s", ", ".join(complete_panels))
        outputs = build_facs_atlas(
            result_dir=args.output,
            panels=complete_panels,
            min_edge_score=args.atlas_min_edge_score,
            min_shared_markers=args.atlas_min_shared_markers,
        )
        save_facs_atlas(outputs, args.output)
        n_local_clusters = int(outputs["cluster_signatures"]["node_id"].nunique())
        summary = pd.Series({
            "n_panels": len(complete_panels),
            "n_local_clusters": n_local_clusters,
            "n_fine_atlas_entries": int(outputs["fine_atlas_library"]["fine_atlas_id"].nunique()),
            "n_marker_union": int(outputs["panel_marker_mapping"]["common_marker"].nunique()),
            "n_match_links": int(outputs["matches"]["relation"].eq("match").sum()) if not outputs["matches"].empty else 0,
            "n_conflict_links": int(outputs["matches"]["relation"].eq("conflict").sum()) if not outputs["matches"].empty else 0,
            "n_weak_overlap_links": int(outputs["matches"]["relation"].eq("weak_overlap").sum()) if not outputs["matches"].empty else 0,
            "n_coarse_candidate_sets": int(outputs["atlas_candidate_sets"]["interpretation"].eq("coarse_entry_multiple_fine_subtypes_possible").sum()) if not outputs["atlas_candidate_sets"].empty else 0,
        })
        summary.to_frame("value").to_csv(args.output / "atlas_summary_metrics.csv")
        print(summary.to_string())
    logging.info("Done. Log: %s", log_path)


if __name__ == "__main__":
    main()
