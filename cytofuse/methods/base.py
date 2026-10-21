"""Shared platform workflow primitives.

Platform modules keep data loading and marker selection local, while this module
owns common result naming, completion checks, and a small inheritance surface for
publication-quality runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PlatformConfig:
    """Common configuration for a protein-platform workflow."""

    platform: str
    output_dir: Path
    random_state: int = 0

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)


class ProteinPlatformWorkflow:
    """Base class for CITE-seq, CyTOF, and FACS platform runners."""

    cell_suffix = "cell_clusters.csv"
    library_suffix = "cluster_marker_library_valid.csv"
    marker_suffix = "marker_params.csv"
    probability_suffix = "protein_probabilities.npz"

    def __init__(self, config: PlatformConfig):
        self.config = config
        self.output_dir = config.resolved_output_dir()

    def result_paths(self, unit_id: str) -> list[Path]:
        unit = str(unit_id)
        return [
            self.output_dir / f"{unit}_{self.cell_suffix}",
            self.output_dir / f"{unit}_{self.library_suffix}",
            self.output_dir / f"{unit}_{self.marker_suffix}",
            self.output_dir / f"{unit}_{self.probability_suffix}",
        ]

    def is_complete(self, unit_id: str) -> bool:
        return all(path.exists() and path.stat().st_size > 0 for path in self.result_paths(unit_id))

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir


def completed_units(workflow: ProteinPlatformWorkflow, unit_ids: Iterable[str]) -> list[str]:
    """Return unit IDs with all standard result files present."""

    return [str(unit_id) for unit_id in unit_ids if workflow.is_complete(str(unit_id))]
