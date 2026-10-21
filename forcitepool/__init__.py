"""Standalone CITE-seq protein taxonomy extension for CytoFuse.

Everything in this package is additive.  The original :mod:`cytofuse` package
is imported as a dependency but is never modified.
"""

from .tree import (
    StateThresholds,
    align_clusters,
    build_cluster_marker_states,
    build_taxonomy,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "StateThresholds",
    "align_clusters",
    "build_cluster_marker_states",
    "build_taxonomy",
]
