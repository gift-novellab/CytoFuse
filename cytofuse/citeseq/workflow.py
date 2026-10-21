"""CITE-seq platform workflow definitions."""

from __future__ import annotations

from cytofuse.methods.base import PlatformConfig, ProteinPlatformWorkflow


class CiteSeqWorkflow(ProteinPlatformWorkflow):
    """CITE-seq-specific workflow shell."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config)
