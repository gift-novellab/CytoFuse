"""FACS platform workflow definitions."""

from __future__ import annotations

from cytofuse.methods.base import PlatformConfig, ProteinPlatformWorkflow


class FacsWorkflow(ProteinPlatformWorkflow):
    """FACS-specific workflow shell."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config)
