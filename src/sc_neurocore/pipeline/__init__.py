# SPDX-License-Identifier: AGPL-3.0-or-later
"""sc_neurocore.pipeline -- Tier: research (experimental / research)."""

__tier__ = "research"

from .ingestion import DataIngestor, MultimodalDataset
from .training import SCTrainingLoop

__all__ = [
    "DataIngestor",
    "MultimodalDataset",
    "SCTrainingLoop",
]
