# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.pipeline -- Tier: research (experimental / r...

"""sc_neurocore.pipeline -- Tier: research (experimental / research)."""

__tier__ = "research"

from .ingestion import DataIngestor, MultimodalDataset
from .training import SCTrainingLoop

__all__ = [
    "DataIngestor",
    "MultimodalDataset",
    "SCTrainingLoop",
]
