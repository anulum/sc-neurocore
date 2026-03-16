# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""sc_neurocore.core -- Tier: research (experimental / research)."""

__tier__ = "research"

from .mdl_parser import MindDescriptionLanguage, MDLSpecification
from .orchestrator import CognitiveOrchestrator
from .tensor_stream import TensorStream

__all__ = [
    "MindDescriptionLanguage",
    "MDLSpecification",
    "CognitiveOrchestrator",
    "TensorStream",
]
