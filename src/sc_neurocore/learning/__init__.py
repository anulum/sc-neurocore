# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.learning -- Tier: research (experimental /

"""sc_neurocore.learning -- Tier: research (experimental / research)."""

__tier__ = "research"

from .advanced import (
    BPTTLearner,
    EligibilityTrace,
    HomeostaticPlasticity,
    MetaLearner,
    RewardModulatedLearner,
    ShortTermPlasticity,
    StructuralPlasticity,
)
from .federated import FederatedAggregator
from .lifelong import EWC_SCLayer
from .neuroevolution import SNNGeneticEvolver

__all__ = [
    "BPTTLearner",
    "EligibilityTrace",
    "FederatedAggregator",
    "EWC_SCLayer",
    "HomeostaticPlasticity",
    "MetaLearner",
    "RewardModulatedLearner",
    "ShortTermPlasticity",
    "SNNGeneticEvolver",
    "StructuralPlasticity",
]
