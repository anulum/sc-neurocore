# SPDX-License-Identifier: AGPL-3.0-or-later
"""sc_neurocore.learning -- Tier: research (experimental / research)."""

__tier__ = "research"

from .federated import FederatedAggregator
from .lifelong import EWC_SCLayer
from .neuroevolution import SNNGeneticEvolver

__all__ = [
    "FederatedAggregator",
    "EWC_SCLayer",
    "SNNGeneticEvolver",
]
