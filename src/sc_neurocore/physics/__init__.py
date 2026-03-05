# SPDX-License-Identifier: AGPL-3.0-or-later
"""sc_neurocore.physics -- Tier: research (experimental / research)."""

__tier__ = "research"

from .heat import StochasticHeatSolver
from .wolfram_hypergraph import WolframHypergraph

__all__ = [
    "StochasticHeatSolver",
    "WolframHypergraph",
]
