"""sc_neurocore.physics -- Tier: research (experimental / research)."""

__tier__ = "research"

from .heat import StochasticHeatSolver
from .wolfram_hypergraph import WolframHypergraph

__all__ = [
    "StochasticHeatSolver",
    "WolframHypergraph",
]
