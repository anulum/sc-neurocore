"""sc_neurocore.robotics -- Tier: research (experimental / research)."""

__tier__ = "research"

from .cpg import StochasticCPG
from .swarm import SwarmCoupling

__all__ = [
    "StochasticCPG",
    "SwarmCoupling",
]
