"""
Neuromorphic Swarm Control Package
===================================

SNN-controlled robot swarm with SCPN-inspired collective fields.

Each agent has its own SNN brain (sensory → recurrent → motor).
Agents communicate via shared chemical, emotional, and symbolic fields.
Swarm behavior emerges from collective neural dynamics.
Neuroevolution co-evolves agent policies.

Modules:
    agent           - Single swarm agent with SNN brain
    swarm_env       - 2D environment (obstacles, targets, boundaries)
    collective_fields - SCPN L2/L5/L7 shared fields
    communication   - Inter-agent spike broadcast + chemical diffusion
    fitness         - Swarm metrics (coverage, cohesion, alignment)
    neuroevolution_swarm - Genetic co-evolution for swarm policies

Author: Claude (Session 2026-02-16)
"""

from .agent import SwarmAgent
from .swarm_env import SwarmEnvironment
from .collective_fields import CollectiveFields
from .communication import SwarmCommunication
from .fitness import SwarmFitness
from .neuroevolution_swarm import SwarmEvolver

__all__ = [
    "SwarmAgent",
    "SwarmEnvironment",
    "CollectiveFields",
    "SwarmCommunication",
    "SwarmFitness",
    "SwarmEvolver",
]
