# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Neuromorphic Swarm Control
==========================

Spiking-neural-network agents with neuroevolution for collective behaviour.

Modules
-------
agent              SwarmAgent with soft-LIF SNN brain
swarm_env          Grid environment with obstacles and targets
collective_fields  Chemical, emotional, and symbolic field layers
fitness            Multi-objective swarm fitness evaluation
neuroevolution_swarm  Genetic algorithm over SNN weight vectors
"""

from .agent import AgentConfig, SwarmAgent
from .swarm_env import EnvConfig, SwarmEnvironment
from .collective_fields import FieldConfig, CollectiveFields
from .fitness import SwarmFitness
from .neuroevolution_swarm import EvolverConfig, SwarmEvolver

__all__ = [
    "AgentConfig",
    "SwarmAgent",
    "EnvConfig",
    "SwarmEnvironment",
    "FieldConfig",
    "CollectiveFields",
    "SwarmFitness",
    "SwarmEvolver",
    "EvolverConfig",
]
