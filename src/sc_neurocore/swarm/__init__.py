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
from .communication import SwarmCommunication
from .fitness import SwarmFitness
from .neuroevolution_swarm import EvolverConfig, SwarmEvolver

__all__ = [
    "AgentConfig",
    "SwarmAgent",
    "EnvConfig",
    "SwarmEnvironment",
    "FieldConfig",
    "CollectiveFields",
    "SwarmCommunication",
    "SwarmFitness",
    "SwarmEvolver",
    "EvolverConfig",
]
