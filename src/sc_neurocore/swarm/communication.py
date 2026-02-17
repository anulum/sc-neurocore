"""
Swarm Communication — Inter-agent coupling
============================================

Handles:
- Spike broadcast: agents broadcast firing rates to neighbors
- Chemical pheromone deposition from neural activity
- Emotional state propagation via mean-field coupling
- Symbolic glyph imprinting from internal state

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .agent import SwarmAgent
from .collective_fields import CollectiveFields
from .swarm_env import SwarmEnvironment


class SwarmCommunication:
    """
    Manages inter-agent communication through collective fields.

    Each tick:
    1. Agents deposit chemicals proportional to neural activity
    2. Agents imprint symbolic glyphs from emotional state
    3. Emotions are synchronized via mean-field coupling
    4. Fields diffuse
    """

    def __init__(
        self,
        env: SwarmEnvironment,
        fields: CollectiveFields,
        broadcast_radius: float = 15.0,
    ):
        self.env = env
        self.fields = fields
        self.broadcast_radius = broadcast_radius

    def step(self, dt: float = 1.0):
        """Execute one communication cycle."""
        agents = self.env.agents

        # 1. Chemical deposition (L2)
        for agent in agents:
            self.fields.deposit_chemical(
                agent.position[0], agent.position[1], agent.chemical_output
            )

        # 2. Symbolic glyph imprinting (L7)
        for agent in agents:
            glyph = agent.emotions[:2] - 0.5  # Center around 0
            for ch in range(2):
                self.fields.deposit_symbolic(
                    agent.position[0], agent.position[1], ch, float(glyph[ch] * 0.1)
                )

        # 3. Emotional synchronization (L5)
        self.fields.synchronize_emotions()

        # 4. Diffuse fields
        self.fields.diffuse(dt)

    def get_sensory_data(self, agent_idx: int) -> dict:
        """
        Get communication-related sensory data for an agent.

        Returns dict with:
            chem_gradient: (2,) chemical gradient at agent position
            symbolic_value: (2,) symbolic field value at agent position
        """
        agent = self.env.agents[agent_idx]
        return {
            "chem_gradient": self.fields.get_chemical_gradient(
                agent.position[0], agent.position[1]
            ),
            "symbolic_value": self.fields.get_symbolic_at(agent.position[0], agent.position[1]),
        }
