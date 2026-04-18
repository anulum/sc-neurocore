# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SwarmAgent -- SNN-driven agent for neuromorphic swarm

from __future__ import annotations
from typing import Any, Optional

"""
SwarmAgent -- SNN-driven agent for neuromorphic swarm control.

Each agent carries a small spiking neural network (soft-LIF) whose weights
are a flat vector suitable for genetic-algorithm optimisation.

Sensory layout (20 channels)
-----------------------------
 0..7   nearest-neighbour distances   (8)
 8..10  nearest-obstacle distances    (3)
11..12  nearest-target distances      (2)
13..14  chemical gradient (dx, dy)    (2)
15..16  symbolic field (s0, s1)       (2)
17..18  emotional state (valence, arousal) (2)
19      chemical output amount        (1)

Motor output (2 channels)
--------------------------
 0  speed   (tanh -> [-1, 1], scaled to [0, max_speed])
 1  turn    (tanh -> [-pi, pi])
"""


from dataclasses import dataclass

import numpy as np


@dataclass
class AgentConfig:
    """Hyper-parameters for a single swarm agent."""

    n_sensory: int = 20
    n_hidden: int = 16
    n_motor: int = 2
    membrane_decay: float = 0.9
    threshold: float = 1.0
    max_speed: float = 2.0
    seed: Optional[int] = None


class SwarmAgent:
    """Spiking-neural-network agent with soft-LIF dynamics.

    Parameters
    ----------
    cfg : AgentConfig
        Neuron and network parameters.
    agent_id : int
        Unique identifier within the swarm.
    """

    def __init__(self, cfg: AgentConfig, agent_id: int = 0) -> None:
        self.cfg = cfg
        self.agent_id = agent_id
        rng = np.random.default_rng(cfg.seed)

        # --- Weight matrices (Xavier-ish init) ---
        scale_in = np.sqrt(2.0 / (cfg.n_sensory + cfg.n_hidden))
        scale_rec = np.sqrt(2.0 / (cfg.n_hidden + cfg.n_hidden))
        scale_out = np.sqrt(2.0 / (cfg.n_hidden + cfg.n_motor))

        self.W_in = rng.normal(0, scale_in, (cfg.n_hidden, cfg.n_sensory))
        self.W_rec = rng.normal(0, scale_rec, (cfg.n_hidden, cfg.n_hidden))
        self.W_out = rng.normal(0, scale_out, (cfg.n_motor, cfg.n_hidden))

        # --- Neuron state ---
        self.membrane = np.zeros(cfg.n_hidden)
        self.firing_rate = np.zeros(cfg.n_hidden)

        # --- Kinematic state ---
        self.position = rng.uniform(0, 100, size=2).astype(np.float64)
        self.heading = rng.uniform(0, 2 * np.pi)

        # --- Emotional / chemical state ---
        self.emotions = np.zeros(8)
        self.chemical_output = 0.0

    # ------------------------------------------------------------------
    # Weight vector (flat) for genetic algorithm
    # ------------------------------------------------------------------

    @property
    def n_weights(self) -> int:
        c = self.cfg
        return c.n_hidden * c.n_sensory + c.n_hidden * c.n_hidden + c.n_motor * c.n_hidden

    @property
    def weights(self) -> np.ndarray[Any, Any]:
        """Return all trainable weights as a flat 1-D vector."""
        return np.concatenate(
            [
                self.W_in.ravel(),
                self.W_rec.ravel(),
                self.W_out.ravel(),
            ]
        )

    @weights.setter
    def weights(self, flat: np.ndarray[Any, Any]) -> None:
        c = self.cfg
        if flat.size != self.n_weights:
            raise ValueError(f"Expected {self.n_weights} weights, got {flat.size}")
        offset = 0
        size_in = c.n_hidden * c.n_sensory
        self.W_in = flat[offset : offset + size_in].reshape(c.n_hidden, c.n_sensory).copy()
        offset += size_in

        size_rec = c.n_hidden * c.n_hidden
        self.W_rec = flat[offset : offset + size_rec].reshape(c.n_hidden, c.n_hidden).copy()
        offset += size_rec

        size_out = c.n_motor * c.n_hidden
        self.W_out = flat[offset : offset + size_out].reshape(c.n_motor, c.n_hidden).copy()

    # ------------------------------------------------------------------
    # Neural forward pass (soft-LIF)
    # ------------------------------------------------------------------

    def think(self, sensory: np.ndarray[Any, Any]) -> tuple[float, float]:
        """Run one SNN tick and return ``(speed, turn_angle)``.

        Parameters
        ----------
        sensory : ndarray, shape (n_sensory,)
            Normalised sensory input vector.

        Returns
        -------
        speed : float  in [0, max_speed]
        turn  : float  in [-pi, pi]
        """
        c = self.cfg
        inp = np.asarray(sensory, dtype=np.float64).ravel()[: c.n_sensory]

        # Membrane integration
        self.membrane = (
            c.membrane_decay * self.membrane + self.W_in @ inp + self.W_rec @ self.firing_rate  # type: ignore[assignment]
        )

        # Soft spike (sigmoid pseudo-rate)
        spike_prob = 1.0 / (1.0 + np.exp(-(self.membrane - c.threshold)))
        self.firing_rate = 0.8 * self.firing_rate + 0.2 * spike_prob  # type: ignore[assignment]

        # Reset membrane where spike probability high
        self.membrane *= 1.0 - spike_prob

        # Motor readout
        motor = self.W_out @ self.firing_rate
        speed = (np.tanh(motor[0]) + 1.0) * 0.5 * c.max_speed  # [0, max_speed]
        turn = np.tanh(motor[1]) * np.pi  # [-pi, pi]

        # Side-effect: chemical output from last sensory channel
        self.chemical_output = float(np.clip(sensory[-1] if len(sensory) > 19 else 0.0, 0, 1))

        return float(speed), float(turn)

    # ------------------------------------------------------------------
    # Kinematic update
    # ------------------------------------------------------------------

    def act(self, speed: float, turn: float) -> None:
        """Update position and heading given motor commands."""
        self.heading = (self.heading + turn) % (2 * np.pi)
        dx = speed * np.cos(self.heading)
        dy = speed * np.sin(self.heading)
        self.position[0] += dx
        self.position[1] += dy

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, rng: np.random.Generator | None = None, width: float = 100.0, height: float = 100.0
    ) -> None:
        """Reset kinematic and neural state (weights untouched)."""
        if rng is None:
            rng = np.random.default_rng()
        self.membrane[:] = 0.0
        self.firing_rate[:] = 0.0
        self.position = rng.uniform(0, [width, height]).astype(np.float64)
        self.heading = rng.uniform(0, 2 * np.pi)
        self.emotions[:] = 0.0
        self.chemical_output = 0.0
