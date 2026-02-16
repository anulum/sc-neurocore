"""
Swarm Agent — Single agent with SNN brain
==========================================

Each agent has:
- Position + heading in 2D space
- SNN brain: sensory → dense → recurrent → motor decoder
- Chemical secretion (L2-inspired)
- Emotional state (L5-inspired)

The SNN uses SCDenseLayer-style weight matrices with LIF neurons
operating in "soft simulation" mode (probability-domain) for performance.

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AgentConfig:
    """Configuration for a single swarm agent."""
    n_sensory: int = 20
    n_hidden: int = 16
    n_motor: int = 2   # speed, turn_angle
    tau_mem: float = 20.0
    v_threshold: float = 1.0
    noise_std: float = 0.02
    max_speed: float = 1.5
    max_turn: float = np.pi / 4  # 45 degrees max turn per step
    chemical_secretion_rate: float = 0.1
    n_emotional_dims: int = 8


class SwarmAgent:
    """
    Single swarm agent with an SNN brain.

    Sensory inputs (20 channels):
        [0:8]   - distances to 8 nearest neighbors (normalized)
        [8:11]  - distances to 3 nearest obstacles (normalized)
        [11:13] - distances to 2 nearest targets (normalized)
        [13:15] - chemical field gradient (x, y) at position
        [15:17] - symbolic field value at position (2 components)
        [17:19] - own emotional state (valence, arousal)
        [19]    - own chemical concentration

    Motor outputs (2 channels):
        [0] - speed (0 to max_speed)
        [1] - turn angle (-max_turn to max_turn)
    """

    def __init__(
        self,
        agent_id: int,
        config: Optional[AgentConfig] = None,
        seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.config = config or AgentConfig()
        self.rng = np.random.RandomState(seed)

        # Position and heading
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0  # radians

        # SNN weights
        self.W_in = self.rng.uniform(0, 0.3, (self.config.n_hidden, self.config.n_sensory))
        self.W_rec = self.rng.uniform(0, 0.15, (self.config.n_hidden, self.config.n_hidden))
        self.W_out = self.rng.uniform(0, 0.4, (self.config.n_motor, self.config.n_hidden))

        # Spectral radius scaling for recurrent weights
        eigvals = np.abs(np.linalg.eigvals(self.W_rec))
        if eigvals.max() > 0:
            self.W_rec *= 0.9 / eigvals.max()

        # Neural state
        self.membrane = np.zeros(self.config.n_hidden)
        self.hidden_state = np.zeros(self.config.n_hidden)
        self.firing_rates = np.zeros(self.config.n_hidden)

        # Emotional state (L5-inspired): 8 dims
        self.emotions = np.full(self.config.n_emotional_dims, 0.5)

        # Chemical secretion level
        self.chemical_output = 0.0

        # Collected weights as flat array for evolution
        self._weights_flat: Optional[np.ndarray] = None

    @property
    def weights(self) -> np.ndarray:
        """Flat weight vector for genetic algorithm."""
        if self._weights_flat is None:
            self._weights_flat = np.concatenate([
                self.W_in.ravel(), self.W_rec.ravel(), self.W_out.ravel()
            ])
        return self._weights_flat

    @weights.setter
    def weights(self, flat: np.ndarray):
        """Set weights from flat vector."""
        self._weights_flat = flat.copy()
        n_in = self.config.n_hidden * self.config.n_sensory
        n_rec = self.config.n_hidden * self.config.n_hidden
        n_out = self.config.n_motor * self.config.n_hidden
        idx = 0
        self.W_in = flat[idx:idx + n_in].reshape(self.config.n_hidden, self.config.n_sensory)
        idx += n_in
        self.W_rec = flat[idx:idx + n_rec].reshape(self.config.n_hidden, self.config.n_hidden)
        idx += n_rec
        self.W_out = flat[idx:idx + n_out].reshape(self.config.n_motor, self.config.n_hidden)

    def sense(
        self,
        neighbor_dists: np.ndarray,
        obstacle_dists: np.ndarray,
        target_dists: np.ndarray,
        chem_gradient: np.ndarray,
        symbolic_value: np.ndarray,
    ) -> np.ndarray:
        """
        Build sensory input vector from environment data.

        All distances are pre-normalized to [0, 1].
        """
        sensory = np.zeros(self.config.n_sensory)

        # Pad/truncate to expected sizes
        n_n = min(len(neighbor_dists), 8)
        sensory[:n_n] = neighbor_dists[:n_n]
        sensory[n_n:8] = 1.0  # far away if fewer than 8 neighbors

        n_o = min(len(obstacle_dists), 3)
        sensory[8:8 + n_o] = obstacle_dists[:n_o]
        sensory[8 + n_o:11] = 1.0

        n_t = min(len(target_dists), 2)
        sensory[11:11 + n_t] = target_dists[:n_t]
        sensory[11 + n_t:13] = 1.0

        sensory[13:15] = np.clip(chem_gradient[:2], 0, 1)
        sensory[15:17] = np.clip(symbolic_value[:2], 0, 1)
        sensory[17] = self.emotions[0]  # valence
        sensory[18] = self.emotions[1]  # arousal
        sensory[19] = np.clip(self.chemical_output, 0, 1)

        return sensory

    def think(self, sensory_input: np.ndarray) -> np.ndarray:
        """
        SNN forward pass: sensory → hidden (with recurrence) → motor.

        Uses "soft simulation" (probability-domain) for speed.
        Returns [speed, turn_angle] as raw motor commands.
        """
        # Input current
        I_in = self.W_in @ sensory_input

        # Recurrent current
        I_rec = self.W_rec @ self.hidden_state

        # Total current + noise
        I_total = I_in + I_rec
        noise = self.rng.normal(0, self.config.noise_std, self.config.n_hidden)

        # LIF membrane dynamics (soft)
        decay = np.exp(-1.0 / self.config.tau_mem)
        self.membrane = self.membrane * decay + I_total + noise

        # Spike / firing rate
        spikes = (self.membrane >= self.config.v_threshold).astype(float)
        self.membrane[spikes > 0] = 0.0  # reset
        self.firing_rates = 0.9 * self.firing_rates + 0.1 * spikes
        self.hidden_state = self.firing_rates

        # Motor decode
        raw_motor = self.W_out @ self.firing_rates

        # Map to speed and turn
        speed = float(np.clip(raw_motor[0], 0, 1)) * self.config.max_speed
        turn = float(np.tanh(raw_motor[1])) * self.config.max_turn

        # Update chemical output based on neural activity
        self.chemical_output = float(np.mean(self.firing_rates)) * self.config.chemical_secretion_rate

        return np.array([speed, turn])

    def act(self, speed: float, turn_angle: float):
        """Update position and heading based on motor commands."""
        self.heading += turn_angle
        self.heading %= 2 * np.pi
        self.x += speed * np.cos(self.heading)
        self.y += speed * np.sin(self.heading)

    def reset_neural_state(self):
        """Reset membrane potentials and firing rates."""
        self.membrane = np.zeros(self.config.n_hidden)
        self.hidden_state = np.zeros(self.config.n_hidden)
        self.firing_rates = np.zeros(self.config.n_hidden)
        self.emotions = np.full(self.config.n_emotional_dims, 0.5)
        self.chemical_output = 0.0

    def get_neural_state(self) -> Dict:
        """Return current neural state for visualization."""
        return {
            "agent_id": self.agent_id,
            "position": (self.x, self.y),
            "heading": self.heading,
            "firing_rates": self.firing_rates.tolist(),
            "membrane": self.membrane.tolist(),
            "chemical_output": self.chemical_output,
            "emotions": self.emotions.tolist(),
            "mean_activity": float(np.mean(self.firing_rates)),
        }

    def clone(self, new_id: Optional[int] = None) -> "SwarmAgent":
        """Create a copy of this agent with the same weights."""
        new_agent = SwarmAgent(
            agent_id=new_id if new_id is not None else self.agent_id,
            config=self.config,
            seed=None,
        )
        new_agent.weights = self.weights.copy()
        return new_agent
