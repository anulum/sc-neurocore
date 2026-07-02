# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SwarmAgent -- SNN-driven agent for neuromorphic swarm

"""SwarmAgent -- SNN-driven agent for neuromorphic swarm control.

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

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np


def _ensure_positive_int(value: int, name: str, *, minimum: int = 1) -> int:
    """Return an integer value after rejecting bools and low values."""
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _ensure_finite_float(value: float, name: str, *, positive: bool = False) -> float:
    """Return a finite float value, optionally requiring strict positivity."""
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    if positive and scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


def _ensure_seed(value: Optional[int], name: str) -> Optional[int]:
    """Return a seed after rejecting bools and negative integers."""
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _validate_weight_vector(
    weights: np.ndarray[Any, Any], expected_size: int
) -> np.ndarray[Any, Any]:
    """Return a finite one-dimensional copy with the expected weight count."""
    try:
        flat = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be a finite one-dimensional float vector") from exc
    if flat.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if flat.size != expected_size:
        raise ValueError(f"Expected {expected_size} weights, got {flat.size}")
    if not np.all(np.isfinite(flat)):
        raise ValueError("weights must be finite")
    return flat.copy()


def _validate_sensory_vector(
    sensory: np.ndarray[Any, Any], expected_size: int
) -> np.ndarray[Any, Any]:
    """Return a finite one-dimensional sensory copy with the expected width."""
    try:
        vector = np.asarray(sensory, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sensory must be a finite one-dimensional float vector") from exc
    if vector.ndim != 1 or vector.size != expected_size:
        raise ValueError(f"sensory must be a one-dimensional vector of length {expected_size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("sensory must contain only finite values")
    return vector.copy()


@dataclass
class AgentConfig:
    """Hyper-parameters for a single swarm agent.

    The sensory layout reserves 20 channels for neighbour, obstacle, target,
    field, emotional, and chemical inputs. Motor output must include at least
    speed and turn channels.
    """

    n_sensory: int = 20
    n_hidden: int = 16
    n_motor: int = 2
    membrane_decay: float = 0.9
    threshold: float = 1.0
    max_speed: float = 2.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate neural dimensions, dynamics, actuation, and seed domains."""
        self.n_sensory = _ensure_positive_int(self.n_sensory, "n_sensory", minimum=20)
        self.n_hidden = _ensure_positive_int(self.n_hidden, "n_hidden")
        self.n_motor = _ensure_positive_int(self.n_motor, "n_motor", minimum=2)
        self.membrane_decay = _ensure_finite_float(self.membrane_decay, "membrane_decay")
        if not 0.0 <= self.membrane_decay < 1.0:
            raise ValueError("membrane_decay must be in [0, 1)")
        self.threshold = _ensure_finite_float(self.threshold, "threshold")
        self.max_speed = _ensure_finite_float(self.max_speed, "max_speed", positive=True)
        self.seed = _ensure_seed(self.seed, "seed")


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
        if not isinstance(agent_id, int) or isinstance(agent_id, bool) or agent_id < 0:
            raise ValueError("agent_id must be a non-negative integer")
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
        """Return the flat trainable-weight vector length."""
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
        """Replace trainable weights from a finite exact-size vector."""
        c = self.cfg
        validated = _validate_weight_vector(flat, self.n_weights)
        offset = 0
        size_in = c.n_hidden * c.n_sensory
        next_w_in = validated[offset : offset + size_in].reshape(c.n_hidden, c.n_sensory).copy()
        offset += size_in

        size_rec = c.n_hidden * c.n_hidden
        next_w_rec = validated[offset : offset + size_rec].reshape(c.n_hidden, c.n_hidden).copy()
        offset += size_rec

        size_out = c.n_motor * c.n_hidden
        next_w_out = validated[offset : offset + size_out].reshape(c.n_motor, c.n_hidden).copy()

        self.W_in = next_w_in
        self.W_rec = next_w_rec
        self.W_out = next_w_out

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
        inp = _validate_sensory_vector(sensory, c.n_sensory)

        with np.errstate(over="ignore", invalid="ignore"):
            candidate_membrane = (
                c.membrane_decay * self.membrane + self.W_in @ inp + self.W_rec @ self.firing_rate
            )
            spike_prob = 1.0 / (1.0 + np.exp(-(candidate_membrane - c.threshold)))
            candidate_firing_rate = 0.8 * self.firing_rate + 0.2 * spike_prob

            # Reset membrane where spike probability high.
            candidate_membrane = candidate_membrane * (1.0 - spike_prob)

            motor = self.W_out @ candidate_firing_rate
            speed = (np.tanh(motor[0]) + 1.0) * 0.5 * c.max_speed  # [0, max_speed]
            turn = np.tanh(motor[1]) * np.pi  # [-pi, pi]
        chemical_output = float(np.clip(inp[-1], 0, 1))

        if (
            not np.all(np.isfinite(candidate_membrane))
            or not np.all(np.isfinite(candidate_firing_rate))
            or not math.isfinite(float(speed))
            or not math.isfinite(float(turn))
        ):
            raise ValueError("sensory produced a non-finite swarm-agent state")

        self.membrane = candidate_membrane.astype(np.float64, copy=True)
        self.firing_rate = candidate_firing_rate.astype(np.float64, copy=True)
        self.chemical_output = chemical_output
        return float(speed), float(turn)

    # ------------------------------------------------------------------
    # Kinematic update
    # ------------------------------------------------------------------

    def act(self, speed: float, turn: float) -> None:
        """Update position and heading given finite motor commands."""
        speed_value = _ensure_finite_float(speed, "speed")
        turn_value = _ensure_finite_float(turn, "turn")
        self.heading = (self.heading + turn_value) % (2 * np.pi)
        dx = speed_value * np.cos(self.heading)
        dy = speed_value * np.sin(self.heading)
        self.position[0] += dx
        self.position[1] += dy

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self, rng: np.random.Generator | None = None, width: float = 100.0, height: float = 100.0
    ) -> None:
        """Reset kinematic and neural state while preserving trainable weights."""
        width_value = _ensure_finite_float(width, "width", positive=True)
        height_value = _ensure_finite_float(height, "height", positive=True)
        if rng is None:
            rng = np.random.default_rng()
        elif not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        next_position = rng.uniform(0, [width_value, height_value]).astype(np.float64)
        next_heading = float(rng.uniform(0, 2 * np.pi))

        self.membrane[:] = 0.0
        self.firing_rate[:] = 0.0
        self.position = next_position
        self.heading = next_heading
        self.emotions[:] = 0.0
        self.chemical_output = 0.0
