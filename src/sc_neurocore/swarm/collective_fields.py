# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CollectiveFields -- chemical, emotional, and symbolic

from __future__ import annotations
from typing import Any

"""
CollectiveFields -- chemical, emotional, and symbolic field layers.

Chemical field uses 2-D Laplacian diffusion with a manual 3x3 kernel
(no scipy dependency).  Emotional fields are per-agent 8-D vectors
synchronised via mean-field coupling.  Symbolic fields carry a 2-channel
grid for abstract signalling.
"""


from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .agent import SwarmAgent
    from .swarm_env import SwarmEnvironment


@dataclass
class FieldConfig:
    """Field layer hyper-parameters."""

    grid_size: int = 50
    diffusion_rate: float = 0.1
    decay_rate: float = 0.05
    emotional_coupling: float = 0.1
    symbolic_decay: float = 0.02
    seed: int | None = None


# 3x3 discrete Laplacian kernel (second-order central differences)
_LAPLACIAN_KERNEL = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _apply_laplacian(field: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Apply 3x3 Laplacian via manual convolution (zero-padded edges).

    Parameters
    ----------
    field : ndarray, shape (H, W)

    Returns
    -------
    lap : ndarray, shape (H, W)
    """
    H, W = field.shape
    lap = np.zeros_like(field)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            w = _LAPLACIAN_KERNEL[di + 1, dj + 1]
            if w == 0.0:
                continue
            # Source slice
            si = max(0, di)
            ei = min(H, H + di)
            sj = max(0, dj)
            ej = min(W, W + dj)
            # Destination slice
            sd = max(0, -di)
            ed = min(H, H - di)
            sjd = max(0, -dj)
            ejd = min(W, W - dj)
            lap[sd:ed, sjd:ejd] += w * field[si:ei, sj:ej]
    return lap


class CollectiveFields:
    """Chemical, emotional, and symbolic field layers for swarm communication.

    Parameters
    ----------
    cfg : FieldConfig
        Field configuration.
    env_width : float
        Physical width of the environment (for coordinate mapping).
    env_height : float
        Physical height of the environment.
    n_agents : int
        Number of agents (for emotional field sizing).
    """

    def __init__(
        self,
        cfg: FieldConfig,
        env_width: float = 100.0,
        env_height: float = 100.0,
        n_agents: int = 20,
    ) -> None:
        self.cfg = cfg
        self.env_width = env_width
        self.env_height = env_height
        self.n_agents = n_agents
        self.rng = np.random.default_rng(cfg.seed)

        gs = cfg.grid_size
        self.chemical_field = np.zeros((gs, gs), dtype=np.float64)
        self.emotional_field = np.zeros((n_agents, 8), dtype=np.float64)
        self.symbolic_field = np.zeros((gs, gs, 2), dtype=np.float64)

    # ------------------------------------------------------------------
    # Coordinate mapping: continuous (x, y) -> grid (row, col)
    # ------------------------------------------------------------------

    def _to_grid(self, x: float, y: float) -> tuple[int, int]:
        gs = self.cfg.grid_size
        col = int(np.clip(x / self.env_width * gs, 0, gs - 1))
        row = int(np.clip(y / self.env_height * gs, 0, gs - 1))
        return row, col

    # ------------------------------------------------------------------
    # Chemical field
    # ------------------------------------------------------------------

    def diffuse(self, dt: float) -> None:
        """Apply Laplacian diffusion + exponential decay to the chemical field."""
        lap = _apply_laplacian(self.chemical_field)
        self.chemical_field += self.cfg.diffusion_rate * dt * lap
        self.chemical_field *= 1.0 - self.cfg.decay_rate * dt
        np.clip(self.chemical_field, 0, None, out=self.chemical_field)

    def deposit_chemical(self, x: float, y: float, amount: float) -> None:
        """Add *amount* of chemical at world coordinate ``(x, y)``."""
        if amount <= 0:
            return
        r, c = self._to_grid(x, y)
        self.chemical_field[r, c] += amount

    def get_chemical_gradient(self, x: float, y: float) -> tuple[float, float]:
        """Return normalised (dx, dy) chemical gradient at ``(x, y)``.

        Uses central differences on the grid, mapped back to world coords.
        """
        r, c = self._to_grid(x, y)
        gs = self.cfg.grid_size
        f = self.chemical_field

        # Central differences with boundary clamp
        dc = (f[r, min(c + 1, gs - 1)] - f[r, max(c - 1, 0)]) * 0.5
        dr = (f[min(r + 1, gs - 1), c] - f[max(r - 1, 0), c]) * 0.5

        # Map grid gradient -> world gradient direction
        dx = float(dc)
        dy = float(dr)
        norm = np.sqrt(dx * dx + dy * dy) + 1e-12
        return dx / norm, dy / norm

    # ------------------------------------------------------------------
    # Emotional field
    # ------------------------------------------------------------------

    def synchronize_emotions(self, coupling: float | None = None) -> None:
        """Pull each agent's emotional vector toward the swarm mean."""
        if coupling is None:
            coupling = self.cfg.emotional_coupling
        mean_emotion = self.emotional_field.mean(axis=0)
        self.emotional_field += coupling * (mean_emotion - self.emotional_field)

    # ------------------------------------------------------------------
    # Symbolic field
    # ------------------------------------------------------------------

    def get_symbolic_at(self, x: float, y: float) -> np.ndarray[Any, Any]:
        """Return the 2-channel symbolic vector at ``(x, y)``."""
        r, c = self._to_grid(x, y)
        symbolic_vector: np.ndarray[Any, Any] = self.symbolic_field[r, c].copy()
        return symbolic_vector

    def deposit_symbolic(self, x: float, y: float, channel: int, amount: float) -> None:
        """Deposit into a symbolic channel at ``(x, y)``."""
        r, c = self._to_grid(x, y)
        self.symbolic_field[r, c, channel] += amount

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def update(self, agents: list[SwarmAgent], env: SwarmEnvironment, dt: float) -> None:
        """Run one collective-field tick.

        1. Diffuse and decay chemical field.
        2. Synchronise emotional field.
        3. Decay symbolic field.
        4. Copy agent emotions into / out of emotional field.
        """
        # Push agent emotions into the field
        for idx, agent in enumerate(agents):
            if idx < self.n_agents:
                self.emotional_field[idx] = agent.emotions

        self.diffuse(dt)
        self.synchronize_emotions()

        # Symbolic decay
        self.symbolic_field *= 1.0 - self.cfg.symbolic_decay * dt

        # Pull updated emotions back to agents
        for idx, agent in enumerate(agents):
            if idx < self.n_agents:
                agent.emotions = self.emotional_field[idx].copy()
