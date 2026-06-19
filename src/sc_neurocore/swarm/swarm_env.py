# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SwarmEnvironment -- 2-D arena with agents, obstacles,

from __future__ import annotations
from typing import Any, Optional

"""
SwarmEnvironment -- 2-D arena with agents, obstacles, and targets.

Obstacles are circles ``(x, y, radius)``.  Targets are points ``(x, y)`` with
optional respawn-on-capture semantics.
"""


from dataclasses import dataclass

import numpy as np

from .agent import AgentConfig, SwarmAgent


@dataclass
class EnvConfig:
    """Environment hyper-parameters."""

    width: float = 100.0
    height: float = 100.0
    n_agents: int = 20
    n_obstacles: int = 5
    n_targets: int = 3
    boundary_mode: str = "wrap"  # "wrap" or "clamp"
    capture_radius: float = 3.0
    respawn_targets: bool = True
    agent_config: Optional[AgentConfig] = None
    seed: Optional[int] = None


class SwarmEnvironment:
    """2-D continuous arena for swarm simulation.

    Parameters
    ----------
    cfg : EnvConfig
        Environment configuration.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        agent_cfg = cfg.agent_config or AgentConfig()

        # --- Agents ---
        self.agents: list[SwarmAgent] = []
        for i in range(cfg.n_agents):
            a = SwarmAgent(agent_cfg, agent_id=i)
            a.position = self.rng.uniform(0, [cfg.width, cfg.height]).astype(np.float64)
            a.heading = self.rng.uniform(0, 2 * np.pi)
            self.agents.append(a)

        # --- Obstacles (x, y, radius) ---
        self.obstacles = np.zeros((cfg.n_obstacles, 3))
        for i in range(cfg.n_obstacles):
            self.obstacles[i, 0] = self.rng.uniform(10, cfg.width - 10)
            self.obstacles[i, 1] = self.rng.uniform(10, cfg.height - 10)
            self.obstacles[i, 2] = self.rng.uniform(2, 8)

        # --- Targets (x, y) ---
        self.targets = np.zeros((cfg.n_targets, 2))
        for i in range(cfg.n_targets):
            self.targets[i] = self._random_target_pos()

        self.targets_captured = 0
        self.step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_target_pos(self) -> np.ndarray[Any, Any]:
        return self.rng.uniform([5, 5], [self.cfg.width - 5, self.cfg.height - 5])

    def _apply_boundary(self, agent: SwarmAgent) -> None:
        if self.cfg.boundary_mode == "wrap":
            agent.position[0] %= self.cfg.width
            agent.position[1] %= self.cfg.height
        else:  # clamp
            agent.position[0] = np.clip(agent.position[0], 0, self.cfg.width)
            agent.position[1] = np.clip(agent.position[1], 0, self.cfg.height)

    # ------------------------------------------------------------------
    # Pairwise / neighbour queries
    # ------------------------------------------------------------------

    def get_positions(self) -> np.ndarray[Any, Any]:
        """Return (n_agents, 2) position array."""
        return np.array([a.position for a in self.agents])

    def get_headings(self) -> np.ndarray[Any, Any]:
        """Return (n_agents,) heading array."""
        return np.array([a.heading for a in self.agents])

    def get_pairwise_distances(self) -> np.ndarray[Any, Any]:
        """Return (n_agents, n_agents) Euclidean distance matrix."""
        pos = self.get_positions()
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        distances: np.ndarray[Any, Any] = np.sqrt((diff**2).sum(axis=-1))
        return distances

    def get_neighbor_distances(self, agent_idx: int, k: int = 8) -> np.ndarray[Any, Any]:
        """Return sorted distances to the *k* nearest neighbours.

        If fewer than *k* other agents exist the array is zero-padded.
        """
        pos = self.get_positions()
        diff = pos - pos[agent_idx]
        dists = np.sqrt((diff**2).sum(axis=-1))
        dists[agent_idx] = np.inf  # exclude self
        sorted_d = np.sort(dists)
        out = np.zeros(k)
        n = min(k, len(sorted_d) - 1)
        out[:n] = sorted_d[:n]
        return out

    def get_obstacle_distances(self, agent_idx: int, k: int = 3) -> np.ndarray[Any, Any]:
        """Distances to the *k* nearest obstacle surfaces (negative = inside)."""
        pos = self.agents[agent_idx].position
        centers = self.obstacles[:, :2]
        radii = self.obstacles[:, 2]
        dists = np.sqrt(((centers - pos) ** 2).sum(axis=-1)) - radii
        sorted_d = np.sort(dists)
        out = np.zeros(k)
        n = min(k, len(sorted_d))
        out[:n] = sorted_d[:n]
        return out

    def get_target_distances(self, agent_idx: int, k: int = 2) -> np.ndarray[Any, Any]:
        """Distances to the *k* nearest targets."""
        pos = self.agents[agent_idx].position
        dists = np.sqrt(((self.targets - pos) ** 2).sum(axis=-1))
        sorted_d = np.sort(dists)
        out = np.zeros(k)
        n = min(k, len(sorted_d))
        out[:n] = sorted_d[:n]
        return out

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, dt: float = 1.0, fields=None) -> None:  # type: ignore[no-untyped-def]
        """Advance the simulation by one tick.

        Parameters
        ----------
        dt : float
            Timestep (used by collective fields diffusion).
        fields : CollectiveFields, optional
            If provided, agents read/write collective fields.
        """
        cfg = self.cfg
        for idx, agent in enumerate(self.agents):
            # Build 20-channel sensory vector
            sensory = np.zeros(agent.cfg.n_sensory)
            nbr_dist = self.get_neighbor_distances(idx, k=8)
            sensory[0:8] = np.clip(nbr_dist / max(cfg.width, cfg.height), 0, 1)
            od = self.get_obstacle_distances(idx, k=3)
            sensory[8:11] = np.clip(od / 50.0, -1, 1)
            td = self.get_target_distances(idx, k=2)
            sensory[11:13] = np.clip(td / max(cfg.width, cfg.height), 0, 1)

            if fields is not None:
                gx, gy = fields.get_chemical_gradient(agent.position[0], agent.position[1])
                sensory[13:15] = [gx, gy]
                sym = fields.get_symbolic_at(agent.position[0], agent.position[1])
                sensory[15:17] = sym
                sensory[17:19] = agent.emotions[:2]
                sensory[19] = agent.chemical_output
            # else: zeros (safe defaults)

            speed, turn = agent.think(sensory)
            agent.act(speed * dt, turn * dt)
            self._apply_boundary(agent)

            # Chemical deposit
            if fields is not None:
                fields.deposit_chemical(
                    agent.position[0], agent.position[1], agent.chemical_output * dt
                )

        # --- Target capture ---
        positions = self.get_positions()
        for t_idx in range(len(self.targets)):
            dists = np.sqrt(((positions - self.targets[t_idx]) ** 2).sum(axis=-1))
            if dists.min() < cfg.capture_radius:
                self.targets_captured += 1
                if cfg.respawn_targets:
                    self.targets[t_idx] = self._random_target_pos()

        # --- Update fields ---
        if fields is not None:
            fields.update(self.agents, self, dt)

        self.step_count += 1

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot."""
        return {
            "step": self.step_count,
            "positions": self.get_positions().tolist(),
            "headings": self.get_headings().tolist(),
            "obstacles": self.obstacles.tolist(),
            "targets": self.targets.tolist(),
            "targets_captured": self.targets_captured,
        }
