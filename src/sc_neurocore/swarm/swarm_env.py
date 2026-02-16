"""
Swarm Environment — 2D arena with obstacles and targets
========================================================

Provides:
- 2D bounded arena (configurable size)
- Circular obstacles
- Point targets (food sources / goals)
- Distance computations (agent-agent, agent-obstacle, agent-target)
- Boundary enforcement (wrap-around or bounce)

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .agent import SwarmAgent, AgentConfig


@dataclass
class Obstacle:
    """Circular obstacle in the arena."""
    x: float
    y: float
    radius: float


@dataclass
class Target:
    """Point target (food source or goal)."""
    x: float
    y: float
    radius: float = 2.0  # Collection radius
    collected: bool = False
    respawn: bool = True


@dataclass
class EnvConfig:
    """Environment configuration."""
    width: float = 100.0
    height: float = 100.0
    n_agents: int = 20
    n_obstacles: int = 5
    n_targets: int = 3
    boundary_mode: str = "wrap"  # "wrap" or "bounce"
    max_sense_range: float = 30.0  # Sensing range for normalization
    agent_config: Optional[AgentConfig] = None  # Shared agent config
    seed: int = 42


class SwarmEnvironment:
    """
    2D arena for swarm agents with obstacles and targets.

    Manages agent placement, collision detection, target collection,
    and provides normalized distance computations for agent sensing.
    """

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Create arena elements
        self.obstacles: List[Obstacle] = []
        self.targets: List[Target] = []
        self.agents: List[SwarmAgent] = []

        self._setup_arena()
        self.tick = 0
        self._target_collections = 0

    def _setup_arena(self):
        """Initialize obstacles, targets, and agents."""
        c = self.config
        margin = 10.0

        # Place obstacles
        self.obstacles = []
        for _ in range(c.n_obstacles):
            x = self.rng.uniform(margin, c.width - margin)
            y = self.rng.uniform(margin, c.height - margin)
            r = self.rng.uniform(2.0, 5.0)
            self.obstacles.append(Obstacle(x=x, y=y, radius=r))

        # Place targets
        self.targets = []
        for _ in range(c.n_targets):
            x = self.rng.uniform(margin, c.width - margin)
            y = self.rng.uniform(margin, c.height - margin)
            self.targets.append(Target(x=x, y=y))

        # Create agents
        self.agents = []
        for i in range(c.n_agents):
            agent = SwarmAgent(agent_id=i, config=c.agent_config, seed=c.seed + i)
            agent.x = self.rng.uniform(margin, c.width - margin)
            agent.y = self.rng.uniform(margin, c.height - margin)
            agent.heading = self.rng.uniform(0, 2 * np.pi)
            self.agents.append(agent)

    def step(self, dt: float = 1.0):
        """
        Advance environment by one tick.

        - Checks target collection
        - Enforces boundaries
        - Handles obstacle collision
        """
        for agent in self.agents:
            # Boundary enforcement
            self._enforce_boundary(agent)

            # Obstacle avoidance (simple push-out)
            for obs in self.obstacles:
                dx = agent.x - obs.x
                dy = agent.y - obs.y
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist < obs.radius + 1.0:
                    # Push agent out
                    if dist > 0:
                        agent.x = obs.x + (obs.radius + 1.5) * dx / dist
                        agent.y = obs.y + (obs.radius + 1.5) * dy / dist

            # Target collection
            for target in self.targets:
                if target.collected:
                    continue
                dx = agent.x - target.x
                dy = agent.y - target.y
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist < target.radius:
                    target.collected = True
                    self._target_collections += 1
                    if target.respawn:
                        target.x = self.rng.uniform(10, self.config.width - 10)
                        target.y = self.rng.uniform(10, self.config.height - 10)
                        target.collected = False

        self.tick += 1

    def _enforce_boundary(self, agent: SwarmAgent):
        """Keep agent within arena bounds."""
        if self.config.boundary_mode == "wrap":
            agent.x %= self.config.width
            agent.y %= self.config.height
        else:  # bounce
            if agent.x < 0:
                agent.x = -agent.x
                agent.heading = np.pi - agent.heading
            elif agent.x > self.config.width:
                agent.x = 2 * self.config.width - agent.x
                agent.heading = np.pi - agent.heading
            if agent.y < 0:
                agent.y = -agent.y
                agent.heading = -agent.heading
            elif agent.y > self.config.height:
                agent.y = 2 * self.config.height - agent.y
                agent.heading = -agent.heading

    def get_agent_positions(self) -> np.ndarray:
        """Return (N, 2) array of agent positions."""
        return np.array([[a.x, a.y] for a in self.agents])

    def get_pairwise_distances(self) -> np.ndarray:
        """Return (N, N) pairwise distance matrix."""
        pos = self.get_agent_positions()
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def get_neighbor_distances(self, agent_idx: int, k: int = 8) -> np.ndarray:
        """Return sorted distances to k nearest neighbors (normalized)."""
        pos = self.get_agent_positions()
        dx = pos[:, 0] - self.agents[agent_idx].x
        dy = pos[:, 1] - self.agents[agent_idx].y
        dists = np.sqrt(dx ** 2 + dy ** 2)
        dists[agent_idx] = np.inf  # Exclude self
        sorted_dists = np.sort(dists)[:k]
        return np.clip(sorted_dists / self.config.max_sense_range, 0, 1)

    def get_obstacle_distances(self, agent_idx: int, k: int = 3) -> np.ndarray:
        """Return distances to k nearest obstacles (normalized)."""
        agent = self.agents[agent_idx]
        dists = []
        for obs in self.obstacles:
            d = np.sqrt((agent.x - obs.x) ** 2 + (agent.y - obs.y) ** 2) - obs.radius
            dists.append(max(0.0, d))
        dists = np.array(sorted(dists)[:k])
        return np.clip(dists / self.config.max_sense_range, 0, 1)

    def get_target_distances(self, agent_idx: int, k: int = 2) -> np.ndarray:
        """Return distances to k nearest uncollected targets (normalized)."""
        agent = self.agents[agent_idx]
        dists = []
        for target in self.targets:
            if not target.collected:
                d = np.sqrt((agent.x - target.x) ** 2 + (agent.y - target.y) ** 2)
                dists.append(d)
        if not dists:
            return np.ones(k)
        dists = np.array(sorted(dists)[:k])
        result = np.ones(k)
        result[:len(dists)] = np.clip(dists / self.config.max_sense_range, 0, 1)
        return result

    def reset(self, keep_agents: bool = False):
        """Reset environment. Optionally keep agent weights."""
        if keep_agents:
            weights = [a.weights.copy() for a in self.agents]

        self._setup_arena()
        self.tick = 0
        self._target_collections = 0

        if keep_agents:
            for agent, w in zip(self.agents, weights):
                agent.weights = w

    def get_state(self) -> Dict:
        """Return full environment state for visualization."""
        return {
            "tick": self.tick,
            "agent_positions": self.get_agent_positions().tolist(),
            "agent_headings": [a.heading for a in self.agents],
            "obstacles": [{"x": o.x, "y": o.y, "r": o.radius} for o in self.obstacles],
            "targets": [
                {"x": t.x, "y": t.y, "collected": t.collected}
                for t in self.targets
            ],
            "target_collections": self._target_collections,
            "arena_size": (self.config.width, self.config.height),
        }
