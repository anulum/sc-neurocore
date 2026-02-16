"""
Swarm Fitness — Metrics for swarm performance
===============================================

Computes:
- Coverage: How much of the arena is explored
- Cohesion: How close agents stay to each other
- Alignment: How aligned agent headings are
- Target collection: How many targets are collected
- Obstacle avoidance: Penalty for being inside obstacles
- Composite fitness: Weighted sum of all metrics

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .swarm_env import SwarmEnvironment


@dataclass
class FitnessWeights:
    """Weights for composite fitness score."""
    coverage: float = 0.3
    cohesion: float = 0.2
    alignment: float = 0.1
    target_reach: float = 0.3
    obstacle_penalty: float = 0.1


class SwarmFitness:
    """Evaluates swarm performance."""

    def __init__(self, weights: Optional[FitnessWeights] = None):
        self.weights = weights or FitnessWeights()

    def coverage_score(self, env: SwarmEnvironment, grid_res: int = 20) -> float:
        """
        Fraction of grid cells visited by at least one agent.

        Higher = swarm explores more of the arena.
        """
        pos = env.get_agent_positions()
        visited = set()
        for x, y in pos:
            gx = int(np.clip(x / env.config.width * grid_res, 0, grid_res - 1))
            gy = int(np.clip(y / env.config.height * grid_res, 0, grid_res - 1))
            visited.add((gx, gy))
        return len(visited) / (grid_res * grid_res)

    def cohesion_score(self, env: SwarmEnvironment) -> float:
        """
        Inverse of mean distance from centroid (normalized).

        Higher = agents stay closer together.
        """
        pos = env.get_agent_positions()
        if len(pos) < 2:
            return 1.0
        centroid = pos.mean(axis=0)
        dists = np.sqrt(((pos - centroid) ** 2).sum(axis=1))
        mean_dist = dists.mean()
        # Normalize: 0 = spread across whole arena, 1 = all at centroid
        max_dist = np.sqrt(env.config.width ** 2 + env.config.height ** 2) / 2
        return float(1.0 - np.clip(mean_dist / max_dist, 0, 1))

    def alignment_score(self, env: SwarmEnvironment) -> float:
        """
        How aligned agent headings are (1 = all same direction).

        Uses circular mean resultant length.
        """
        headings = np.array([a.heading for a in env.agents])
        z = np.exp(1j * headings)
        return float(np.abs(z.mean()))

    def target_score(self, env: SwarmEnvironment, max_collections: int = 10) -> float:
        """
        Fraction of target collections achieved.
        """
        return float(np.clip(env._target_collections / max(max_collections, 1), 0, 1))

    def obstacle_penalty(self, env: SwarmEnvironment) -> float:
        """
        Penalty for agents being too close to obstacles.

        Returns value in [0, 1] where 0 = no penalty, 1 = all agents inside obstacles.
        """
        penalty = 0.0
        for agent in env.agents:
            for obs in env.obstacles:
                dx = agent.x - obs.x
                dy = agent.y - obs.y
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist < obs.radius * 1.5:
                    penalty += 1.0 - dist / (obs.radius * 1.5)
        max_penalty = len(env.agents) * len(env.obstacles)
        return float(np.clip(penalty / max(max_penalty, 1), 0, 1))

    def composite_fitness(self, env: SwarmEnvironment) -> float:
        """
        Weighted composite fitness score.

        Returns float in [0, 1].
        """
        w = self.weights
        cov = self.coverage_score(env)
        coh = self.cohesion_score(env)
        ali = self.alignment_score(env)
        tgt = self.target_score(env)
        obs = self.obstacle_penalty(env)

        score = (
            w.coverage * cov
            + w.cohesion * coh
            + w.alignment * ali
            + w.target_reach * tgt
            - w.obstacle_penalty * obs
        )
        return float(np.clip(score, 0, 1))

    def get_breakdown(self, env: SwarmEnvironment) -> dict:
        """Return all fitness components."""
        return {
            "coverage": self.coverage_score(env),
            "cohesion": self.cohesion_score(env),
            "alignment": self.alignment_score(env),
            "target_reach": self.target_score(env),
            "obstacle_penalty": self.obstacle_penalty(env),
            "composite": self.composite_fitness(env),
        }
