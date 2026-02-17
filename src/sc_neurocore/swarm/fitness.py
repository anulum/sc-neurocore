"""
SwarmFitness -- multi-objective fitness evaluation for swarm behaviour.

All methods are static so the class acts as a namespace.  ``composite()``
combines the individual scores with fixed weights into a single scalar
suitable for neuroevolution ranking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .swarm_env import SwarmEnvironment


class SwarmFitness:
    """Static fitness functions for swarm evaluation."""

    # ------------------------------------------------------------------
    # Individual objectives
    # ------------------------------------------------------------------

    @staticmethod
    def coverage_score(positions: np.ndarray, area: tuple[float, float]) -> float:
        """Fraction of the arena covered by the swarm.

        Divides the arena into a 10x10 grid and counts the fraction of
        cells that contain at least one agent.
        """
        grid_n = 10
        w, h = area
        cols = np.clip((positions[:, 0] / w * grid_n).astype(int), 0, grid_n - 1)
        rows = np.clip((positions[:, 1] / h * grid_n).astype(int), 0, grid_n - 1)
        occupied = set(zip(rows.tolist(), cols.tolist()))
        return len(occupied) / (grid_n * grid_n)

    @staticmethod
    def cohesion_score(positions: np.ndarray) -> float:
        """Reward moderate inter-agent distance (not too spread, not too clumped).

        Returns a value in [0, 1] peaking when the mean pairwise distance
        equals one-quarter of the bounding-box diagonal.
        """
        if len(positions) < 2:
            return 0.0
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt((diff**2).sum(axis=-1))
        # Upper triangle only
        triu_idx = np.triu_indices(len(positions), k=1)
        mean_dist = dists[triu_idx].mean()
        bbox_diag = np.sqrt((positions[:, 0].ptp()) ** 2 + (positions[:, 1].ptp()) ** 2) + 1e-12
        ideal = bbox_diag * 0.25
        return float(np.exp(-(((mean_dist - ideal) / ideal) ** 2)))

    @staticmethod
    def alignment_score(headings: np.ndarray) -> float:
        """Mean resultant length of heading angles (Rayleigh statistic).

        Returns 1.0 when all agents face the same direction, 0.0 when
        headings are uniformly distributed.
        """
        if len(headings) == 0:
            return 0.0
        cx = np.cos(headings).mean()
        cy = np.sin(headings).mean()
        return float(np.sqrt(cx**2 + cy**2))

    @staticmethod
    def target_score(positions: np.ndarray, targets: np.ndarray) -> float:
        """Proximity reward: inverse mean distance to nearest target per agent.

        Normalised to [0, 1] via ``1 / (1 + mean_dist / 10)``.
        """
        if len(targets) == 0:
            return 0.0
        # (n_agents, n_targets)
        diff = positions[:, np.newaxis, :] - targets[np.newaxis, :, :]
        dists = np.sqrt((diff**2).sum(axis=-1))
        nearest = dists.min(axis=1)
        mean_nearest = nearest.mean()
        return float(1.0 / (1.0 + mean_nearest / 10.0))

    @staticmethod
    def obstacle_penalty(positions: np.ndarray, obstacles: np.ndarray) -> float:
        """Fraction of agents inside any obstacle (surface penetration)."""
        if len(obstacles) == 0:
            return 0.0
        centers = obstacles[:, :2]
        radii = obstacles[:, 2]
        # (n_agents, n_obstacles)
        diff = positions[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt((diff**2).sum(axis=-1))
        inside = (dists < radii[np.newaxis, :]).any(axis=1)
        return float(inside.mean())

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    @staticmethod
    def composite(env: "SwarmEnvironment") -> float:
        """Weighted sum of all objectives.

        Weights::

            0.30 * coverage
          + 0.20 * cohesion
          + 0.10 * alignment
          + 0.30 * target
          - 0.10 * obstacle_penalty

        Returns a scalar (higher is better).
        """
        positions = env.get_positions()
        headings = env.get_headings()
        area = (env.cfg.width, env.cfg.height)

        cov = SwarmFitness.coverage_score(positions, area)
        coh = SwarmFitness.cohesion_score(positions)
        aln = SwarmFitness.alignment_score(headings)
        tgt = SwarmFitness.target_score(positions, env.targets)
        obs = SwarmFitness.obstacle_penalty(positions, env.obstacles)

        return 0.30 * cov + 0.20 * coh + 0.10 * aln + 0.30 * tgt - 0.10 * obs
