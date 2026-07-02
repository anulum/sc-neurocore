# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SwarmEvolver -- genetic algorithm over SNN weight vectors

"""SwarmEvolver -- genetic algorithm over SNN weight vectors.

Each individual in the population is a flat weight vector that gets
injected into every agent of a SwarmEnvironment.  Fitness is evaluated
by running the swarm for *n_eval_steps* and computing the composite
fitness score.

Selection uses truncation (elite) selection.  Offspring are produced by
uniform crossover of two random elite parents plus Gaussian mutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np

from .agent import AgentConfig, SwarmAgent, _ensure_seed, _validate_weight_vector
from .swarm_env import EnvConfig, SwarmEnvironment
from .collective_fields import FieldConfig, CollectiveFields
from .fitness import SwarmFitness


@dataclass
class EvolverConfig:
    """Neuroevolution hyper-parameters for homogeneous swarm agents."""

    pop_size: int = 20
    n_elite: int = 4
    mutation_rate: float = 0.1
    mutation_std: float = 0.3
    n_eval_steps: int = 200
    use_fields: bool = False
    env_config: Optional[EnvConfig] = None
    agent_config: Optional[AgentConfig] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate population, selection, mutation, evaluation, and seed domains."""
        if (
            not isinstance(self.pop_size, int)
            or isinstance(self.pop_size, bool)
            or self.pop_size < 2
        ):
            raise ValueError("pop_size must be an integer >= 2")
        if not isinstance(self.n_elite, int) or isinstance(self.n_elite, bool):
            raise ValueError("n_elite must be an integer")
        if self.n_elite == 4 and self.pop_size < 4:
            self.n_elite = self.pop_size
        if self.n_elite < 1 or self.n_elite > self.pop_size:
            raise ValueError("n_elite must be in [1, pop_size]")
        self.mutation_rate = float(self.mutation_rate)
        if not math.isfinite(self.mutation_rate) or not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be finite and in [0, 1]")
        self.mutation_std = float(self.mutation_std)
        if not math.isfinite(self.mutation_std) or self.mutation_std < 0.0:
            raise ValueError("mutation_std must be finite and non-negative")
        if (
            not isinstance(self.n_eval_steps, int)
            or isinstance(self.n_eval_steps, bool)
            or self.n_eval_steps <= 0
        ):
            raise ValueError("n_eval_steps must be a positive integer")
        self.seed = _ensure_seed(self.seed, "seed")


class SwarmEvolver:
    """Genetic algorithm that evolves SNN weights for swarm control.

    Parameters
    ----------
    cfg : EvolverConfig
        Evolution and evaluation parameters.
    """

    def __init__(self, cfg: EvolverConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.agent_config = cfg.agent_config or AgentConfig()

        # Determine weight vector size from a template agent
        template = SwarmAgent(self.agent_config, agent_id=0)
        self.n_weights = template.n_weights

        # Initialise population with small random weights
        self.population = [self.rng.normal(0, 0.5, self.n_weights) for _ in range(cfg.pop_size)]
        self.fitnesses = np.zeros(cfg.pop_size)
        self.generation = 0
        self.best_fitness_history: list[float] = []

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _make_env(self) -> SwarmEnvironment:
        """Build a fresh environment with the correct agent config."""
        env_cfg = self.cfg.env_config or EnvConfig()
        # Ensure the environment uses our agent_config so weight sizes match
        env_cfg = EnvConfig(
            width=env_cfg.width,
            height=env_cfg.height,
            n_agents=env_cfg.n_agents,
            n_obstacles=env_cfg.n_obstacles,
            n_targets=env_cfg.n_targets,
            boundary_mode=env_cfg.boundary_mode,
            capture_radius=env_cfg.capture_radius,
            respawn_targets=env_cfg.respawn_targets,
            agent_config=self.agent_config,
            seed=int(self.rng.integers(0, 2**31)),
        )
        return SwarmEnvironment(env_cfg)

    def evaluate_individual(self, weights: np.ndarray[Any, Any]) -> float:
        """Create environment, inject *weights* into every agent, run, score.

        Parameters
        ----------
        weights : ndarray, shape (n_weights,)

        Returns
        -------
        fitness : float
        """
        validated_weights = _validate_weight_vector(weights, self.n_weights)
        env = self._make_env()

        # Inject same weights into all agents (homogeneous swarm)
        for agent in env.agents:
            agent.weights = validated_weights

        fields: CollectiveFields | None = None
        if self.cfg.use_fields:
            fields = CollectiveFields(
                FieldConfig(),
                env_width=env.cfg.width,
                env_height=env.cfg.height,
                n_agents=env.cfg.n_agents,
            )

        for _ in range(self.cfg.n_eval_steps):
            env.step(dt=1.0, fields=fields)

        return SwarmFitness.composite(env)

    # ------------------------------------------------------------------
    # Selection & reproduction
    # ------------------------------------------------------------------

    def _select_elite(self) -> list[np.ndarray[Any, Any]]:
        """Return the top-N weight vectors by fitness."""
        order = np.argsort(self.fitnesses)[::-1]
        return [self.population[i].copy() for i in order[: self.cfg.n_elite]]

    def _crossover(
        self, parent_a: np.ndarray[Any, Any], parent_b: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Uniform crossover: each gene randomly from either parent."""
        validated_parent_a = _validate_weight_vector(parent_a, self.n_weights)
        validated_parent_b = _validate_weight_vector(parent_b, self.n_weights)
        mask = self.rng.random(self.n_weights) < 0.5
        child = np.where(mask, validated_parent_a, validated_parent_b)
        return child

    def _mutate(self, individual: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Gaussian mutation applied to a random subset of genes."""
        candidate = _validate_weight_vector(individual, self.n_weights)
        mask = self.rng.random(self.n_weights) < self.cfg.mutation_rate
        noise = self.rng.normal(0, self.cfg.mutation_std, self.n_weights)
        candidate[mask] += noise[mask]
        if not np.all(np.isfinite(candidate)):
            raise ValueError("mutation produced non-finite weights")
        return candidate

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def evolve_generation(self) -> float:
        """Evaluate population, select, reproduce.  Return best fitness."""
        # Evaluate
        for i, w in enumerate(self.population):
            self.fitnesses[i] = self.evaluate_individual(w)

        best = float(self.fitnesses.max())
        self.best_fitness_history.append(best)

        # Select elite
        elite = self._select_elite()

        # Build next generation
        new_pop: list[np.ndarray[Any, Any]] = list(elite)  # elite survive unchanged
        while len(new_pop) < self.cfg.pop_size:
            pa = elite[self.rng.integers(0, len(elite))]
            pb = elite[self.rng.integers(0, len(elite))]
            child = self._crossover(pa, pb)
            child = self._mutate(child)
            new_pop.append(child)

        self.population = new_pop
        self.generation += 1
        return best

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_best_weights(self) -> np.ndarray[Any, Any]:
        """Return the weight vector with the highest fitness."""
        idx = int(np.argmax(self.fitnesses))
        return self.population[idx].copy()

    def run(self, n_generations: int) -> list[float]:
        """Run *n_generations* of evolution.  Return list of best fitnesses."""
        if (
            not isinstance(n_generations, int)
            or isinstance(n_generations, bool)
            or n_generations < 0
        ):
            raise ValueError("n_generations must be a non-negative integer")
        for _ in range(n_generations):
            self.evolve_generation()
        return list(self.best_fitness_history)
