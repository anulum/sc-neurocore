"""
SwarmEvolver -- genetic algorithm over SNN weight vectors.

Each individual in the population is a flat weight vector that gets
injected into every agent of a SwarmEnvironment.  Fitness is evaluated
by running the swarm for *n_eval_steps* and computing the composite
fitness score.

Selection uses truncation (elite) selection.  Offspring are produced by
uniform crossover of two random elite parents plus Gaussian mutation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .agent import AgentConfig, SwarmAgent
from .swarm_env import EnvConfig, SwarmEnvironment
from .collective_fields import FieldConfig, CollectiveFields
from .fitness import SwarmFitness


@dataclass
class EvolverConfig:
    """Neuroevolution hyper-parameters."""

    pop_size: int = 20
    n_elite: int = 4
    mutation_rate: float = 0.1
    mutation_std: float = 0.3
    n_eval_steps: int = 200
    use_fields: bool = False
    env_config: Optional[EnvConfig] = None
    agent_config: Optional[AgentConfig] = None
    seed: Optional[int] = None


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

    def evaluate_individual(self, weights: np.ndarray) -> float:
        """Create environment, inject *weights* into every agent, run, score.

        Parameters
        ----------
        weights : ndarray, shape (n_weights,)

        Returns
        -------
        fitness : float
        """
        env = self._make_env()

        # Inject same weights into all agents (homogeneous swarm)
        for agent in env.agents:
            agent.weights = weights

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

    def _select_elite(self) -> list[np.ndarray]:
        """Return the top-N weight vectors by fitness."""
        order = np.argsort(self.fitnesses)[::-1]
        return [self.population[i].copy() for i in order[: self.cfg.n_elite]]

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        """Uniform crossover: each gene randomly from either parent."""
        mask = self.rng.random(self.n_weights) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation applied to a random subset of genes."""
        mask = self.rng.random(self.n_weights) < self.cfg.mutation_rate
        noise = self.rng.normal(0, self.cfg.mutation_std, self.n_weights)
        individual[mask] += noise[mask]
        return individual

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
        new_pop: list[np.ndarray] = list(elite)  # elite survive unchanged
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

    def get_best_weights(self) -> np.ndarray:
        """Return the weight vector with the highest fitness."""
        idx = int(np.argmax(self.fitnesses))
        return self.population[idx].copy()

    def run(self, n_generations: int) -> list[float]:
        """Run *n_generations* of evolution.  Return list of best fitnesses."""
        for _ in range(n_generations):
            self.evolve_generation()
        return list(self.best_fitness_history)
