"""
Swarm Neuroevolution — Co-evolve agent SNN policies
=====================================================

Extends the genetic algorithm pattern for swarm co-evolution:
- All agents share weights from a single genome per individual
- Fitness is evaluated by running the swarm for N ticks
- Elite selection + crossover + mutation on flat weight vectors

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .agent import SwarmAgent, AgentConfig
from .collective_fields import CollectiveFields, FieldConfig
from .communication import SwarmCommunication
from .fitness import SwarmFitness, FitnessWeights
from .swarm_env import SwarmEnvironment, EnvConfig

logger = logging.getLogger(__name__)


@dataclass
class EvolverConfig:
    """Configuration for swarm co-evolution."""
    population_size: int = 12
    mutation_rate: float = 0.05
    mutation_std: float = 0.1
    elite_fraction: float = 0.25
    ticks_per_eval: int = 200
    env_config: Optional[EnvConfig] = None
    agent_config: Optional[AgentConfig] = None
    fitness_weights: Optional[FitnessWeights] = None
    seed: int = 42


class SwarmEvolver:
    """
    Genetic algorithm for co-evolving swarm agent SNN policies.

    Each individual in the population is a flat weight vector that
    is shared across all agents. Fitness is evaluated by running the
    full swarm simulation for ticks_per_eval steps.
    """

    def __init__(self, config: Optional[EvolverConfig] = None):
        self.config = config or EvolverConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.fitness_evaluator = SwarmFitness(self.config.fitness_weights)

        # Create a template agent to get weight vector size
        agent_cfg = self.config.agent_config or AgentConfig()
        template = SwarmAgent(0, config=agent_cfg, seed=0)
        self.weight_size = len(template.weights)

        # Initialize population
        self.population: List[np.ndarray] = [
            self.rng.uniform(0, 0.5, self.weight_size)
            for _ in range(self.config.population_size)
        ]

        self.generation = 0
        self.best_fitness = 0.0
        self.best_weights: Optional[np.ndarray] = None
        self.history: List[Dict] = []

    def evaluate_individual(self, weights: np.ndarray) -> float:
        """
        Evaluate one individual by running a full swarm simulation.

        All agents get the same weights. Returns composite fitness.
        """
        env_cfg = self.config.env_config or EnvConfig()
        # Ensure env creates agents with matching config
        env_cfg.agent_config = self.config.agent_config
        env = SwarmEnvironment(config=env_cfg)
        fields = CollectiveFields(FieldConfig(
            arena_width=env_cfg.width,
            arena_height=env_cfg.height,
        ))
        comm = SwarmCommunication(env, fields)

        # Set all agents to use these weights
        for agent in env.agents:
            agent.weights = weights.copy()
            agent.reset_neural_state()

        # Run simulation
        for _ in range(self.config.ticks_per_eval):
            # Each agent senses, thinks, acts
            for i, agent in enumerate(env.agents):
                neighbor_d = env.get_neighbor_distances(i)
                obstacle_d = env.get_obstacle_distances(i)
                target_d = env.get_target_distances(i)
                comm_data = comm.get_sensory_data(i)

                sensory = agent.sense(
                    neighbor_dists=neighbor_d,
                    obstacle_dists=obstacle_d,
                    target_dists=target_d,
                    chem_gradient=comm_data["chem_gradient"],
                    symbolic_value=comm_data["symbolic_value"],
                )
                motor = agent.think(sensory)
                agent.act(motor[0], motor[1])

            # Communication step
            comm.step()
            # Environment step
            env.step()

        return self.fitness_evaluator.composite_fitness(env)

    def evolve(self, generations: int, callback=None) -> np.ndarray:
        """
        Run genetic algorithm for N generations.

        Args:
            generations: Number of generations to evolve.
            callback: Optional callable(gen, best_fitness, breakdown) per gen.

        Returns:
            Best weight vector found.
        """
        for gen in range(generations):
            # Evaluate all individuals
            scores = [self.evaluate_individual(w) for w in self.population]

            # Rank
            ranked = np.argsort(scores)[::-1]
            best_score = scores[ranked[0]]
            best_idx = ranked[0]

            if best_score > self.best_fitness:
                self.best_fitness = best_score
                self.best_weights = self.population[best_idx].copy()

            self.history.append({
                "generation": self.generation,
                "best_fitness": float(best_score),
                "mean_fitness": float(np.mean(scores)),
                "std_fitness": float(np.std(scores)),
            })

            logger.info(
                "Gen %d: best=%.4f mean=%.4f std=%.4f",
                self.generation, best_score, np.mean(scores), np.std(scores),
            )

            if callback:
                callback(self.generation, best_score, self.history[-1])

            # Selection: keep elites
            n_elite = max(2, int(self.config.population_size * self.config.elite_fraction))
            elites = [self.population[ranked[i]].copy() for i in range(n_elite)]

            # Build next generation
            next_gen = list(elites)
            while len(next_gen) < self.config.population_size:
                # Tournament selection
                i1, i2 = self.rng.choice(n_elite, 2, replace=False)
                p1, p2 = elites[i1], elites[i2]

                # Uniform crossover
                child = self._crossover(p1, p2)
                # Gaussian mutation
                self._mutate(child)
                next_gen.append(child)

            self.population = next_gen
            self.generation += 1

        return self.best_weights if self.best_weights is not None else self.population[0]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Uniform crossover."""
        mask = self.rng.rand(self.weight_size) > 0.5
        child = np.where(mask, p1, p2)
        return child

    def _mutate(self, weights: np.ndarray):
        """Gaussian mutation with rate control."""
        mask = self.rng.rand(self.weight_size) < self.config.mutation_rate
        noise = self.rng.normal(0, self.config.mutation_std, self.weight_size)
        weights[mask] += noise[mask]
        np.clip(weights, 0, 1, out=weights)

    def get_best_agent(self) -> SwarmAgent:
        """Return a SwarmAgent loaded with the best evolved weights."""
        agent_cfg = self.config.agent_config or AgentConfig()
        agent = SwarmAgent(0, config=agent_cfg)
        if self.best_weights is not None:
            agent.weights = self.best_weights.copy()
        return agent
