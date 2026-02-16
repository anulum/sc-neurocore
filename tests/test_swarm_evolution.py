"""Tests for SwarmEvolver (neuroevolution)."""

import numpy as np
import pytest

from sc_neurocore.swarm.neuroevolution_swarm import SwarmEvolver, EvolverConfig
from sc_neurocore.swarm.swarm_env import EnvConfig
from sc_neurocore.swarm.agent import AgentConfig
from sc_neurocore.swarm.fitness import SwarmFitness


class TestSwarmEvolver:
    @pytest.fixture
    def small_evolver(self):
        """Small evolver for fast tests."""
        return SwarmEvolver(EvolverConfig(
            population_size=4,
            ticks_per_eval=20,
            env_config=EnvConfig(n_agents=5, width=50, height=50, n_obstacles=1, n_targets=1),
            agent_config=AgentConfig(n_hidden=8),
            seed=42,
        ))

    def test_init(self, small_evolver):
        assert len(small_evolver.population) == 4
        assert small_evolver.generation == 0

    def test_population_weight_sizes(self, small_evolver):
        for w in small_evolver.population:
            assert len(w) == small_evolver.weight_size

    def test_evaluate_individual(self, small_evolver):
        score = small_evolver.evaluate_individual(small_evolver.population[0])
        assert 0.0 <= score <= 1.0

    def test_evolve_one_generation(self, small_evolver):
        best = small_evolver.evolve(generations=1)
        assert small_evolver.generation == 1
        assert len(best) == small_evolver.weight_size
        assert len(small_evolver.history) == 1

    def test_evolve_improves_or_stable(self, small_evolver):
        """Fitness should not decrease (elitism preserves best)."""
        small_evolver.evolve(generations=3)
        # Best fitness across generations
        fitnesses = [h["best_fitness"] for h in small_evolver.history]
        # With elitism, best should be non-decreasing
        for i in range(1, len(fitnesses)):
            assert fitnesses[i] >= fitnesses[i - 1] - 0.01  # Small tolerance

    def test_callback(self, small_evolver):
        received = []
        small_evolver.evolve(generations=2, callback=lambda g, f, info: received.append(g))
        assert received == [0, 1]

    def test_get_best_agent(self, small_evolver):
        small_evolver.evolve(generations=1)
        agent = small_evolver.get_best_agent()
        assert agent is not None
        assert len(agent.weights) == small_evolver.weight_size

    def test_crossover(self, small_evolver):
        p1 = np.zeros(small_evolver.weight_size)
        p2 = np.ones(small_evolver.weight_size)
        child = small_evolver._crossover(p1, p2)
        # Child should have mix of 0s and 1s
        assert 0 < child.sum() < small_evolver.weight_size

    def test_mutate(self, small_evolver):
        w = np.full(small_evolver.weight_size, 0.5)
        w_orig = w.copy()
        small_evolver._mutate(w)
        # Some weights should change
        assert not np.allclose(w, w_orig)
        # All should be in [0, 1]
        assert np.all(w >= 0) and np.all(w <= 1)


class TestSwarmFitness:
    def test_coverage_score(self):
        env = _make_env()
        fitness = SwarmFitness()
        score = fitness.coverage_score(env)
        assert 0.0 <= score <= 1.0

    def test_cohesion_score(self):
        env = _make_env()
        fitness = SwarmFitness()
        score = fitness.cohesion_score(env)
        assert 0.0 <= score <= 1.0

    def test_alignment_score(self):
        env = _make_env()
        fitness = SwarmFitness()
        score = fitness.alignment_score(env)
        assert 0.0 <= score <= 1.0

    def test_composite_in_range(self):
        env = _make_env()
        fitness = SwarmFitness()
        score = fitness.composite_fitness(env)
        assert 0.0 <= score <= 1.0

    def test_breakdown(self):
        env = _make_env()
        fitness = SwarmFitness()
        bd = fitness.get_breakdown(env)
        assert "coverage" in bd
        assert "composite" in bd


def _make_env():
    from sc_neurocore.swarm.swarm_env import SwarmEnvironment, EnvConfig
    return SwarmEnvironment(EnvConfig(n_agents=5, n_obstacles=2, n_targets=1, seed=42))
