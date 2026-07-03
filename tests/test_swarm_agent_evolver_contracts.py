# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Swarm agent and evolver contract tests

"""Public contract tests for swarm agents and neuroevolution."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
import pytest

from sc_neurocore.swarm.agent import AgentConfig, SwarmAgent
from sc_neurocore.swarm.neuroevolution_swarm import EvolverConfig, SwarmEvolver
from sc_neurocore.swarm.swarm_env import EnvConfig


FloatArray = NDArray[np.float64]


def _sensory_vector(agent: SwarmAgent, *, chemical: float = 0.0) -> FloatArray:
    sensory = np.zeros(agent.cfg.n_sensory, dtype=np.float64)
    sensory[-1] = chemical
    return sensory


def test_agent_config_rejects_invalid_runtime_domains() -> None:
    with pytest.raises(ValueError, match="n_sensory"):
        AgentConfig(n_sensory=0)
    with pytest.raises(ValueError, match="membrane_decay"):
        AgentConfig(membrane_decay=1.0)
    with pytest.raises(ValueError, match="max_speed"):
        AgentConfig(max_speed=float("nan"))
    with pytest.raises(ValueError, match="seed"):
        AgentConfig(seed=-1)


def test_agent_rejects_invalid_identifier() -> None:
    with pytest.raises(ValueError, match="agent_id"):
        SwarmAgent(AgentConfig(n_hidden=4), agent_id=-1)


def test_agent_think_rejects_malformed_sensory_without_mutating_state() -> None:
    agent = SwarmAgent(AgentConfig(n_hidden=4, seed=1))
    membrane_before = agent.membrane.copy()
    rates_before = agent.firing_rate.copy()
    chemical_before = agent.chemical_output
    bad_sensory = _sensory_vector(agent)
    bad_sensory[3] = np.nan

    with pytest.raises(ValueError, match="sensory"):
        agent.think(bad_sensory)

    np.testing.assert_array_equal(agent.membrane, membrane_before)
    np.testing.assert_array_equal(agent.firing_rate, rates_before)
    assert agent.chemical_output == chemical_before


def test_agent_think_rejects_wrong_size_and_non_numeric_sensory() -> None:
    agent = SwarmAgent(AgentConfig(n_hidden=4, seed=2))

    with pytest.raises(ValueError, match="sensory"):
        agent.think(np.zeros(agent.cfg.n_sensory - 1, dtype=np.float64))
    with pytest.raises(ValueError, match="sensory"):
        agent.think(cast(NDArray[np.float64], np.array([object()] * agent.cfg.n_sensory)))


def test_agent_think_rejects_non_finite_candidate_state_without_mutating() -> None:
    agent = SwarmAgent(AgentConfig(n_hidden=4, seed=2))
    agent.weights = np.full(agent.n_weights, np.finfo(np.float64).max, dtype=np.float64)
    membrane_before = agent.membrane.copy()
    sensory = _sensory_vector(agent)
    sensory[0] = 2.0

    with pytest.raises(ValueError, match="non-finite"):
        agent.think(sensory)

    np.testing.assert_array_equal(agent.membrane, membrane_before)


def test_agent_think_uses_full_sensory_vector_and_clamps_chemical_output() -> None:
    agent = SwarmAgent(AgentConfig(n_hidden=4, max_speed=1.5, seed=2))
    speed, turn = agent.think(_sensory_vector(agent, chemical=2.0))

    assert 0.0 <= speed <= 1.5
    assert -np.pi <= turn <= np.pi
    assert agent.chemical_output == 1.0


def test_agent_weights_reject_malformed_vectors_without_mutating() -> None:
    agent = SwarmAgent(AgentConfig(n_hidden=4, seed=3))
    original = agent.weights.copy()

    with pytest.raises(ValueError, match="one-dimensional"):
        agent.weights = np.zeros((1, agent.n_weights), dtype=np.float64)
    with pytest.raises(ValueError, match=f"Expected {agent.n_weights} weights"):
        agent.weights = np.zeros(agent.n_weights - 1, dtype=np.float64)
    with pytest.raises(ValueError, match="weights"):
        agent.weights = cast(NDArray[np.float64], np.array([object()] * agent.n_weights))
    with pytest.raises(ValueError, match="finite"):
        mutated = original.copy()
        mutated[0] = np.inf
        agent.weights = mutated

    np.testing.assert_array_equal(agent.weights, original)


def test_agent_reset_rejects_invalid_arena_without_mutating_state() -> None:
    agent = SwarmAgent(AgentConfig(n_hidden=4, seed=4))
    agent.reset(np.random.default_rng(10), width=10.0, height=5.0)
    position_before = agent.position.copy()
    heading_before = agent.heading

    with pytest.raises(ValueError, match="width"):
        agent.reset(np.random.default_rng(11), width=0.0, height=5.0)
    with pytest.raises(ValueError, match="height"):
        agent.reset(np.random.default_rng(11), width=10.0, height=float("nan"))
    with pytest.raises(ValueError, match="rng"):
        agent.reset(cast(np.random.Generator, object()), width=10.0, height=5.0)

    np.testing.assert_array_equal(agent.position, position_before)
    assert agent.heading == heading_before


def test_agent_reset_default_rng_preserves_weight_vector() -> None:
    agent = SwarmAgent(AgentConfig(n_hidden=4, seed=4))
    weights_before = agent.weights.copy()

    agent.reset(width=10.0, height=5.0)

    assert agent.position.shape == (2,)
    assert 0.0 <= agent.position[0] <= 10.0
    assert 0.0 <= agent.position[1] <= 5.0
    assert 0.0 <= agent.heading <= 2 * np.pi
    np.testing.assert_array_equal(agent.weights, weights_before)


def test_evolver_config_rejects_invalid_domains() -> None:
    with pytest.raises(ValueError, match="pop_size"):
        EvolverConfig(pop_size=0)
    with pytest.raises(ValueError, match="n_elite"):
        EvolverConfig(pop_size=4, n_elite=0)
    with pytest.raises(ValueError, match="n_elite"):
        EvolverConfig(pop_size=4, n_elite=5)
    with pytest.raises(ValueError, match="n_elite"):
        EvolverConfig(pop_size=4, n_elite=cast(int, True))
    with pytest.raises(ValueError, match="mutation_rate"):
        EvolverConfig(mutation_rate=1.5)
    with pytest.raises(ValueError, match="mutation_std"):
        EvolverConfig(mutation_std=float("nan"))
    with pytest.raises(ValueError, match="n_eval_steps"):
        EvolverConfig(n_eval_steps=0)

    assert EvolverConfig(pop_size=3).n_elite == 3


def test_evolver_rejects_malformed_individual_weights() -> None:
    cfg = EvolverConfig(
        pop_size=4,
        n_elite=2,
        n_eval_steps=2,
        agent_config=AgentConfig(n_hidden=3, seed=5),
        env_config=EnvConfig(n_agents=2, n_obstacles=0, n_targets=0, seed=6),
        seed=7,
    )
    evolver = SwarmEvolver(cfg)
    malformed = evolver.population[0].copy()
    malformed[0] = np.nan

    with pytest.raises(ValueError, match="weights"):
        evolver.evaluate_individual(malformed)


def test_evolver_generation_rejects_non_finite_mutation_result() -> None:
    cfg = EvolverConfig(
        pop_size=2,
        n_elite=1,
        mutation_rate=1.0,
        mutation_std=float(np.finfo(np.float64).max),
        n_eval_steps=1,
        agent_config=AgentConfig(n_hidden=3, seed=6),
        env_config=EnvConfig(n_agents=1, n_obstacles=0, n_targets=0, seed=7),
        seed=0,
    )
    evolver = SwarmEvolver(cfg)

    with pytest.raises(ValueError, match="non-finite weights"):
        evolver.evolve_generation()


def test_evolver_runs_with_fields_and_returns_best_weight_copy() -> None:
    cfg = EvolverConfig(
        pop_size=4,
        n_elite=2,
        mutation_rate=0.25,
        mutation_std=0.05,
        n_eval_steps=2,
        use_fields=True,
        agent_config=AgentConfig(n_hidden=3, seed=8),
        env_config=EnvConfig(n_agents=2, n_obstacles=0, n_targets=1, seed=9),
        seed=10,
    )
    evolver = SwarmEvolver(cfg)

    history = evolver.run(1)
    best = evolver.get_best_weights()
    best[0] += 1.0

    assert len(history) == 1
    assert evolver.generation == 1
    assert np.isfinite(history[0])
    assert not np.array_equal(best, evolver.get_best_weights())


def test_evolver_run_rejects_negative_generation_count() -> None:
    cfg = EvolverConfig(
        pop_size=4,
        n_elite=2,
        n_eval_steps=1,
        agent_config=AgentConfig(n_hidden=3),
    )
    evolver = SwarmEvolver(cfg)

    with pytest.raises(ValueError, match="n_generations"):
        evolver.run(-1)
