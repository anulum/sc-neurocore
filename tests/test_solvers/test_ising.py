"""Tests for the StochasticIsingGraph solver."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.solvers.ising import StochasticIsingGraph


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def _make_solver(num_spins=4, seed=0):
    np.random.seed(seed)
    J = np.zeros((num_spins, num_spins))
    h = np.zeros(num_spins)
    return StochasticIsingGraph(num_spins=num_spins, J=J, h=h, temperature=1.0, anneal_rate=0.9)


def test_ising_initial_shapes():
    """Spins and matrices should have correct shapes."""
    solver = _make_solver(num_spins=5)
    assert solver.spins.shape == (5,)
    assert solver.bipolar_spins.shape == (5,)


def test_ising_get_energy_zero_for_zero_J_h():
    """Energy should be zero when J and h are zero."""
    solver = _make_solver(num_spins=3)
    assert np.isclose(solver.get_energy(), 0.0)


def test_ising_step_returns_energy():
    """step should return a float energy value."""
    solver = _make_solver(num_spins=3)
    energy = solver.step()
    assert isinstance(energy, float)


def test_ising_temperature_anneals():
    """Temperature should decrease by anneal_rate each step."""
    solver = _make_solver(num_spins=3)
    t0 = solver.temperature
    _ = solver.step()
    assert np.isclose(solver.temperature, t0 * solver.anneal_rate)


def test_ising_spins_binary():
    """Spins should remain in {0,1}."""
    solver = _make_solver(num_spins=4)
    _ = solver.step()
    assert set(np.unique(solver.spins).tolist()) <= {0, 1}


def test_ising_bipolar_spins_values():
    """Bipolar spins should be in {-1,1}."""
    solver = _make_solver(num_spins=4)
    assert set(np.unique(solver.bipolar_spins).tolist()) <= {-1, 1}


def test_ising_config_length():
    """get_config should return correct length."""
    solver = _make_solver(num_spins=6)
    cfg = solver.get_config()
    assert cfg.shape == (6,)


def test_ising_determinism_with_seed():
    """Same numpy seed should yield same initial spins."""
    solver_a = _make_solver(num_spins=4, seed=10)
    solver_b = _make_solver(num_spins=4, seed=10)
    assert np.array_equal(solver_a.spins, solver_b.spins)


def test_ising_nonzero_bias_affects_energy():
    """Nonzero h should contribute to energy."""
    np.random.seed(0)
    J = np.zeros((3, 3))
    h = np.array([0.1, 0.0, -0.2])
    solver = StochasticIsingGraph(num_spins=3, J=J, h=h, temperature=1.0, anneal_rate=1.0)
    energy = solver.get_energy()
    assert energy != 0.0


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_ising_perf_small():
    """Benchmark a small number of Ising steps."""
    solver = _make_solver(num_spins=32)
    start = time.perf_counter()
    for _ in range(50):
        _ = solver.step()
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
