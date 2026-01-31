"""Tests for ReactionDiffusionSolver chemical computing."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.exotic.chemical import ReactionDiffusionSolver


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_chemical_init_shapes():
    """A and B should match (height, width)."""
    solver = ReactionDiffusionSolver(width=5, height=4)
    assert solver.A.shape == (4, 5)
    assert solver.B.shape == (4, 5)


def test_chemical_seed_patch_nonzero():
    """Seeded B region should introduce nonzero values."""
    solver = ReactionDiffusionSolver(width=10, height=10)
    assert np.any(solver.B > 0.0)


def test_chemical_laplacian_constant_zero():
    """Laplacian of constant matrix should be zero."""
    solver = ReactionDiffusionSolver(width=4, height=4)
    const = np.ones((4, 4))
    lap = solver.laplacian(const)
    assert np.allclose(lap, 0.0)


def test_chemical_step_clips_range():
    """Step should keep A and B within [0,1]."""
    solver = ReactionDiffusionSolver(width=8, height=8)
    solver.step()
    assert np.all(solver.A >= 0.0)
    assert np.all(solver.A <= 1.0)
    assert np.all(solver.B >= 0.0)
    assert np.all(solver.B <= 1.0)


def test_chemical_step_changes_state():
    """Step should change B state over time."""
    solver = ReactionDiffusionSolver(width=8, height=8)
    before = solver.B.copy()
    solver.step()
    assert not np.allclose(before, solver.B)


def test_chemical_get_state_returns_B():
    """get_state should return B field."""
    solver = ReactionDiffusionSolver(width=6, height=6)
    assert np.array_equal(solver.get_state(), solver.B)


def test_chemical_dt_zero_no_change():
    """dt=0 should leave state unchanged."""
    solver = ReactionDiffusionSolver(width=6, height=6, dt=0.0)
    a_before = solver.A.copy()
    b_before = solver.B.copy()
    solver.step()
    assert np.allclose(a_before, solver.A)
    assert np.allclose(b_before, solver.B)


def test_chemical_small_grid_runs():
    """Small grids should run without errors."""
    solver = ReactionDiffusionSolver(width=3, height=3)
    solver.step()
    assert solver.B.shape == (3, 3)


def test_chemical_seed_determinism():
    """Numpy seed should make initialization deterministic."""
    np.random.seed(5)
    a = ReactionDiffusionSolver(width=10, height=10)
    np.random.seed(5)
    b = ReactionDiffusionSolver(width=10, height=10)
    assert np.allclose(a.B, b.B)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_chemical_perf_small():
    """Benchmark a single reaction-diffusion step."""
    solver = ReactionDiffusionSolver(width=32, height=32)
    start = time.perf_counter()
    solver.step()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
