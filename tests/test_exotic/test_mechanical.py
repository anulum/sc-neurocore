"""Tests for MechanicalLatticeLayer dynamics and training."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.exotic.mechanical import MechanicalLatticeLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_mechanical_K_shape_and_diag_zero():
    """K should be NxN with zero diagonal."""
    np.random.seed(0)
    layer = MechanicalLatticeLayer(n_nodes=4)
    assert layer.K.shape == (4, 4)
    assert np.allclose(np.diag(layer.K), 0.0)


def test_mechanical_relax_updates_state():
    """Relax should update non-clamped nodes."""
    layer = MechanicalLatticeLayer(n_nodes=3)
    inputs = np.array([1.0, 0.0, 0.0])
    before = layer.x.copy()
    layer.relax(inputs, clamped_nodes=[])
    assert not np.allclose(before, layer.x)


def test_mechanical_relax_respects_clamped_nodes():
    """Clamped nodes should remain unchanged."""
    layer = MechanicalLatticeLayer(n_nodes=3)
    layer.x[:] = 0.5
    layer.relax(np.ones(3), clamped_nodes=[1])
    assert layer.x[1] == 0.5


def test_mechanical_relax_zero_forces():
    """Zero forces should keep displacements at zero."""
    layer = MechanicalLatticeLayer(n_nodes=3)
    layer.relax(np.zeros(3), clamped_nodes=[])
    assert np.allclose(layer.x, 0.0)


def test_mechanical_train_updates_K():
    """Training should update stiffness values."""
    layer = MechanicalLatticeLayer(n_nodes=3, learning_rate=0.1)
    layer.x[:] = np.array([0.0, 1.0, 2.0])
    before = layer.K.copy()
    layer.train()
    assert not np.allclose(before, layer.K)


def test_mechanical_train_clips_K():
    """K values should be clipped to [0.1, 2.0] (excluding diagonal which is always 0)."""
    layer = MechanicalLatticeLayer(n_nodes=3, learning_rate=10.0)
    layer.x[:] = np.array([0.0, 5.0, -5.0])
    layer.train()
    # Check off-diagonal elements are clipped to [0.1, 2.0]
    off_diag_mask = ~np.eye(3, dtype=bool)
    off_diag_values = layer.K[off_diag_mask]
    assert np.all(off_diag_values >= 0.1)
    assert np.all(off_diag_values <= 2.0)
    # Diagonal should always be 0
    assert np.allclose(np.diag(layer.K), 0.0)


def test_mechanical_diag_zero_after_train():
    """Diagonal should remain zero after training."""
    layer = MechanicalLatticeLayer(n_nodes=3)
    layer.x[:] = np.array([0.1, 0.2, 0.3])
    layer.train()
    assert np.allclose(np.diag(layer.K), 0.0)


def test_mechanical_zero_k_sum_skips_update():
    """Nodes with zero stiffness should be skipped in relax."""
    layer = MechanicalLatticeLayer(n_nodes=2)
    layer.K[0, :] = 0.0
    layer.x[:] = 0.0
    layer.relax(np.ones(2), clamped_nodes=[])
    assert layer.x[0] == 0.0


def test_mechanical_seed_determinism():
    """Numpy seed should make K deterministic."""
    np.random.seed(7)
    a = MechanicalLatticeLayer(n_nodes=3)
    np.random.seed(7)
    b = MechanicalLatticeLayer(n_nodes=3)
    assert np.allclose(a.K, b.K)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_mechanical_perf_small():
    """Benchmark a small relaxation and training cycle."""
    layer = MechanicalLatticeLayer(n_nodes=16)
    inputs = np.random.random(16)
    start = time.perf_counter()
    layer.relax(inputs, clamped_nodes=[])
    layer.train()
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
