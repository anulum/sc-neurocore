"""Tests for ConnectomeEmulator whole-brain emulation."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.bio.uploading import ConnectomeEmulator


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_uploading_adj_shape():
    """Adjacency should be (n_neurons, n_neurons)."""
    np.random.seed(0)
    emu = ConnectomeEmulator(n_neurons=4, sparsity=0.5)
    assert emu.adj.shape == (4, 4)


def test_uploading_spikes_shape_and_dtype():
    """Spikes should be length n_neurons and uint8."""
    emu = ConnectomeEmulator(n_neurons=3, sparsity=0.0)
    out = emu.step()
    assert out.shape == (3,)
    assert out.dtype == np.uint8


def test_uploading_zero_sparsity_all_zero_adj():
    """sparsity=0 should zero out adjacency."""
    emu = ConnectomeEmulator(n_neurons=3, sparsity=0.0)
    assert np.all(emu.adj == 0.0)


def test_uploading_step_with_zero_adj_no_spikes():
    """Zero adjacency should yield no spikes with zero currents."""
    emu = ConnectomeEmulator(n_neurons=3, sparsity=0.0)
    out = emu.step()
    assert np.all(out == 0)


def test_uploading_spikes_update_internal_state():
    """step should update internal spikes array."""
    emu = ConnectomeEmulator(n_neurons=2, sparsity=0.0)
    _ = emu.step()
    assert np.array_equal(emu.spikes, np.zeros(2, dtype=np.uint8))


def test_uploading_adjacency_sparsity_effect():
    """Higher sparsity should leave more connections."""
    np.random.seed(1)
    dense = ConnectomeEmulator(n_neurons=5, sparsity=1.0)
    np.random.seed(1)
    sparse = ConnectomeEmulator(n_neurons=5, sparsity=0.1)
    assert np.count_nonzero(dense.adj) >= np.count_nonzero(sparse.adj)


def test_uploading_neuron_count():
    """Neuron list length should match n_neurons."""
    emu = ConnectomeEmulator(n_neurons=4, sparsity=0.5)
    assert len(emu.neurons) == 4


def test_uploading_step_returns_binary():
    """Output spikes should be binary."""
    emu = ConnectomeEmulator(n_neurons=4, sparsity=0.0)
    out = emu.step()
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_uploading_deterministic_with_seed():
    """Numpy seed should make adjacency deterministic."""
    np.random.seed(2)
    a = ConnectomeEmulator(n_neurons=3, sparsity=0.3)
    np.random.seed(2)
    b = ConnectomeEmulator(n_neurons=3, sparsity=0.3)
    assert np.allclose(a.adj, b.adj)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_uploading_perf_small():
    """Benchmark a small connectome step."""
    emu = ConnectomeEmulator(n_neurons=64, sparsity=0.05)
    start = time.perf_counter()
    _ = emu.step()
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
