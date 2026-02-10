"""Tests for DysonSwarmNet Matrioshka brain dynamics."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.exotic.matrioshka import DysonSwarmNet


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_matrioshka_shells_initialized():
    """Shells list should match n_shells and node count."""
    net = DysonSwarmNet(n_shells=3, n_nodes_per_shell=4)
    assert len(net.shells) == 3
    assert all(shell.shape == (4,) for shell in net.shells)


def test_matrioshka_process_shape():
    """Process should return vector with node count."""
    net = DysonSwarmNet(n_shells=2, n_nodes_per_shell=3)
    out = net.process(np.ones(3))
    assert out.shape == (3,)


def test_matrioshka_process_updates_shells():
    """Nonzero input should update shells."""
    net = DysonSwarmNet(n_shells=2, n_nodes_per_shell=3)
    net.process(np.ones(3))
    assert np.any(net.shells[0] != 0.0)


def test_matrioshka_single_shell_equals_tanh():
    """Single shell should output tanh(input)."""
    net = DysonSwarmNet(n_shells=1, n_nodes_per_shell=2)
    inp = np.array([0.5, -0.5])
    out = net.process(inp)
    assert np.allclose(out, np.tanh(inp))


def test_matrioshka_zero_input_outputs_zero():
    """Zero input should yield zero output."""
    net = DysonSwarmNet(n_shells=3, n_nodes_per_shell=2)
    out = net.process(np.zeros(2))
    assert np.allclose(out, 0.0)


def test_matrioshka_energy_decay():
    """Output magnitude should decay across shells."""
    net = DysonSwarmNet(n_shells=3, n_nodes_per_shell=1)
    net.process(np.array([1.0]))
    assert abs(net.shells[2][0]) <= abs(net.shells[1][0]) + 1e-9


def test_matrioshka_deterministic():
    """Process should be deterministic for same input."""
    net = DysonSwarmNet(n_shells=2, n_nodes_per_shell=2)
    inp = np.array([0.1, 0.2])
    out1 = net.process(inp)
    out2 = net.process(inp)
    assert np.allclose(out1, out2)


def test_matrioshka_shells_updated_in_order():
    """Each shell should hold its processed data."""
    net = DysonSwarmNet(n_shells=2, n_nodes_per_shell=2)
    inp = np.array([0.3, 0.7])
    net.process(inp)
    assert np.allclose(net.shells[0], np.tanh(inp))


def test_matrioshka_output_dtype_float():
    """Output should be float dtype."""
    net = DysonSwarmNet(n_shells=2, n_nodes_per_shell=2)
    out = net.process(np.array([1.0, 1.0]))
    assert np.issubdtype(out.dtype, np.floating)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_matrioshka_perf_small():
    """Benchmark processing for a small swarm."""
    net = DysonSwarmNet(n_shells=5, n_nodes_per_shell=32)
    inp = np.random.random(32)
    start = time.perf_counter()
    _ = net.process(inp)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
