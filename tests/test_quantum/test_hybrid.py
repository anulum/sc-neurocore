"""Tests for QuantumStochasticLayer hybrid dynamics."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.quantum.hybrid import QuantumStochasticLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_hybrid_output_shape():
    """Output should be (n_qubits, length)."""
    layer = QuantumStochasticLayer(n_qubits=3, length=16)
    inp = np.zeros((3, 8))
    out = layer.forward(inp)
    assert out.shape == (3, 16)


def test_hybrid_output_binary():
    """Output bitstreams should be binary."""
    layer = QuantumStochasticLayer(n_qubits=2, length=8)
    inp = np.zeros((2, 4))
    out = layer.forward(inp)
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_hybrid_all_zero_input_yields_ones():
    """Zero input should produce measurement probability of 1."""
    layer = QuantumStochasticLayer(n_qubits=2, length=8)
    inp = np.zeros((2, 8))
    out = layer.forward(inp)
    assert np.all(out == 1)


def test_hybrid_all_one_input_yields_zeros():
    """All-one input should produce measurement probability of 0."""
    layer = QuantumStochasticLayer(n_qubits=2, length=8)
    inp = np.ones((2, 8))
    out = layer.forward(inp)
    assert np.all(out == 0)


def test_hybrid_half_input_mean_near_half():
    """Half probability inputs should yield ~0.5 output mean."""
    np.random.seed(0)
    layer = QuantumStochasticLayer(n_qubits=1, length=200)
    inp = np.vstack([np.array([1] * 100 + [0] * 100, dtype=np.uint8)])
    out = layer.forward(inp)
    mean = out.mean()
    assert 0.35 <= mean <= 0.65


def test_hybrid_input_length_independent():
    """Input length may differ from layer length."""
    layer = QuantumStochasticLayer(n_qubits=1, length=32)
    inp = np.zeros((1, 5))
    out = layer.forward(inp)
    assert out.shape == (1, 32)


def test_hybrid_qubit_mismatch_raises():
    """Mismatched qubit count should raise a broadcasting error."""
    layer = QuantumStochasticLayer(n_qubits=2, length=8)
    inp = np.zeros((3, 4))
    with pytest.raises(ValueError):
        _ = layer.forward(inp)


def test_hybrid_deterministic_with_seed():
    """Numpy seed should make outputs deterministic."""
    layer = QuantumStochasticLayer(n_qubits=2, length=8)
    inp = np.zeros((2, 8))
    np.random.seed(42)
    out_a = layer.forward(inp)
    np.random.seed(42)
    out_b = layer.forward(inp)
    assert np.array_equal(out_a, out_b)


def test_hybrid_output_dtype():
    """Output should be uint8 dtype."""
    layer = QuantumStochasticLayer(n_qubits=1, length=8)
    inp = np.zeros((1, 8))
    out = layer.forward(inp)
    assert out.dtype == np.uint8


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_hybrid_perf_small():
    """Benchmark a small hybrid forward pass."""
    layer = QuantumStochasticLayer(n_qubits=8, length=128)
    inp = np.random.randint(0, 2, size=(8, 64), dtype=np.uint8)
    start = time.perf_counter()
    _ = layer.forward(inp)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
