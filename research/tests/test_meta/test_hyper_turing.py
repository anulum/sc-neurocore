"""Tests for OracleLayer hyper-computation utilities."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.meta.hyper_turing import OracleLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_oracle_solve_halting_true_for_constant():
    """Constant bitstream should be classified as halting."""
    oracle = OracleLayer()
    bits = np.ones(200, dtype=np.uint8)
    assert oracle.solve_halting(bits) is True


def test_oracle_solve_halting_false_for_fluctuating():
    """Fluctuating bitstream should be classified as non-halting."""
    oracle = OracleLayer()
    bits = np.tile([0, 1], 100).astype(np.uint8)
    assert oracle.solve_halting(bits) is False


def test_oracle_solve_halting_short_stream():
    """Short streams should still be evaluated."""
    oracle = OracleLayer()
    bits = np.zeros(10, dtype=np.uint8)
    assert oracle.solve_halting(bits) is True


def test_oracle_predictive_compute_shape():
    """predictive_compute should preserve current_data shape."""
    oracle = OracleLayer()
    current = np.array([1.0, 2.0])
    future = np.array([0.0, 0.0, 1.0])
    out = oracle.predictive_compute(current, future)
    assert out.shape == current.shape


def test_oracle_predictive_compute_weighting():
    """predictive_compute should average current with future mean."""
    oracle = OracleLayer()
    current = np.array([1.0, 3.0])
    future = np.array([0.0, 1.0])
    out = oracle.predictive_compute(current, future)
    expected = current * 0.5 + 0.5 * future.mean()
    assert np.allclose(out, expected)


def test_oracle_predictive_compute_float():
    """predictive_compute should return float array."""
    oracle = OracleLayer()
    out = oracle.predictive_compute(np.array([1, 2]), np.array([1, 1]))
    assert np.issubdtype(out.dtype, np.floating)


def test_oracle_halting_returns_bool():
    """solve_halting should return bool."""
    oracle = OracleLayer()
    bits = np.ones(200, dtype=np.uint8)
    assert isinstance(oracle.solve_halting(bits), bool)


def test_oracle_halting_variance_sensitive():
    """Small variance should still classify halting."""
    oracle = OracleLayer()
    bits = np.concatenate([np.zeros(150), np.zeros(50)]).astype(np.uint8)
    assert oracle.solve_halting(bits) is True


def test_oracle_predictive_compute_deterministic():
    """Predictive compute should be deterministic for same inputs."""
    oracle = OracleLayer()
    current = np.array([0.2, 0.4])
    future = np.array([0.6, 0.8])
    out1 = oracle.predictive_compute(current, future)
    out2 = oracle.predictive_compute(current, future)
    assert np.allclose(out1, out2)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_oracle_perf_small():
    """Benchmark oracle computations."""
    oracle = OracleLayer()
    bits = np.random.randint(0, 2, size=1000, dtype=np.uint8)
    current = np.random.random(100)
    future = np.random.random(200)
    start = time.perf_counter()
    _ = oracle.solve_halting(bits)
    _ = oracle.predictive_compute(current, future)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
