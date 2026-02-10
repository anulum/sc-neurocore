"""Tests for AnyonBraidLayer topology mechanics."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.exotic.anyon import AnyonBraidLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_anyon_state_length_minimum():
    """State vector should have at least length 1."""
    layer = AnyonBraidLayer(n_anyons=2)
    assert layer.state.size >= 1


def test_anyon_R_matrix_shape():
    """R-matrix should be 2x2."""
    layer = AnyonBraidLayer(n_anyons=4)
    assert layer.R.shape == (2, 2)


def test_anyon_measure_normalized():
    """Measurement probabilities should sum to 1."""
    layer = AnyonBraidLayer(n_anyons=4)
    probs = layer.measure()
    assert np.isclose(np.sum(probs), 1.0)


def test_anyon_measure_nonnegative():
    """Measurement probabilities should be non-negative."""
    layer = AnyonBraidLayer(n_anyons=4)
    probs = layer.measure()
    assert np.all(probs >= 0.0)


def test_anyon_braid_updates_state():
    """Braid should modify state when length >= 2."""
    layer = AnyonBraidLayer(n_anyons=4)
    before = layer.state.copy()
    layer.braid(0)
    assert not np.allclose(before, layer.state)


def test_anyon_braid_noop_for_short_state():
    """Braid should not change state if state length < 2."""
    layer = AnyonBraidLayer(n_anyons=2)
    before = layer.state.copy()
    layer.braid(0)
    assert np.allclose(before, layer.state)


def test_anyon_state_dtype_complex():
    """State should be complex dtype."""
    layer = AnyonBraidLayer(n_anyons=4)
    assert np.issubdtype(layer.state.dtype, np.complexfloating)


def test_anyon_braid_preserves_norm():
    """Braid should preserve state norm in 2D subspace."""
    layer = AnyonBraidLayer(n_anyons=4)
    norm_before = np.linalg.norm(layer.state[:2])
    layer.braid(0)
    norm_after = np.linalg.norm(layer.state[:2])
    assert np.isclose(norm_before, norm_after)


def test_anyon_measure_stable_after_braid():
    """Measure should still normalize after braiding."""
    layer = AnyonBraidLayer(n_anyons=4)
    layer.braid(0)
    probs = layer.measure()
    assert np.isclose(np.sum(probs), 1.0)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_anyon_perf_small():
    """Benchmark braiding and measurement."""
    layer = AnyonBraidLayer(n_anyons=6)
    start = time.perf_counter()
    for _ in range(100):
        layer.braid(0)
        _ = layer.measure()
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
