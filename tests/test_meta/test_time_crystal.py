"""Tests for TimeCrystalLayer dynamics."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.meta.time_crystal import TimeCrystalLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_time_crystal_init_shapes():
    """Spins and disorder should match n_spins."""
    np.random.seed(0)
    layer = TimeCrystalLayer(n_spins=5)
    assert layer.spins.shape == (5,)
    assert layer.disorder.shape == (5,)


def test_time_crystal_drive_shape():
    """Drive should return spins of same shape."""
    layer = TimeCrystalLayer(n_spins=4)
    out = layer.drive()
    assert out.shape == (4,)


def test_time_crystal_drive_bounds():
    """Spins should remain within [-1,1] after drive."""
    layer = TimeCrystalLayer(n_spins=4)
    out = layer.drive()
    assert np.all(out >= -1.0)
    assert np.all(out <= 1.0)


def test_time_crystal_bitstream_length():
    """Bitstream length should equal cycles."""
    layer = TimeCrystalLayer(n_spins=3)
    bits = layer.get_bitstream(cycles=10)
    assert bits.shape == (10,)


def test_time_crystal_bitstream_binary():
    """Bitstream should be binary."""
    layer = TimeCrystalLayer(n_spins=3)
    bits = layer.get_bitstream(cycles=5)
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_time_crystal_drive_no_flip_changes_state():
    """Drive without flip should still update spins."""
    layer = TimeCrystalLayer(n_spins=3)
    before = layer.spins.copy()
    layer.drive(flip_pulse=False)
    assert not np.allclose(before, layer.spins)


def test_time_crystal_disorder_effect():
    """Nonzero disorder should exist when disorder_strength > 0."""
    layer = TimeCrystalLayer(n_spins=3, disorder_strength=0.8)
    assert np.any(layer.disorder != 0.0)


def test_time_crystal_seed_determinism():
    """Numpy seed should make initial spins deterministic."""
    np.random.seed(1)
    a = TimeCrystalLayer(n_spins=4)
    np.random.seed(1)
    b = TimeCrystalLayer(n_spins=4)
    assert np.allclose(a.spins, b.spins)


def test_time_crystal_drive_finite():
    """Drive output should be finite."""
    layer = TimeCrystalLayer(n_spins=4)
    out = layer.drive()
    assert np.all(np.isfinite(out))


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_time_crystal_perf_small():
    """Benchmark multiple drive cycles."""
    layer = TimeCrystalLayer(n_spins=64)
    start = time.perf_counter()
    _ = layer.get_bitstream(cycles=50)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
