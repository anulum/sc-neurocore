"""Tests for PhotonicBitstreamLayer behavior."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.optics.photonic_layer import PhotonicBitstreamLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_photonic_interference_shape_and_range():
    """Interference should be (channels, length) within [0,1]."""
    layer = PhotonicBitstreamLayer(n_channels=3)
    intensities = layer.simulate_interference(length=16)
    assert intensities.shape == (3, 16)
    assert np.all(intensities >= 0.0)
    assert np.all(intensities <= 1.0)


def test_photonic_forward_shape():
    """Forward should return (n_channels, length)."""
    layer = PhotonicBitstreamLayer(n_channels=2)
    out = layer.forward(np.array([0.2, 0.8]), length=8)
    assert out.shape == (2, 8)


def test_photonic_forward_binary():
    """Output bitstreams should be binary."""
    layer = PhotonicBitstreamLayer(n_channels=2)
    out = layer.forward(np.array([0.2, 0.8]), length=8)
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_photonic_zero_prob_outputs_zero():
    """Zero probabilities should yield all zeros."""
    layer = PhotonicBitstreamLayer(n_channels=2)
    out = layer.forward(np.array([0.0, 0.0]), length=8)
    assert np.all(out == 0)


def test_photonic_high_prob_outputs_mostly_ones():
    """High probability should yield mostly ones."""
    np.random.seed(0)
    layer = PhotonicBitstreamLayer(n_channels=1)
    out = layer.forward(np.array([0.95]), length=200)
    assert out.mean() > 0.8


def test_photonic_mid_prob_outputs_mid_mean():
    """Mid probability should yield mid-range mean."""
    np.random.seed(1)
    layer = PhotonicBitstreamLayer(n_channels=1)
    out = layer.forward(np.array([0.5]), length=200)
    assert 0.35 <= out.mean() <= 0.65


def test_photonic_input_shape_mismatch_raises():
    """Mismatch in channel count should raise a broadcasting error."""
    layer = PhotonicBitstreamLayer(n_channels=2)
    with pytest.raises(ValueError):
        _ = layer.forward(np.array([0.1]), length=8)


def test_photonic_deterministic_with_seed():
    """Numpy seed should make output deterministic."""
    layer = PhotonicBitstreamLayer(n_channels=2)
    probs = np.array([0.4, 0.6])
    np.random.seed(2)
    out_a = layer.forward(probs, length=8)
    np.random.seed(2)
    out_b = layer.forward(probs, length=8)
    assert np.array_equal(out_a, out_b)


def test_photonic_laser_power_attribute():
    """Laser power should be stored on the instance."""
    layer = PhotonicBitstreamLayer(n_channels=1, laser_power=2.5)
    assert layer.laser_power == 2.5


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_photonic_perf_small():
    """Benchmark a small photonic forward pass."""
    layer = PhotonicBitstreamLayer(n_channels=4)
    probs = np.random.random(4)
    start = time.perf_counter()
    _ = layer.forward(probs, length=128)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
