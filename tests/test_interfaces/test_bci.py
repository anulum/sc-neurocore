"""Tests for BCIDecoder normalization and bitstream encoding."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.interfaces.bci import BCIDecoder


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_bci_normalize_scales_to_unit():
    """normalize_signal should scale min to 0 and max to 1."""
    decoder = BCIDecoder(channels=3)
    signal = np.array([0.0, 2.0, 4.0])
    norm = decoder.normalize_signal(signal)
    assert np.isclose(norm.min(), 0.0)
    assert np.isclose(norm.max(), 1.0)


def test_bci_normalize_constant_returns_zero():
    """Constant signals should normalize to zeros."""
    decoder = BCIDecoder(channels=2)
    signal = np.array([5.0, 5.0])
    norm = decoder.normalize_signal(signal)
    assert np.allclose(norm, 0.0)


def test_bci_encode_bitstream_shape_2d():
    """2D signals should encode to (channels, length)."""
    decoder = BCIDecoder(channels=2)
    signal = np.array([[1.0, 2.0], [2.0, 3.0]])
    bits = decoder.encode_to_bitstream(signal, length=16)
    assert bits.shape == (2, 16)


def test_bci_encode_bitstream_shape_1d():
    """1D signals should encode to (channels, length)."""
    decoder = BCIDecoder(channels=3)
    signal = np.array([0.1, 0.2, 0.3])
    bits = decoder.encode_to_bitstream(signal, length=8)
    assert bits.shape == (3, 8)


def test_bci_encode_bitstream_binary():
    """Encoded bitstreams should be binary."""
    decoder = BCIDecoder(channels=2)
    signal = np.array([0.2, 0.8])
    bits = decoder.encode_to_bitstream(signal, length=8)
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_bci_encode_length_mismatch_raises():
    """Mismatch between channels and signal length should raise."""
    decoder = BCIDecoder(channels=2)
    signal = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        _ = decoder.encode_to_bitstream(signal, length=8)


def test_bci_negative_signal_normalizes():
    """Negative signals should normalize into [0,1]."""
    decoder = BCIDecoder(channels=2)
    signal = np.array([-1.0, 1.0])
    norm = decoder.normalize_signal(signal)
    assert np.all(norm >= 0.0)
    assert np.all(norm <= 1.0)


def test_bci_zero_signal_yields_zero_bits():
    """All-zero input should produce all-zero bitstream."""
    decoder = BCIDecoder(channels=2)
    signal = np.zeros(2)
    bits = decoder.encode_to_bitstream(signal, length=16)
    assert np.all(bits == 0)


def test_bci_deterministic_with_seed():
    """Setting numpy seed should make encoding deterministic."""
    decoder = BCIDecoder(channels=2)
    signal = np.array([0.4, 0.6])
    np.random.seed(42)
    bits_a = decoder.encode_to_bitstream(signal, length=8)
    np.random.seed(42)
    bits_b = decoder.encode_to_bitstream(signal, length=8)
    assert np.array_equal(bits_a, bits_b)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_bci_perf_small():
    """Benchmark encoding a short signal window."""
    decoder = BCIDecoder(channels=8)
    signal = np.random.random((8, 32))
    start = time.perf_counter()
    _ = decoder.encode_to_bitstream(signal, length=64)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
