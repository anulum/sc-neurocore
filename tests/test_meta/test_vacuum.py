"""Tests for VacuumNoiseSource zero-point computation."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.meta.vacuum import VacuumNoiseSource


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_vacuum_output_shape():
    """Generated bits should be (dimension, length)."""
    source = VacuumNoiseSource(dimension=3, plate_distance=1.0)
    bits = source.generate_virtual_bits(length=8)
    assert bits.shape == (3, 8)


def test_vacuum_output_binary():
    """Generated bits should be binary."""
    source = VacuumNoiseSource(dimension=2, plate_distance=1.0)
    bits = source.generate_virtual_bits(length=8)
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_vacuum_plate_distance_effect():
    """Different plate distances should alter output."""
    np.random.seed(0)
    close = VacuumNoiseSource(dimension=2, plate_distance=0.5)
    bits_close = close.generate_virtual_bits(length=100)
    np.random.seed(0)
    far = VacuumNoiseSource(dimension=2, plate_distance=5.0)
    bits_far = far.generate_virtual_bits(length=100)
    assert not np.array_equal(bits_close, bits_far)


def test_vacuum_dimension_one():
    """Dimension 1 should produce 1xL output."""
    source = VacuumNoiseSource(dimension=1, plate_distance=1.0)
    bits = source.generate_virtual_bits(length=16)
    assert bits.shape == (1, 16)


def test_vacuum_bits_dtype():
    """Output dtype should be uint8."""
    source = VacuumNoiseSource(dimension=2, plate_distance=1.0)
    bits = source.generate_virtual_bits(length=8)
    assert bits.dtype == np.uint8


def test_vacuum_repeatable_with_seed():
    """Numpy seed should make outputs deterministic."""
    source = VacuumNoiseSource(dimension=2, plate_distance=1.0)
    np.random.seed(1)
    a = source.generate_virtual_bits(length=8)
    np.random.seed(1)
    b = source.generate_virtual_bits(length=8)
    assert np.array_equal(a, b)


def test_vacuum_output_mean_reasonable():
    """Output mean should be within a reasonable probability range."""
    source = VacuumNoiseSource(dimension=2, plate_distance=1.0)
    bits = source.generate_virtual_bits(length=200)
    mean = bits.mean()
    assert 0.3 <= mean <= 0.7


def test_vacuum_handles_large_distance():
    """Large plate distance should still produce valid output."""
    source = VacuumNoiseSource(dimension=2, plate_distance=100.0)
    bits = source.generate_virtual_bits(length=16)
    assert bits.shape == (2, 16)


def test_vacuum_handles_small_distance():
    """Small plate distance should still produce valid output."""
    source = VacuumNoiseSource(dimension=2, plate_distance=0.2)
    bits = source.generate_virtual_bits(length=16)
    assert bits.shape == (2, 16)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_vacuum_perf_small():
    """Benchmark a small vacuum noise generation."""
    source = VacuumNoiseSource(dimension=8, plate_distance=1.0)
    start = time.perf_counter()
    _ = source.generate_virtual_bits(length=256)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
