"""Tests for federated aggregation utilities."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.learning.federated import FederatedAggregator


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_federated_aggregate_majority():
    """Majority vote should aggregate correctly."""
    a = np.array([1, 0, 1], dtype=np.uint8)
    b = np.array([1, 1, 0], dtype=np.uint8)
    c = np.array([0, 1, 1], dtype=np.uint8)
    out = FederatedAggregator.aggregate_gradients([a, b, c])
    assert np.array_equal(out, np.array([1, 1, 1], dtype=np.uint8))


def test_federated_aggregate_empty_raises():
    """Empty gradients list should raise ValueError."""
    with pytest.raises(ValueError):
        _ = FederatedAggregator.aggregate_gradients([])


def test_federated_even_clients_threshold():
    """For two clients, aggregation acts like AND."""
    a = np.array([1, 0, 1], dtype=np.uint8)
    b = np.array([1, 1, 0], dtype=np.uint8)
    out = FederatedAggregator.aggregate_gradients([a, b])
    assert np.array_equal(out, np.array([1, 0, 0], dtype=np.uint8))


def test_federated_secure_sum_protocol():
    """secure_sum_protocol should sum gradients."""
    a = np.array([1, 0, 1], dtype=np.uint8)
    b = np.array([1, 1, 0], dtype=np.uint8)
    out = FederatedAggregator.secure_sum_protocol([a, b])
    assert np.array_equal(out, np.array([2, 1, 1], dtype=np.uint8))


def test_federated_secure_sum_empty_raises():
    """secure_sum_protocol should error on empty list."""
    with pytest.raises(ValueError):
        _ = FederatedAggregator.secure_sum_protocol([])


def test_federated_shape_preserved():
    """Aggregation should preserve gradient shape."""
    grads = [np.zeros((2, 3), dtype=np.uint8) for _ in range(3)]
    out = FederatedAggregator.aggregate_gradients(grads)
    assert out.shape == (2, 3)


def test_federated_all_zeros():
    """All-zero gradients should yield zeros."""
    grads = [np.zeros(4, dtype=np.uint8) for _ in range(3)]
    out = FederatedAggregator.aggregate_gradients(grads)
    assert np.all(out == 0)


def test_federated_all_ones():
    """All-one gradients should yield ones."""
    grads = [np.ones(5, dtype=np.uint8) for _ in range(3)]
    out = FederatedAggregator.aggregate_gradients(grads)
    assert np.all(out == 1)


def test_federated_mixed_shapes_raise():
    """Mismatched shapes should raise during stacking."""
    a = np.zeros(3, dtype=np.uint8)
    b = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        _ = FederatedAggregator.aggregate_gradients([a, b])


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_federated_perf_small():
    """Benchmark aggregating small gradients."""
    grads = [np.random.randint(0, 2, size=1000, dtype=np.uint8) for _ in range(5)]
    start = time.perf_counter()
    _ = FederatedAggregator.aggregate_gradients(grads)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
