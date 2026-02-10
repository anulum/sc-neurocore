"""Tests for DVSInputLayer event processing and bitstream frames."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.interfaces.dvs_input import DVSInputLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_dvs_init_surface_shape():
    """Surface should initialize to (height, width)."""
    layer = DVSInputLayer(height=4, width=5)
    assert layer.surface.shape == (4, 5)


def test_dvs_process_events_empty_returns_surface():
    """Empty event list returns current surface unchanged."""
    layer = DVSInputLayer(height=2, width=2)
    surface_before = layer.surface.copy()
    out = layer.process_events([])
    assert np.array_equal(out, surface_before)


def test_dvs_process_events_output_shape_and_range():
    """Processed output should be in [0,1] and correct shape."""
    layer = DVSInputLayer(height=3, width=3)
    out = layer.process_events([(1, 1, 10.0, 1)])
    assert out.shape == (3, 3)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_dvs_events_out_of_bounds_ignored():
    """Out-of-bounds events should not change surface."""
    layer = DVSInputLayer(height=2, width=2)
    out = layer.process_events([(5, 5, 1.0, 1)])
    assert np.allclose(out, 0.0)


def test_dvs_last_update_time_updates():
    """last_update_time should update to latest event timestamp."""
    layer = DVSInputLayer(height=2, width=2)
    _ = layer.process_events([(0, 0, 5.0, 1)])
    assert layer.last_update_time == 5.0


def test_dvs_decay_applied_between_batches():
    """Surface should decay before adding new events."""
    layer = DVSInputLayer(height=1, width=1, decay_tau=10.0)
    _ = layer.process_events([(0, 0, 0.0, 1)])
    val_before = layer.surface[0, 0]
    _ = layer.process_events([(0, 0, 10.0, 1)])
    val_after = layer.surface[0, 0]
    assert val_after < val_before + 1.0


def test_dvs_generate_bitstream_frame_shape():
    """Bitstream frame should be (H, W, length)."""
    layer = DVSInputLayer(height=2, width=3)
    bits = layer.generate_bitstream_frame(length=8)
    assert bits.shape == (2, 3, 8)


def test_dvs_generate_bitstream_frame_binary():
    """Bitstream frame should be binary."""
    layer = DVSInputLayer(height=2, width=2)
    bits = layer.generate_bitstream_frame(length=4)
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_dvs_negative_coordinates_ignored():
    """Negative coordinates should be ignored."""
    layer = DVSInputLayer(height=2, width=2)
    out = layer.process_events([(-1, -1, 1.0, 1)])
    assert np.allclose(out, 0.0)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_dvs_perf_small():
    """Benchmark processing a small event batch."""
    layer = DVSInputLayer(height=32, width=32)
    events = [(i % 32, i % 32, float(i), 1) for i in range(100)]
    start = time.perf_counter()
    _ = layer.process_events(events)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
