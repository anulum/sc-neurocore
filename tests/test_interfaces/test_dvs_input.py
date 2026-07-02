# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for DVSInputLayer event processing and bitstream

"""Tests for DVSInputLayer event processing and bitstream frames."""

import os
import time
from typing import Any

import numpy as np
import pytest

from sc_neurocore.interfaces.dvs_input import DVSInputLayer


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_dvs_init_surface_shape() -> None:
    """Surface should initialize to (height, width)."""
    layer = DVSInputLayer(height=4, width=5)
    assert layer.surface.shape == (4, 5)


def test_dvs_process_events_empty_returns_surface() -> None:
    """Empty event list returns current probability surface without mutation."""
    layer = DVSInputLayer(height=2, width=2)
    surface_before = layer.surface.copy()
    out = layer.process_events([])
    assert np.array_equal(out, surface_before)


def test_dvs_process_events_empty_returns_probability_copy() -> None:
    """Empty batches should not expose the mutable internal event surface."""
    layer = DVSInputLayer(height=1, width=1)
    _ = layer.process_events([(0, 0, 0.0, 1), (0, 0, 0.0, -1)])

    out = layer.process_events([])

    assert float(out[0, 0]) == pytest.approx(float(np.tanh(layer.surface[0, 0])))
    assert float(out[0, 0]) < float(layer.surface[0, 0])
    assert not np.shares_memory(out, layer.surface)


def test_dvs_process_events_output_shape_and_range() -> None:
    """Processed output should be in [0,1] and correct shape."""
    layer = DVSInputLayer(height=3, width=3)
    out = layer.process_events([(1, 1, 10.0, 1)])
    assert out.shape == (3, 3)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_dvs_events_out_of_bounds_ignored() -> None:
    """Out-of-bounds events should not change surface."""
    layer = DVSInputLayer(height=2, width=2)
    out = layer.process_events([(5, 5, 1.0, 1)])
    assert np.allclose(out, 0.0)


def test_dvs_last_update_time_updates() -> None:
    """last_update_time should update to latest event timestamp."""
    layer = DVSInputLayer(height=2, width=2)
    _ = layer.process_events([(0, 0, 5.0, 1)])
    assert layer.last_update_time == 5.0


def test_dvs_decay_applied_between_batches() -> None:
    """Surface should decay before adding new events."""
    layer = DVSInputLayer(height=1, width=1, decay_tau=10.0)
    _ = layer.process_events([(0, 0, 0.0, 1)])
    val_before = layer.surface[0, 0]
    _ = layer.process_events([(0, 0, 10.0, 1)])
    val_after = layer.surface[0, 0]
    assert val_after < val_before + 1.0


def test_dvs_generate_bitstream_frame_shape() -> None:
    """Bitstream frame should be (H, W, length)."""
    layer = DVSInputLayer(height=2, width=3)
    bits = layer.generate_bitstream_frame(length=8)
    assert bits.shape == (2, 3, 8)


def test_dvs_generate_bitstream_frame_binary() -> None:
    """Bitstream frame should be binary."""
    layer = DVSInputLayer(height=2, width=2)
    bits = layer.generate_bitstream_frame(length=4)
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_dvs_negative_coordinates_ignored() -> None:
    """Negative coordinates should be ignored."""
    layer = DVSInputLayer(height=2, width=2)
    out = layer.process_events([(-1, -1, 1.0, 1)])
    assert np.allclose(out, 0.0)


@pytest.mark.parametrize("decay_tau", [0.0, -1.0, np.inf, True, "100.0"])
def test_dvs_rejects_invalid_decay_tau(decay_tau: Any) -> None:
    """Decay time constant must be finite and positive."""
    with pytest.raises(ValueError, match="decay_tau must be finite and positive"):
        DVSInputLayer(height=2, width=2, decay_tau=decay_tau)


@pytest.mark.parametrize(
    ("height", "width"),
    [
        (0, 2),
        (2, 0),
        (-1, 2),
        (True, 2),
        (2, False),
        (1.5, 2),
    ],
)
def test_dvs_rejects_invalid_dimensions(height: Any, width: Any) -> None:
    """DVS frame dimensions must be positive integer pixel counts."""
    with pytest.raises(ValueError, match="height and width must be positive integers"):
        DVSInputLayer(height=height, width=width)


def test_dvs_rejects_invalid_polarity() -> None:
    """AER polarity must be encoded as -1, 0, or 1."""
    layer = DVSInputLayer(height=2, width=2)
    with pytest.raises(ValueError, match="polarity must be -1, 0, or 1"):
        layer.process_events([(0, 0, 1.0, 7)])


@pytest.mark.parametrize(
    "event",
    [
        (0.5, 0, 1.0, 1),
        (0, "1", 1.0, 1),
        (True, 0, 1.0, 1),
        (0, False, 1.0, 1),
    ],
)
def test_dvs_rejects_non_integer_event_addresses(event: tuple[Any, Any, float, int]) -> None:
    """AER event coordinates must be integer pixel addresses."""
    layer = DVSInputLayer(height=2, width=2)
    with pytest.raises(ValueError, match="event coordinates must be integer pixel addresses"):
        layer.process_events([event])


def test_dvs_rejects_boolean_polarity() -> None:
    """Boolean values are not accepted as AER polarity aliases."""
    layer = DVSInputLayer(height=2, width=2)
    with pytest.raises(ValueError, match="polarity must be -1, 0, or 1"):
        layer.process_events([(0, 0, 1.0, True)])


@pytest.mark.parametrize("timestamp", [np.inf, -np.inf, np.nan, True, "1.0"])
def test_dvs_rejects_invalid_timestamps(timestamp: Any) -> None:
    """AER event timestamps must be finite real scalars."""
    layer = DVSInputLayer(height=2, width=2)
    with pytest.raises(ValueError, match="event timestamp must be finite"):
        layer.process_events([(0, 0, timestamp, 1)])


def test_dvs_rejects_non_monotonic_timestamps() -> None:
    """Event batches must be timestamp ordered before decay integration."""
    layer = DVSInputLayer(height=2, width=2)
    with pytest.raises(ValueError, match="timestamps must be monotonically non-decreasing"):
        layer.process_events([(0, 0, 2.0, 1), (1, 1, 1.0, -1)])


def test_dvs_rejects_cross_batch_timestamp_rewind_without_mutation() -> None:
    """A later batch cannot move the DVS clock backwards or mutate state."""
    layer = DVSInputLayer(height=2, width=2)
    _ = layer.process_events([(0, 0, 5.0, 1)])
    surface_before = layer.surface.copy()
    last_update_before = layer.last_update_time

    with pytest.raises(ValueError, match="event timestamp cannot be earlier"):
        layer.process_events([(1, 1, 4.0, 1)])

    assert layer.last_update_time == last_update_before
    np.testing.assert_array_equal(layer.surface, surface_before)


def test_dvs_rejects_malformed_event_without_mutation() -> None:
    """Malformed event batches fail before decay or address writes occur."""
    layer = DVSInputLayer(height=2, width=2)
    _ = layer.process_events([(0, 0, 1.0, 1)])
    surface_before = layer.surface.copy()
    last_update_before = layer.last_update_time
    malformed_event: Any = (1.5, 1, 2.0, 1)

    with pytest.raises(ValueError, match="event coordinates must be integer pixel addresses"):
        layer.process_events([malformed_event])

    assert layer.last_update_time == last_update_before
    np.testing.assert_array_equal(layer.surface, surface_before)


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_dvs_rejects_invalid_bitstream_length(length: Any) -> None:
    """Generated bitstream frames require a positive integer sample length."""
    layer = DVSInputLayer(height=2, width=2)
    with pytest.raises(ValueError, match="length must be a positive integer"):
        layer.generate_bitstream_frame(length=length)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_dvs_perf_small() -> None:
    """Benchmark processing a small event batch."""
    layer = DVSInputLayer(height=32, width=32)
    events = [(i % 32, i % 32, float(i), 1) for i in range(100)]
    start = time.perf_counter()
    _ = layer.process_events(events)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
