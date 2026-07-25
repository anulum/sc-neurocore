# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS input surface contracts

"""Focused DVS input surface contracts."""

from tests.interfaces.dvs_input_support import *


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


def test_dvs_negative_coordinates_ignored() -> None:
    """Negative coordinates should be ignored."""
    layer = DVSInputLayer(height=2, width=2)
    out = layer.process_events([(-1, -1, 1.0, 1)])
    assert np.allclose(out, 0.0)
