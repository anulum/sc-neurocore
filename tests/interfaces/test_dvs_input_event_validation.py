# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DVS input event validation contracts

"""Focused DVS input event validation contracts."""

from tests.interfaces.dvs_input_support import *


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
