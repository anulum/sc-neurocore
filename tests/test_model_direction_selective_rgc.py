# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: DirectionSelectiveRGC

"""Module-specific behavioural tests for DirectionSelectiveRGC."""

from __future__ import annotations

import math

import pytest

from sc_neurocore.neurons.models.direction_selective_rgc import DirectionSelectiveRGC


def _exact_voltage(v: float, drive: float, tau: float, dt: float) -> float:
    return drive + (v - drive) * math.exp(-dt / tau)


def test_direction_selective_rgc_exact_membrane_relaxation_contract() -> None:
    """The membrane state follows exact first-order relaxation, not Euler drift."""
    cell = DirectionSelectiveRGC(tau=7.0, theta=100.0, dt=1.25, w_centre=1.4, w_surround=0.2, v=0.35)
    intensity = 2.0
    surround = 0.5
    expected_surround = 0.9 * cell._surround + 0.1 * surround
    expected_drive = cell.w_centre * (intensity - cell._prev_intensity) - cell.w_surround * expected_surround
    expected_v = _exact_voltage(cell.v, expected_drive, cell.tau, cell.dt)

    assert cell.step_rf(intensity, surround) == 0

    assert cell._prev_intensity == intensity
    assert cell._surround == pytest.approx(expected_surround)
    assert cell.v == pytest.approx(expected_v, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize(
    ("intensity", "surround"),
    [(float("nan"), 0.0), (0.0, float("inf")), (-0.1, 0.0), (0.0, -0.1)],
)
def test_direction_selective_rgc_invalid_drive_preserves_state(intensity: float, surround: float) -> None:
    """Invalid optical drive fails before mutating receptive-field state."""
    cell = DirectionSelectiveRGC.new_on()
    before = (cell.v, cell._prev_intensity, cell._surround)

    with pytest.raises(ValueError):
        cell.step_rf(intensity, surround)

    assert (cell.v, cell._prev_intensity, cell._surround) == before


def test_direction_selective_rgc_corrupted_runtime_state_preserves_state() -> None:
    """A corrupted runtime buffer cannot be amplified into partial state mutation."""
    cell = DirectionSelectiveRGC.new_on()
    cell._surround = float("nan")
    before = (cell.v, cell._prev_intensity, cell._surround)

    with pytest.raises(ValueError):
        cell.step_rf(1.0, 0.0)

    assert cell.v == before[0]
    assert cell._prev_intensity == before[1]
    assert math.isnan(cell._surround)


def test_direction_selective_rgc_spike_reset_keeps_temporal_buffers() -> None:
    """Spike reset applies only to membrane voltage after candidate acceptance."""
    cell = DirectionSelectiveRGC(theta=0.01, tau=10.0, dt=1.0)

    assert cell.step_rf(3.0, 0.0) == 1

    assert cell.v == 0.0
    assert cell._prev_intensity == 3.0
    assert cell._surround == 0.0
