# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (spike_events) from former test_model_resonate_and_fire.py

from __future__ import annotations

from tests.model_resonate_and_fire_support import *  # noqa: F403

def test_spike_is_upward_y_crossing_and_installs_source_reset() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.0,
        y=0.99,
        b=0.0,
        omega=1.0,
        threshold=1.0,
        dt=0.1,
    )
    assert neuron.step(10.0) == 1
    assert (neuron.x, neuron.y) == (0.0, 1.0)


def test_source_reset_at_threshold_does_not_immediately_retrigger() -> None:
    neuron = ResonateAndFireNeuron(
        x=0.0,
        y=0.99,
        b=0.0,
        omega=1.0,
        threshold=1.0,
        dt=0.1,
    )
    assert neuron.step(10.0) == 1
    assert neuron.step(0.0) == 0
    assert neuron.y < neuron.threshold


def test_radius_above_threshold_is_not_itself_an_event() -> None:
    neuron = ResonateAndFireNeuron(
        x=2.0,
        y=0.0,
        b=-1.0,
        omega=1.0e-9,
        threshold=1.0,
        dt=0.01,
    )
    assert math.hypot(neuron.x, neuron.y) > neuron.threshold
    assert neuron.step(0.0) == 0


def test_downward_y_crossing_is_not_an_event() -> None:
    neuron = ResonateAndFireNeuron(
        x=-1.0,
        y=1.01,
        b=0.0,
        omega=1.0,
        threshold=1.0,
        dt=0.02,
    )
    assert neuron.step(0.0) == 0
    assert neuron.y < 1.01


def test_zero_current_quiescent_state_remains_quiescent() -> None:
    neuron = ResonateAndFireNeuron()
    assert [neuron.step(0.0) for _ in range(100)] == [0] * 100
    assert (neuron.x, neuron.y) == (0.0, 0.0)


@pytest.mark.parametrize(("current", "expected_spikes"), ((5.0, 0), (10.0, 15)))
def test_source_default_constant_drive_regimes(current: float, expected_spikes: int) -> None:
    """Separate the source-default subthreshold and spiking regimes."""
    neuron = ResonateAndFireNeuron()
    assert sum(neuron.step(current) for _ in range(500)) == expected_spikes
