# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import math

import pytest

from sc_neurocore.neurons.models import SCChaoticMapNeuron


def test_preserved_two_state_recurrence() -> None:
    neuron = SCChaoticMapNeuron(x=0.4, y=-0.2)
    expected_x = 0.7 * 0.4 / (1.0 + math.exp(-2.4)) + 0.2 + 0.1
    expected_y = 0.95 * -0.2 + 0.05 * 0.4
    neuron.step(0.1)
    assert neuron.x == pytest.approx(expected_x, abs=1e-15)
    assert neuron.y == pytest.approx(expected_y, abs=1e-15)


def test_crossing_event_is_edge_triggered() -> None:
    neuron = SCChaoticMapNeuron(x=0.4, y=-0.2)
    assert neuron.step(0.1) == 1
    assert neuron.step(0.1) == 0


@pytest.mark.parametrize("field", ["x", "y", "k_f", "k_s", "alpha", "delta", "x_threshold"])
def test_constructor_rejects_non_finite_configuration(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        SCChaoticMapNeuron(**{field: math.nan})


def test_rejected_current_is_atomic() -> None:
    neuron = SCChaoticMapNeuron(x=0.4, y=-0.2)
    with pytest.raises(ValueError, match="current"):
        neuron.step(math.nan)
    assert (neuron.x, neuron.y) == (0.4, -0.2)


def test_reset_preserves_configuration() -> None:
    neuron = SCChaoticMapNeuron(x=1.0, y=2.0, alpha=3.0)
    neuron.reset()
    assert (neuron.x, neuron.y, neuron.alpha) == (0.0, 0.0, 3.0)
