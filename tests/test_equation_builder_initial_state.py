# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Initial equation state and held-state runtime contracts

"""Exercise partial initial state through model timesteps, event resets and restart."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_builder import SUPPORTED_METHODS, EquationNeuron


@pytest.mark.parametrize("method", SUPPORTED_METHODS)
def test_partial_initial_state_matches_explicit_defaults_without_mutating_the_caller(
    method: str,
) -> None:
    """Missing equation words start at zero; supplied held state survives every stage."""
    supplied = {"held": 0.5}
    model = EquationNeuron(
        {"v": "held + I"},
        state=supplied,
        method=method,
        dt=0.25,
        threshold="v > 0.5",
        reset={"held": "held + 0.25", "v": "-0.25"},
        spike_value="held + v",
    )
    assert model.initial_state == model.get_state() == {"held": 0.5, "v": 0.0}
    assert supplied == {"held": 0.5}
    expected = (
        [
            ({"held": 0.75, "v": -0.25}, 1, 1.25),
            ({"held": 1.0, "v": -0.25}, 1, 1.75),
            ({"held": 1.25, "v": -0.25}, 1, 2.25),
        ]
        if method == "map"
        else [
            ({"held": 0.5, "v": 0.1875}, 0, 0.0),
            ({"held": 0.5, "v": 0.375}, 0, 0.0),
            ({"held": 0.75, "v": -0.25}, 1, 1.0625),
        ]
    )
    for state, spike, value in expected:
        assert model.step(0.25) == spike
        assert model.get_state() == state
        assert model.last_spike_value == value
    assert model.initial_state == {"held": 0.5, "v": 0.0}
    assert supplied == {"held": 0.5}
    model.reset()
    assert model.get_state() == {"held": 0.5, "v": 0.0}
    assert model.last_spike_value == 0.0
    assert model.step(0.25) == expected[0][1]
    assert model.get_state() == expected[0][0]


@pytest.mark.parametrize("supplied", [None, {}, {"v": 0.5}])
def test_complete_and_default_initial_states_keep_their_existing_runtime_contract(
    supplied: dict[str, float] | None,
) -> None:
    """Existing default and explicit reset values retain the same map trajectory."""
    model = EquationNeuron({"v": "0.5 * v + I"}, state=supplied, method="map")
    initial = 0.5 if supplied else 0.0
    assert model.get_state() == {"v": initial}
    model.step(0.25)
    assert model.get_state() == {"v": initial * 0.5 + 0.25}
    model.reset()
    assert model.get_state() == {"v": initial}


def test_strict_units_still_require_every_authored_state_quantity() -> None:
    """A dimensioned state is never silently replaced by an untyped numeric zero."""
    from sc_neurocore.neurons._units import UNIT_REGISTRY

    with pytest.raises(ValueError, match="explicit state quantities.*"):
        EquationNeuron(
            {"v": "-v"},
            state={"held": 0.5 * UNIT_REGISTRY.volt},
            dt=0.25 * UNIT_REGISTRY.second,
            units="strict",
        )
