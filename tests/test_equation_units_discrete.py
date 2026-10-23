# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for strict dimensional checking in EquationNeuron

"""Exercise discrete expression units through real strict models and native replay."""

from __future__ import annotations

import pytest
from pint import DimensionalityError as DimensionalError
from sc_neurocore.neurons._units import UNIT_REGISTRY
from sc_neurocore.neurons.equation_builder import from_equations


@pytest.mark.parametrize("millivolts", [-2500.0, -1500.0, -500.0, 500.0, 1250.0, 1500.0, 2500.0])
def test_strict_round_matches_base_unit_runtime(millivolts: float) -> None:
    """Rounding millivolt display quantities uses the same volt scale as runtime."""
    neuron = from_equations(
        "dv/dt = round(v) / tau",
        init={"v": millivolts * UNIT_REGISTRY.millivolt},
        params={"tau": 1.0 * UNIT_REGISTRY.second},
        dt=0.125 * UNIT_REGISTRY.second,
        units="strict",
    )
    neuron.step()
    result = neuron.get_state()["v"].to(UNIT_REGISTRY.volt).magnitude
    initial = millivolts / 1000.0
    assert result == initial + 0.125 * round(initial)


@pytest.mark.parametrize("millivolts", [-1.0, 0.0, 1.0])
def test_strict_heaviside_compares_physical_input_with_dimensionless_zero_value(
    millivolts: float,
) -> None:
    """Physical signs select a dimensionless value before the dimensional gain."""
    neuron = from_equations(
        "dv/dt = gain * heaviside(v, zero)",
        init={"v": millivolts * UNIT_REGISTRY.millivolt},
        params={
            "gain": 1.0 * UNIT_REGISTRY.volt / UNIT_REGISTRY.second,
            "zero": 0.25 * UNIT_REGISTRY.dimensionless,
        },
        dt=0.125 * UNIT_REGISTRY.second,
        units="strict",
    )
    neuron.step()
    expected = 0.0 if millivolts < 0 else 0.25 if millivolts == 0 else 1.0
    assert (
        neuron.get_state()["v"].to(UNIT_REGISTRY.volt).magnitude
        == millivolts / 1000.0 + 0.125 * expected
    )


def test_strict_heaviside_refuses_dimensional_zero_value() -> None:
    """A second operand carrying volts cannot become a dimensionless step output."""
    with pytest.raises(DimensionalError):
        from_equations(
            "dv/dt = gain * heaviside(v, v)",
            init={"v": 1.0 * UNIT_REGISTRY.volt},
            params={"gain": 1.0 * UNIT_REGISTRY.volt / UNIT_REGISTRY.second},
            dt=0.125 * UNIT_REGISTRY.second,
            units="strict",
        )


@pytest.mark.parametrize(("call", "expected"), [("round(1.5)", 2.0), ("heaviside(0, 0.25)", 0.25)])
def test_discrete_scalar_coefficient_is_dimensionless(call: str, expected: float) -> None:
    """Bare numeric operands are admitted as dimensionless coefficients in strict models."""
    neuron = from_equations(
        "dv/dt = gain * " + call,
        init={"v": 0.0 * UNIT_REGISTRY.volt},
        params={"gain": 1.0 * UNIT_REGISTRY.volt / UNIT_REGISTRY.second},
        dt=0.125 * UNIT_REGISTRY.second,
        units="strict",
    )
    neuron.step()
    assert neuron.get_state()["v"].to(UNIT_REGISTRY.volt).magnitude == 0.125 * expected
