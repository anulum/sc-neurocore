# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Physical unit scales and dimensional quantity algebra

"""Check unit-definition parsing and coherent arithmetic without model values."""

from __future__ import annotations

from fractions import Fraction

import pytest

from sc_neurocore.studio.candidate_unit_values import (
    CandidateUnitRefused,
    Unit,
    coherent,
    dimensionless,
    parse_unit,
    product,
)


def test_electrical_units_and_fractional_roots_retain_the_declared_scale() -> None:
    """Recognise the actual AdEx pA/pF and nS*mV identities without conversion."""
    voltage, current, time = (parse_unit(text) for text in ("mV", "pA", "ms"))
    coherent(product(current, parse_unit("pF"), divide=True), product(voltage, time, divide=True))
    coherent(product(parse_unit("nS"), voltage), current)
    coherent(voltage.power(Fraction(1, 2)), parse_unit("mV**0.5"))
    coherent(voltage.power(Fraction(0)), Unit())
    coherent(parse_unit("mV"), parse_unit("millivolt"))
    coherent(Unit(constant=Fraction(0), literal=True), voltage)
    coherent(voltage, Unit(constant=Fraction(0), literal=True))
    dimensionless(parse_unit("radian"))


@pytest.mark.parametrize("text", ["degC", "dB", "mV**1000", "mV**-1000", "not_a_unit", "mV/0"])
def test_unrepresentable_or_nonlinear_units_have_authored_refusals(text: str) -> None:
    """Reject unit definitions that numeric magnitudes cannot faithfully carry.

    Parameters
    ----------
    text : str
        Invalid, nonlinear or unrepresentable unit definition.
    """
    with pytest.raises(CandidateUnitRefused):
        parse_unit(text)


@pytest.mark.parametrize("text", ["mV", "percent", "degree"])
def test_physical_or_scaled_dimensionless_inputs_are_not_plain_numbers(text: str) -> None:
    """Keep physical magnitudes and hidden scale conversions out of functions.

    Parameters
    ----------
    text : str
        Physical or scaled dimensionless argument unit.
    """
    with pytest.raises(CandidateUnitRefused):
        dimensionless(parse_unit(text))


@pytest.mark.parametrize(
    ("scale", "exponent"), [(1e300, 64), (1e-300, 64), (1e-300, -64), (1e300, -64)]
)
def test_unit_powers_refuse_overflow_and_underflow(scale: float, exponent: int) -> None:
    """Avoid admitting a unit scale that collapses or exceeds finite arithmetic.

    Parameters
    ----------
    scale : float
        Finite declared scale.
    exponent : int
        Static exponent with an unrepresentable resulting scale.
    """
    with pytest.raises(CandidateUnitRefused, match="unit scale is not finite"):
        Unit(scale=scale).power(Fraction(exponent))


@pytest.mark.parametrize("divide", [False, True])
def test_unit_products_refuse_unrepresentable_scales(divide: bool) -> None:
    """Reject overflow while multiplying or dividing finite unit definitions.

    Parameters
    ----------
    divide : bool
        Exercise either overflowing product or quotient.
    """
    second = 1e-300 if divide else 1e300
    with pytest.raises(CandidateUnitRefused, match="unit scale is not finite"):
        product(Unit(scale=1e300), Unit(scale=second), divide=divide)
