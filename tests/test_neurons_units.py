# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the optional pint dimensional-unit helpers

"""Contracts for the optional pint-based dimensional-unit helpers."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons import _units

_UR = _units.UNIT_REGISTRY


@pytest.mark.parametrize("fn_name", ["exp", "log", "sin", "cos", "tanh", "sinh", "cosh", "sigmoid"])
def test_namespace_math_functions_reduce_dimensionless_quantities(fn_name: str) -> None:
    """Each transcendental in the quantity namespace consumes a dimensionless quantity."""
    namespace = _units.build_quantity_namespace()

    result = namespace[fn_name](1.5 * _UR.dimensionless)

    assert _units.is_quantity(result)


def test_sqrt_handles_dimensional_quantity_and_plain_float() -> None:
    """_sqrt roots a dimensional quantity (carrying units) and a bare float."""
    assert _units._sqrt(4.0 * _UR.metre).magnitude == pytest.approx(2.0)
    assert _units._sqrt(9.0).magnitude == pytest.approx(3.0)


def test_validate_expression_evaluates_and_checks_expected_units() -> None:
    """validate_quantity_expression evaluates an expression and verifies expected units."""
    namespace = _units.build_quantity_namespace()
    env = {"x": 2.0 * _UR.dimensionless, **namespace}

    assert _units.is_quantity(_units.validate_quantity_expression("exp(x)", env, label="t"))

    checked = _units.validate_quantity_expression(
        "x", {"x": 3.0 * _UR.metre}, expected_quantity=1.0 * _UR.metre, label="t"
    )
    assert checked.magnitude == pytest.approx(3.0)


def test_validate_expression_rejects_non_quantity_against_expected() -> None:
    """A bare number cannot satisfy an expected dimensional quantity."""
    with pytest.raises(_units.DimensionalError):
        _units.validate_quantity_expression(
            "x", {"x": 5.0}, expected_quantity=1.0 * _UR.metre, label="t"
        )


def test_validate_expression_maps_unknown_symbol_to_value_error() -> None:
    """An unknown symbol is reported as a ValueError naming the label."""
    with pytest.raises(ValueError, match="Unknown symbol"):
        _units.validate_quantity_expression("mystery", {}, label="t")


def test_validate_expression_propagates_dimensional_error() -> None:
    """A dimensional mismatch during evaluation propagates as DimensionalError."""
    with pytest.raises(_units.DimensionalError):
        _units.validate_quantity_expression(
            "a + b", {"a": 1.0 * _UR.metre, "b": 1.0 * _UR.second}, label="t"
        )


def test_validate_expression_wraps_other_errors() -> None:
    """A non-name, non-dimensional evaluation error is wrapped as a ValueError."""
    with pytest.raises(ValueError, match="Could not evaluate"):
        _units.validate_quantity_expression("1 / 0", {}, label="t")


def test_namespace_math_functions_accept_bare_floats() -> None:
    """A bare float passes through _dimensionless_magnitude's non-quantity branch."""
    namespace = _units.build_quantity_namespace()

    assert _units.is_quantity(namespace["exp"](0.0))


def test_clip_handles_quantity_and_plain_float() -> None:
    """_clip bounds a dimensional quantity in its own units and a bare float."""
    bounded_q = _units._clip(5.0 * _UR.metre, 1.0 * _UR.metre, 3.0 * _UR.metre)
    assert bounded_q.magnitude == pytest.approx(3.0)

    bounded_f = _units._clip(5.0, 1.0, 3.0)
    assert bounded_f.magnitude == pytest.approx(3.0)


def test_quantity_to_base_normalises_units() -> None:
    """quantity_to_base reduces a quantity to SI base units."""
    base = _units.quantity_to_base(2.0 * _UR.kilometre)
    assert base.magnitude == pytest.approx(2000.0)


def test_require_quantity_rejects_non_quantity() -> None:
    """require_quantity rejects a value that is not a pint Quantity under strict units."""
    with pytest.raises(ValueError, match="must be a pint Quantity"):
        _units.require_quantity(5.0, "weight")


def test_require_pint_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """require_pint raises a dependency error when pint is not installed."""
    from sc_neurocore.exceptions import SCDependencyError

    monkeypatch.setattr(_units, "HAS_PINT", False)
    with pytest.raises(SCDependencyError, match="pint is not available"):
        _units.require_pint()
