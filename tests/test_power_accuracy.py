# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real power domains and compiler numeric contracts
"""Check independent power measurements and the complete public numeric-contract projection."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from sc_neurocore.compiler import power_units
from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
from sc_neurocore.compiler.power_accuracy import measure_power
from sc_neurocore.compiler.q_format import QFormat
from sc_neurocore.neurons.equation_builder import EquationNeuron


@pytest.mark.parametrize(
    "exponent,count", [(None, 112), (-3.0, 15), (1.5, 8), (-0.5, 7), (0.0, 16)]
)
def test_small_domain_cardinality_means_every_valid_input(
    exponent: float | None, count: int
) -> None:
    """Exhaustive flags count mathematical domains rather than all possible signed tuples."""
    measured = measure_power(4, 2, exponent)
    public = measured.to_public_dict()
    assert measured.exhaustive and measured.codes_evaluated == count
    assert public["domain_cardinality"] == count
    assert measured.declared_error_bound is not None
    assert 0 <= measured.max_error <= measured.declared_error_bound
    assert len(measured.worst_inputs) == (2 if exponent is None else 1)
    assert "Decimal" in str(public["reference"])


def test_wide_samples_do_not_claim_exhaustive_or_tight_error() -> None:
    """A difficult literal keeps a sampled maximum separate from a conservative global bound."""
    measured = measure_power(32, 29, 1e10)
    assert not measured.exhaustive and measured.codes_evaluated > 100
    assert measured.declared_error_bound is not None
    assert measured.max_error <= measured.declared_error_bound
    literal = measured.exponent
    assert literal is not None
    assert abs(Fraction(literal.value) - Fraction(literal.code, 1 << literal.fraction)) <= Fraction(
        1, 2 << literal.fraction
    )
    assert (
        measured.to_public_dict()["outside_domain"]
        == "zero sentinel, outside the mathematical approximation claim"
    )


def test_contract_retains_each_constant_and_dynamic_specialization() -> None:
    """Distinct powers cannot collapse into one entry keyed only by the function name."""
    model = EquationNeuron({"v": "I**1.5 + I**-3 + I**(I+.5)"}, method="map", dt=0.125)
    contract = hardware_numeric_contract(model, QFormat(8, 8), function_lowering="v2")
    powers = [
        unit.to_public_dict()["accuracy"]
        for unit in contract.function_units
        if unit.function == "power"
    ]
    assert len(powers) == 3
    specializations = [
        measure_power(16, 8, exponent).to_public_dict() for exponent in (-3, 1.5, None)
    ]
    assert powers == specializations
    assert all(quantity.value != 1.5 for quantity in contract.quantities)
    assert not contract.mirror_refusal


def test_named_constant_and_shadowed_state_take_different_exponent_paths() -> None:
    """Unshadowed pi is independent precision; a declared pi state remains a Q-word operand."""
    literal = EquationNeuron({"v": "I**pi"}, method="map", dt=0.125)
    dynamic = EquationNeuron({"v": "I**pi", "pi": "pi"}, state={"pi": 0.5}, method="map", dt=0.125)
    first = hardware_numeric_contract(literal, QFormat(1, 15), function_lowering="v2")
    second = hardware_numeric_contract(dynamic, QFormat(1, 15), function_lowering="v2")
    assert first.function_units[0].accuracy == measure_power(16, 15, math.pi)
    assert second.function_units[0].accuracy == measure_power(16, 15)
    assert first.representable and second.representable


def test_dynamic_exponent_reports_its_encoded_literal_subexpressions() -> None:
    """A tiny addition in a dynamic exponent is visible as a rounded ordinary Q literal."""
    model = EquationNeuron({"v": "I**(w+1e-8)", "w": "w"}, state={"w": 0.5}, method="map", dt=0.125)
    contract = hardware_numeric_contract(model, QFormat(8, 8), function_lowering="v2")
    quantity = next(item for item in contract.quantities if item.value == 1e-8)
    assert quantity.rtl_value == 0 and quantity.status == "underflows_to_zero"
    assert not quantity.blocking


@pytest.mark.parametrize(
    "power,gain,real",
    [
        ("gain**-3", 16.0, 1 / 4096),
        ("gain**9", 0.25, 1 / 262144),
        ("gain**2.0", 0.00390625, 1 / 65536),
        ("gain**1.5", 0.00390625, 1 / 4096),
        ("gain**alpha", 16.0, 1 / 4096),
    ],
)
def test_a_power_divisor_lost_to_rounding_blocks_the_format(
    power: str, gain: float, real: float
) -> None:
    """The real nonzero constant power cannot silently become a hardware zero divisor."""
    model = EquationNeuron(
        {"v": f"1/({power})"},
        parameters={"gain": gain, "alpha": -3.0},
        method="map",
        dt=0.125,
    )
    contract = hardware_numeric_contract(model, QFormat(8, 8), function_lowering="v2")
    divisor = next(item for item in contract.quantities if item.kind == "divisor")
    assert divisor.blocking and divisor.rtl_value == 0 and divisor.value == real
    assert not contract.representable


def test_integer_format_preserves_legacy_constant_multiply_folding() -> None:
    """Fraction-zero divisor checks do not form a negative rounding-bit shift."""
    model = EquationNeuron({"v": "1/(gain**2)"}, parameters={"gain": 2.0}, method="map", dt=1)
    contract = hardware_numeric_contract(model, QFormat(8, 0), function_lowering="v2")
    assert contract.representable and not contract.mirror_refusal


def test_canonical_root_divisors_retain_their_existing_unit_contracts() -> None:
    """Shared classification preserves root accuracy entries rather than inventing powers."""
    model = EquationNeuron(
        {"v": "1/(gain**.5)+1/(gain**(1/3))"},
        parameters={"gain": 4.0},
        method="map",
        dt=0.125,
    )
    contract = hardware_numeric_contract(model, QFormat(4, 4), function_lowering="v2")
    assert [unit.function for unit in contract.function_units] == ["cbrt", "sqrt"]
    assert not [quantity for quantity in contract.quantities if quantity.kind == "divisor"]


@pytest.mark.parametrize("base,exponent", [(2.0, "I"), (-1.0, "1.5"), (0.0, "-3")])
def test_unfoldable_power_divisors_do_not_invent_constant_values(
    base: float, exponent: str
) -> None:
    """An input-dependent or nonreal expression cannot masquerade as a real folded divisor."""
    model = EquationNeuron(
        {"v": f"1/(gain**{exponent})"}, parameters={"gain": base}, method="map", dt=0.125
    )
    contract = hardware_numeric_contract(model, QFormat(8, 8), function_lowering="v2")
    assert not [quantity for quantity in contract.quantities if quantity.kind == "divisor"]
    assert len(contract.function_units) == 1
    assert "outside the mathematical approximation claim" in str(
        contract.function_units[0].accuracy.to_public_dict()["outside_domain"]
    )


def test_measurement_refuses_a_corrupted_polynomial_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injected coefficient corruption reaches real integer and independent Decimal evaluation.

    This verifies refusal of a violated bound; ordinary accuracy and compiler
    execution tests use the unmodified coefficient generator.
    """
    log, exp = power_units.coefficients()
    corrupted = (exp[0] + (1 << 54), *exp[1:])
    with monkeypatch.context() as fault:
        fault.setattr(power_units, "coefficients", lambda: (log, corrupted))
        with pytest.raises(ValueError, match="exceeds its declared arithmetic error bound"):
            measure_power(6, 3, 1.25)
    recovered = measure_power(6, 3, 1.25)
    assert recovered.declared_error_bound is not None
    assert recovered.max_error <= recovered.declared_error_bound
