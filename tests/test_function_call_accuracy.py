# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent numerical checks on inferred call domains
"""Check mathematical-domain counts and measured error through the public measurement API."""

import math

import pytest
import numpy as np

from sc_neurocore.compiler.fixed_expression_ranges import CodeRange
from sc_neurocore.compiler.function_call_accuracy import measure_call


def test_empty_real_domain_has_no_zero_error_claim() -> None:
    """A logarithm fed only negative encoded values has no real approximation evidence."""
    accuracy = measure_call("log", 8, 4, (CodeRange(-16, -1),))
    assert accuracy.domain_cardinality == accuracy.codes_evaluated == 0
    assert accuracy.excluded_domain_codes == 16
    assert accuracy.max_error is None and accuracy.worst_inputs == ()


def test_narrow_exp_measurement_matches_an_independent_enumeration() -> None:
    """Every code in the narrow argument interval is compared with a real exponential."""
    from sc_neurocore.compiler.function_units import evaluate

    accuracy = measure_call("exp", 8, 4, (CodeRange(-32, 31),))
    expected = max(
        abs(evaluate("exp", code, 8, 4) / 16 - min(127 / 16, math.exp(code / 16)))
        for code in range(-32, 32)
    )
    assert accuracy.codes_evaluated == accuracy.domain_cardinality == 64
    assert accuracy.exhaustive and accuracy.max_error == pytest.approx(expected)


@pytest.mark.parametrize("function", ["atan2", "heaviside", "hypot"])
def test_binary_call_measures_the_declared_operand_box(function: str) -> None:
    """Both operands contribute to cardinality and the complete worst-input tuple."""
    accuracy = measure_call(function, 4, 2, (CodeRange(-2, 3), CodeRange(0, 2)))
    assert accuracy.exhaustive and accuracy.codes_evaluated == accuracy.domain_cardinality == 18
    assert len(accuracy.worst_inputs) == 2 and accuracy.excluded_domain_codes == 0


@pytest.mark.parametrize(
    "exponent,cardinality,excluded", [(-1, 8, 1), (1.5, 5, 4), (-1.5, 4, 5), (0, 9, 0)]
)
def test_literal_power_domain_exclusions(exponent: float, cardinality: int, excluded: int) -> None:
    """Integral parity, zero and negative-base admission remain distinct."""
    accuracy = measure_call("power", 4, 2, (CodeRange(-4, 4),), exponent=exponent)
    assert accuracy.domain_cardinality == accuracy.codes_evaluated == cardinality
    assert accuracy.excluded_domain_codes == excluded
    assert accuracy.declared_error_bound is not None and accuracy.max_error is not None
    assert accuracy.max_error <= accuracy.declared_error_bound


def test_dynamic_power_uses_only_strictly_positive_base_codes() -> None:
    """Invalid base tuples are counted outside the approximation claim."""
    accuracy = measure_call("power", 4, 2, (CodeRange(-4, 4), CodeRange(-2, 2)))
    assert accuracy.domain_cardinality == accuracy.codes_evaluated == 20
    assert accuracy.excluded_domain_codes == 25 and accuracy.exhaustive


def test_inverse_trigonometric_calls_clip_both_domain_endpoints() -> None:
    """The closed arcsine domain excludes encoded inputs on both sides of [-1,1]."""
    accuracy = measure_call("arcsin", 8, 4, (CodeRange(-32, 32),))
    assert accuracy.valid_code_intervals == (((-16, 16),),)
    assert accuracy.domain_cardinality == accuracy.codes_evaluated == 33
    assert accuracy.excluded_domain_codes == 32


@pytest.mark.parametrize(
    "function,lowering,exponent,operands",
    [
        ("exp", "unknown", None, (CodeRange(0, 1),)),
        ("power", "lut-v1", 2, (CodeRange(0, 1),)),
        ("exp", "v2", 2, (CodeRange(0, 1),)),
        ("atan2", "v2", None, (CodeRange(0, 1),)),
        ("exp", "v2", None, (CodeRange(-129, 1),)),
    ],
)
def test_incompatible_measurement_requests_are_refused(
    function: str, lowering: str, exponent: float | None, operands: tuple[CodeRange, ...]
) -> None:
    """No result is issued for an incorrect lowering, arity, specialisation or word interval."""
    with pytest.raises(ValueError):
        measure_call(function, 8, 4, operands, lowering=lowering, exponent=exponent)


@pytest.mark.parametrize(
    "function,width,fraction,ranges,exponent",
    [
        ("round", 17, 4, (CodeRange(-65536, 65535),), None),
        ("heaviside", 9, 4, (CodeRange(-256, 255), CodeRange(-256, 255)), None),
        ("power", 14, 0, (CodeRange(-8192, 8191),), -1),
    ],
)
def test_large_call_domains_state_their_sampled_scope(
    function: str, width: int, fraction: int, ranges: tuple[CodeRange, ...], exponent: float | None
) -> None:
    """Wide unary, binary and disjoint power domains retain honest sampled cardinalities."""
    accuracy = measure_call(function, width, fraction, ranges, exponent=exponent)
    assert not accuracy.exhaustive
    assert 0 < accuracy.codes_evaluated < accuracy.domain_cardinality
    assert accuracy.max_error is not None and accuracy.declared_error_bound is not None
    assert accuracy.max_error <= accuracy.declared_error_bound


def test_actual_unit_bound_failure_propagates_through_the_call_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupted computed output cannot be turned into a successful or hidden accuracy record."""
    from sc_neurocore.compiler import function_units
    from sc_neurocore.compiler.lut_accuracy import measure
    from sc_neurocore.compiler.function_call_domains import function_call_contracts
    from sc_neurocore.neurons.equation_builder import EquationNeuron

    measure("round", 8, 4, "v2")
    original = function_units.UNITS["round"]

    def shifted(code: int, width: int, fraction: int) -> int:
        """Change the actual unit result by one output LSB."""
        return original(code, width, fraction) + 1

    with monkeypatch.context() as context:
        context.setitem(function_units.UNITS, "round", shifted)
        with pytest.raises(ValueError, match="declared arithmetic error bound"):
            measure_call("round", 8, 4, (CodeRange(97, 97),))
        model = EquationNeuron({"v": "round(p)"}, parameters={"p": 98 / 16}, method="map")
        with pytest.raises(ValueError, match="declared arithmetic error bound"):
            function_call_contracts(model, 8, 4)
    assert measure_call("round", 8, 4, (CodeRange(97, 97),)).max_error == 0.0
    assert function_call_contracts(model, 8, 4)[0].accuracy is not None


@pytest.mark.parametrize(("reference", "bounds"), [("sqrt", (-37, -37)), ("arccos", (0, 32))])
def test_nonfinite_reference_is_refused_and_recovers(
    monkeypatch: pytest.MonkeyPatch, reference: str, bounds: tuple[int, int]
) -> None:
    """An invalid reference is refused even after a prefix of valid measurements.

    Parameters
    ----------
    monkeypatch : MonkeyPatch
        Substitutes an actual NumPy function with a different real domain.
    reference : str
        Square root fails immediately; inverse cosine fails after valid inputs.
    bounds : tuple of int
        Encoded input interval that reaches the substituted function's invalid domain.
    """
    from sc_neurocore.compiler import lut_accuracy
    from sc_neurocore.neurons.equation_namespace import build_eval_namespace

    lut_accuracy.measure("exp", 8, 4, "v2")
    original = build_eval_namespace

    def wrong_domain_namespace() -> dict[str, object]:
        """Use a real NumPy function whose invalid inputs occur at either series end."""
        namespace = original()
        namespace["exp"] = np.sqrt if reference == "sqrt" else np.arccos
        return namespace

    with monkeypatch.context() as context:
        context.setattr(lut_accuracy, "build_eval_namespace", wrong_domain_namespace)
        with pytest.raises(ValueError, match="not finite"):
            measure_call("exp", 8, 4, (CodeRange(*bounds),))
    assert measure_call("exp", 8, 4, (CodeRange(*bounds),)).max_error is not None
