# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypotenuse physical and derivative contracts
"""Exercise finite coordinates, physical dimensions and the norm's singular origin."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons._units import DimensionalError, UNIT_REGISTRY as u
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.neurons.equation_evaluation import evaluate_scalar
from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_program import compile_expression
from sc_neurocore.neurons.expression_derivative import ExpressionDifferentiationError, differentiate


def test_strict_norm_converts_compatible_display_units() -> None:
    """Mixed volt/millivolt coordinates produce the correct physical drift."""
    neuron = from_equations(
        "dv/dt = hypot(v, offset)/tau",
        init={"v": -3000 * u.millivolt},
        params={"offset": 4 * u.volt, "tau": 1 * u.second},
        dt=0.125 * u.second,
        units="strict",
    )
    neuron.step()
    assert neuron.get_state()["v"].to(u.volt).magnitude == -3 + 0.125 * 5


@pytest.mark.parametrize("first,second", [(3, 4), (-3, 4), (0, 0)])
def test_strict_scalar_norm_is_dimensionless(first: int, second: int) -> None:
    """Bare constants acquire a dimensionless norm before the dimensional gain."""
    neuron = from_equations(
        f"dv/dt = gain*hypot({first},{second})",
        init={"v": 0 * u.volt},
        params={"gain": 1 * u.volt / u.second},
        dt=0.125 * u.second,
        units="strict",
    )
    neuron.step()
    assert neuron.get_state()["v"].to(u.volt).magnitude == 0.125 * math.hypot(first, second)


def test_norm_refuses_incompatible_physical_dimensions() -> None:
    """A time coordinate cannot silently be added to a voltage coordinate."""
    with pytest.raises(DimensionalError):
        from_equations(
            "dv/dt=hypot(v,tau)/tau",
            init={"v": 1 * u.volt},
            params={"tau": 1 * u.second},
            dt=0.125 * u.second,
            units="strict",
        )


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 1j])
@pytest.mark.parametrize("position", [0, 1])
def test_runtime_norm_refuses_nonfinite_or_complex_coordinates(bad: complex, position: int) -> None:
    """Both coordinate positions refuse values outside the declared real finite domain."""
    values = [0.0, 0.0]
    operands = [bad, 0.0] if position == 0 else [0.0, bad]
    namespace = build_eval_namespace()
    with pytest.raises(ValueError, match="finite real"):
        namespace["hypot"](*operands)
    assert namespace["hypot"](*values) == 0


def test_runtime_norm_does_not_overflow_by_squaring_finite_coordinates() -> None:
    """A representable result survives coordinates whose individual squares overflow float64."""
    namespace = build_eval_namespace()
    assert namespace["hypot"](3e200, 4e200) == pytest.approx(5e200)
    assert namespace["hypot"](np.array([3.0, 0.0]), np.array([4.0, -2.0])).tolist() == [5.0, 2.0]


@pytest.mark.parametrize("expression", ["hypot(v,2)", "hypot(-2,v)", "hypot(v*v,1)"])
def test_norm_derivative_matches_independent_finite_difference(expression: str) -> None:
    """Proven nonzero coordinates admit either partial and its chain rule."""
    derivative = compile_expression(differentiate(expression, "v"))
    source = compile_expression(expression)
    namespace = build_eval_namespace()
    for at in [-3.0, -0.1, 0.0, 0.1, 3.0]:
        analytic = evaluate_scalar(derivative, {**namespace, "v": at})
        upper = evaluate_scalar(source, {**namespace, "v": at + 1e-5})
        lower = evaluate_scalar(source, {**namespace, "v": at - 1e-5})
        assert analytic == pytest.approx((upper - lower) / 2e-5, abs=1e-8)


@pytest.mark.parametrize("expression", ["hypot(v)", "hypot(v,1,2)", "hypot(v,second=1)"])
def test_norm_differentiation_refuses_unsupported_call_signatures(expression: str) -> None:
    """The public differentiator never drops coordinates or keyword operands."""
    with pytest.raises(ExpressionDifferentiationError, match="exactly two positional"):
        differentiate(expression, "v")


@pytest.mark.parametrize("expression", ["hypot(v,v)", "hypot(v,0)"])
def test_norm_derivative_refuses_an_unproven_nonzero_domain(expression: str) -> None:
    """An exponential integrator never invents a Jacobian at the norm's origin."""
    with pytest.raises(ExpressionDifferentiationError, match="provably nonzero"):
        EquationNeuron({"v": expression}, method="exp_euler")


def test_state_independent_norm_survives_as_a_derivative_coefficient() -> None:
    """Constant-coordinate norms need no state-domain proof to multiply the drift."""
    derivative = compile_expression(differentiate("v*hypot(I,z)", "v"))
    assert evaluate_scalar(derivative, {**build_eval_namespace(), "I": 3.0, "z": 4.0}) == 5
