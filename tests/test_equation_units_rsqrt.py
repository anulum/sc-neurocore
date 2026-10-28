# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reciprocal root physical dimensions and runtime domains
"""Exercise positive finite inputs and inverse-square-root physical dimensions."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from sc_neurocore.neurons._units import DimensionalError, UNIT_REGISTRY as u
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations
from sc_neurocore.neurons.equation_namespace import build_eval_namespace


@pytest.mark.parametrize(
    "value", [math.ulp(0.0), sys.float_info.min, 0.25, 1.0, 4.0, sys.float_info.max]
)
def test_finite_positive_roots_preserve_float_range(value: float) -> None:
    """Subnormals and maximal finite inputs avoid a premature reciprocal overflow."""
    with np.errstate(all="raise"):
        assert build_eval_namespace()["rsqrt"](value) == pytest.approx(
            1 / math.sqrt(value), rel=1e-15
        )


@pytest.mark.parametrize("bad", [0.0, -0.0, -1.0, math.inf, -math.inf, math.nan, 1j])
def test_invalid_domain_refuses_before_warnings(bad: complex) -> None:
    """Invalid scalar and mixed-array domains cannot silently produce infinity or NaN."""
    with np.errstate(all="raise"), pytest.raises(ValueError, match="finite positive real"):
        build_eval_namespace()["rsqrt"](bad)
    with np.errstate(all="raise"), pytest.raises(ValueError, match="finite positive real"):
        build_eval_namespace()["rsqrt"](np.array([1.0, bad]))


def test_vector_roots_and_runtime_failure_preserve_state() -> None:
    """A real neuron refuses a zero drive without committing an invalid next state."""
    assert build_eval_namespace()["rsqrt"](np.array([0.25, 1.0, 4.0])).tolist() == [2, 1, 0.5]
    neuron = EquationNeuron({"v": "rsqrt(I)"}, state={"v": 0.25}, method="map")
    with pytest.raises(ValueError, match="finite positive"):
        neuron.step(0.0)
    assert neuron.state["v"] == 0.25
    neuron.step(4.0)
    assert neuron.state["v"] == 0.5


@pytest.mark.parametrize("dimensioned", [False, True])
def test_strict_units_preserve_inverse_square_root_dimension(dimensioned: bool) -> None:
    """Display-unit conversion precedes the dimensional reciprocal root."""
    quantity = 4000 * u.millivolt if dimensioned else 4 * u.dimensionless
    gain = 1 * u.volt**1.5 / u.second if dimensioned else 1 * u.volt / u.second
    neuron = from_equations(
        "dv/dt=gain*rsqrt(offset)",
        init={"v": 0 * u.volt},
        params={"gain": gain, "offset": quantity},
        dt=0.125 * u.second,
        units="strict",
    )
    neuron.step()
    assert neuron.get_state()["v"].to(u.volt).magnitude == 0.0625


def test_wrong_dimensions_refuse() -> None:
    """The inverse half-power unit cannot be mistaken for a voltage drift."""
    with pytest.raises(DimensionalError):
        from_equations(
            "dv/dt=rsqrt(offset)/tau",
            init={"v": 1 * u.volt},
            params={"offset": 4 * u.volt, "tau": 1 * u.second},
            dt=0.125 * u.second,
            units="strict",
        )


def test_strict_literal_argument_is_dimensionless() -> None:
    """A bare positive literal acquires a dimensionless reciprocal root."""
    neuron = from_equations(
        "dv/dt=gain*rsqrt(4)",
        init={"v": 0 * u.volt},
        params={"gain": 1 * u.volt / u.second},
        dt=0.125 * u.second,
        units="strict",
    )
    neuron.step()
    assert neuron.get_state()["v"].to(u.volt).magnitude == 0.0625
