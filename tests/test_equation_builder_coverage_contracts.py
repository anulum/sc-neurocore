# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation builder coverage contract tests

"""Focused coverage contracts for equation-builder sandbox edge paths."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons._units import UNIT_REGISTRY
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations


def test_constructor_rejects_unknown_units_mode() -> None:
    """Only the documented unit modes are accepted at construction time."""
    with pytest.raises(ValueError, match="units must be"):
        EquationNeuron(equations={"v": "I"}, state={"v": 0.0}, units="permissive")


def test_runtime_helper_namespace_covers_safe_math_edges() -> None:
    """Runtime helpers cover clipped sigmoid, exprel(0), and sqrt domain errors."""
    helper_neuron = EquationNeuron(
        equations={"v": "sigmoid(1000.0) + exprel(v) + sqrt(4.0)"},
        state={"v": 0.0},
        dt=0.1,
    )

    helper_neuron.step()

    assert np.isfinite(helper_neuron.state["v"])

    sqrt_neuron = EquationNeuron(
        equations={"v": "sqrt(v)"},
        state={"v": -1.0},
        dt=0.1,
    )

    with pytest.raises(ValueError, match="sqrt domain error"):
        sqrt_neuron.step()


def test_sandbox_rejects_blocked_non_dunder_attribute() -> None:
    """Attribute validation blocks dangerous names even when they are not dunders."""
    with pytest.raises(ValueError, match="Blocked attribute"):
        EquationNeuron(equations={"v": "v.os"}, state={"v": 0.0})


def test_strict_units_require_state_for_each_equation_variable() -> None:
    """Strict units fail closed when a differential variable lacks initial units."""
    with pytest.raises(ValueError, match="explicit state quantities.*w"):
        from_equations(
            "dv/dt = -v / tau",
            "dw/dt = -w / tau",
            params={"tau": 10.0 * UNIT_REGISTRY.millisecond},
            init={"v": -65.0 * UNIT_REGISTRY.millivolt},
            dt=0.1 * UNIT_REGISTRY.millisecond,
            units="strict",
        )


def test_strict_units_detect_input_needed_only_by_reset_rule() -> None:
    """Reset expressions participate in strict-mode input-unit discovery."""
    with pytest.raises(ValueError, match="requires input_unit"):
        from_equations(
            "dv/dt = -v / tau",
            threshold="v > v_threshold",
            reset="v = R * I",
            params={
                "R": 100.0e6 * UNIT_REGISTRY.ohm,
                "tau": 10.0 * UNIT_REGISTRY.millisecond,
            },
            init={"v": -65.0 * UNIT_REGISTRY.millivolt},
            constants={"v_threshold": -50.0 * UNIT_REGISTRY.millivolt},
            dt=0.1 * UNIT_REGISTRY.millisecond,
            units="strict",
        )


def test_strict_runtime_rejects_unknown_quantity_kwarg_units() -> None:
    """Strict runtime rejects pint-valued public inputs without a declared unit."""
    neuron = from_equations(
        "dv/dt = -v / tau",
        params={"tau": 10.0 * UNIT_REGISTRY.millisecond},
        init={"v": -65.0 * UNIT_REGISTRY.millivolt},
        dt=0.1 * UNIT_REGISTRY.millisecond,
        units="strict",
    )
    auxiliary_drive = cast(float, 1.0 * UNIT_REGISTRY.volt)

    with pytest.raises(ValueError, match="No runtime unit declared"):
        neuron.step(auxiliary_drive=auxiliary_drive)
