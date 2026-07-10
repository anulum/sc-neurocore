# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Equation strict-unit runtime — pint dimensional validation

"""Strict pint-backed dimensional validation for equation-defined neurons.

Opt-in ``units="strict"`` mode validates that every dynamics, reset, and
threshold expression is dimensionally consistent before the model is compiled,
then converts each pint quantity to a base-unit float for the numeric runtime.
This is a self-contained responsibility split out of :class:`EquationNeuron`:
the neuron passes its parsed equations and raw quantities in and receives a
:class:`StrictRuntime` of base-unit floats plus the unit maps it needs to
convert runtime inputs and re-attach display units. The non-strict hot path
never touches this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from sc_neurocore.neurons._units import (
    UNIT_REGISTRY,
    build_quantity_namespace,
    is_quantity,
    quantity_to_base,
    require_pint,
    require_quantity,
    validate_quantity_expression,
)


@dataclass(frozen=True)
class StrictRuntime:
    """Base-unit runtime values and unit maps produced by strict validation.

    ``parameters``/``state``/``constants``/``dt`` are the base-unit floats the
    integrator steps; ``runtime_units`` maps each name (and the input ``I``) to
    its base unit for converting runtime inputs; ``base_state_units`` and
    ``display_state_units`` let :meth:`EquationNeuron.get_state` re-attach the
    caller's display units to each state variable.
    """

    parameters: dict[str, float]
    state: dict[str, float]
    constants: dict[str, float]
    dt: float
    runtime_units: dict[str, Any]
    base_state_units: dict[str, Any]
    display_state_units: dict[str, Any]


def prepare_strict_runtime(
    *,
    equations: dict[str, str],
    threshold_expr: str | None,
    reset_rules: dict[str, str],
    input_unit_name: str,
    raw_parameters: dict[str, Any],
    raw_state: dict[str, Any],
    raw_constants: dict[str, Any],
    dt: Any,
    input_unit: Any | None,
) -> StrictRuntime:
    """Convert pint quantities to base-unit floats for runtime."""
    require_pint()

    runtime_units: dict[str, Any] = {}
    base_state_units: dict[str, Any] = {}
    display_state_units: dict[str, Any] = {}

    missing_state = sorted(set(equations) - set(raw_state))
    if missing_state:
        raise ValueError(
            "units='strict' requires explicit state quantities for all equation variables: "
            + ", ".join(missing_state)
        )

    dt_quantity = require_quantity(dt, "dt")
    dt_base = quantity_to_base(dt_quantity)
    dt_base.to(UNIT_REGISTRY.second)

    quantity_parameters = {
        name: require_quantity(value, f"parameter {name}") for name, value in raw_parameters.items()
    }
    quantity_state = {
        name: require_quantity(value, f"state {name}") for name, value in raw_state.items()
    }
    quantity_constants = {
        name: require_quantity(value, f"constant {name}") for name, value in raw_constants.items()
    }

    quantity_env = build_quantity_namespace()
    quantity_env.update(quantity_parameters)
    quantity_env.update(quantity_constants)
    quantity_env.update(quantity_state)
    quantity_env["xi"] = 1.0 * UNIT_REGISTRY.dimensionless

    uses_input = any(re.search(r"\bI\b", expr) for expr in equations.values())
    if threshold_expr:
        uses_input = uses_input or bool(re.search(r"\bI\b", threshold_expr))
    if any(re.search(r"\bI\b", expr) for expr in reset_rules.values()):
        uses_input = True

    if uses_input:
        if input_unit is None:
            raise ValueError(
                "units='strict' requires input_unit when equations reference the special input 'I'"
            )
        input_quantity = require_quantity(input_unit, "input_unit")
        quantity_env[input_unit_name] = input_quantity
        runtime_units[input_unit_name] = quantity_to_base(input_quantity).units

    for var, expr in equations.items():
        expected = quantity_state[var] / dt_quantity
        validate_quantity_expression(
            expr,
            quantity_env,
            expected_quantity=expected,
            label=f"d{var}/dt",
        )

    for var, expr in reset_rules.items():
        validate_quantity_expression(
            expr,
            quantity_env,
            expected_quantity=quantity_state[var],
            label=f"reset {var}",
        )

    if threshold_expr:
        threshold_result = validate_quantity_expression(
            threshold_expr,
            quantity_env,
            label="threshold",
        )
        if not isinstance(threshold_result, (bool, np.bool_)):
            raise ValueError("Threshold expression must evaluate to a boolean in strict units mode")

    runtime_parameters = {}
    runtime_state = {}
    runtime_constants = {}

    for name, quantity in quantity_parameters.items():
        runtime_parameters[name] = float(quantity_to_base(quantity).magnitude)
        runtime_units[name] = quantity_to_base(quantity).units

    for name, quantity in quantity_constants.items():
        runtime_constants[name] = float(quantity_to_base(quantity).magnitude)
        runtime_units[name] = quantity_to_base(quantity).units

    for name, quantity in quantity_state.items():
        base_quantity = quantity_to_base(quantity)
        runtime_state[name] = float(base_quantity.magnitude)
        runtime_units[name] = base_quantity.units
        base_state_units[name] = base_quantity.units
        display_state_units[name] = quantity.units

    return StrictRuntime(
        parameters=runtime_parameters,
        state=runtime_state,
        constants=runtime_constants,
        dt=float(dt_base.magnitude),
        runtime_units=runtime_units,
        base_state_units=base_state_units,
        display_state_units=display_state_units,
    )


def convert_runtime_value(
    *,
    strict_units: bool,
    runtime_units: dict[str, Any],
    name: str,
    value: Any,
) -> float:
    """Convert a runtime value from pint quantity to float."""
    if not strict_units:
        return float(value)
    if not is_quantity(value):
        return float(value)
    if name not in runtime_units:
        raise ValueError(f"No runtime unit declared for {name!r}")
    return float(quantity_to_base(value).to(runtime_units[name]).magnitude)
