# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — authored equation bindings and intrinsic lowering

"""Validate authored function names before introducing compiler-owned intrinsics."""

from __future__ import annotations

import ast
from collections.abc import Iterator

from ..neurons.equation_builder import EquationNeuron
from .diffusion_noise import lower_diffusion_noise
from .expr_lut_tables import check_call_bindings
from .philox_threshold import lower_philox_threshold


def equation_expressions(neuron: EquationNeuron) -> Iterator[tuple[str, str]]:
    """Yield model expressions and their locations in the hardware contract.

    Parameters
    ----------
    neuron : EquationNeuron
        Model before or after intrinsic lowering.

    Yields
    ------
    tuple of str
        Location and expression, including exponential-Euler Jacobians and events.
    """
    for variable, expression in neuron.equations.items():
        yield f"equation {variable}", expression
    if neuron.method == "exp_euler":
        for variable, expression in neuron.jacobian_expressions.items():
            yield f"jacobian {variable}", expression
    if neuron.threshold_expr:
        yield "threshold", neuron.threshold_expr
    for variable, expression in neuron.reset_rules.items():
        yield f"reset {variable}", expression
    if neuron.rate_expression:
        yield "escape rate", neuron.rate_expression
    if neuron.probability_expression:
        yield "spike probability", neuron.probability_expression
    if neuron.spike_value_expr:
        yield "spike value", neuron.spike_value_expr


def prepare_neuron(neuron: EquationNeuron) -> EquationNeuron:
    """Check authored bindings, then lower implicit noise and Philox thresholds.

    Parameters
    ----------
    neuron : EquationNeuron
        Original model supplied to a compiler entry point.

    Returns
    -------
    EquationNeuron
        Lowered model; the original is preserved by the intrinsic transforms.

    Raises
    ------
    ValueError
        If an authored call names a numeric state, parameter or constant.

    Notes
    -----
    Internal calls introduced afterward, such as ``randn`` for ``xi`` and
    ``exprel`` for exponential Euler, are independent of numeric model bindings.
    """
    bound = {*neuron.equations, *neuron.parameters, *neuron.constants}
    for _where, expression in equation_expressions(neuron):
        check_call_bindings(ast.parse(expression, mode="eval"), bound)
    return lower_philox_threshold(lower_diffusion_noise(neuron))
