# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real power differentiation through the equation runtime
"""Compare general power derivatives with independent finite differences."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_namespace import build_eval_namespace
from sc_neurocore.neurons.equation_program import compile_expression
from sc_neurocore.neurons.expression_derivative import differentiate


@pytest.mark.parametrize("expression", ["(v*v+1)**-3", "(v*v+1)**1.5", "(v*v+1)**(v+.5)"])
def test_general_power_chain_rule_runs_in_the_real_evaluator(expression: str) -> None:
    """Both varying exponents and bases retain the original positive real derivative."""
    source = compile_expression(expression)
    derivative = compile_expression(differentiate(expression, "v"))
    for at in [-1.5, -0.1, 0.0, 0.1, 1.5]:
        env = build_eval_namespace() | {"v": at}
        h = 1e-5
        estimate = (float(source(env | {"v": at + h})) - float(source(env | {"v": at - h}))) / (
            2 * h
        )
        assert float(derivative(env)) == pytest.approx(estimate, rel=1e-8, abs=1e-8)
