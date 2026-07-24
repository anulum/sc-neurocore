# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestASTDepthExhaustion from former test_equation_builder_adversarial.py

"""Focused suite: TestASTDepthExhaustion from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403


class TestASTDepthExhaustion:
    """Attempts to exhaust the parser with deeply nested expressions."""

    def test_depth_limit_exceeded(self) -> None:
        # Build a deeply nested expression: (((((...(v)...)))))
        depth = 30
        expr = "v"
        for _ in range(depth):
            expr = f"({expr} + 1)"
        with pytest.raises(ValueError, match="AST depth"):
            EquationNeuron(
                equations={"v": expr},
                state={"v": 0.0},
            )

    def test_depth_just_under_limit_accepted(self) -> None:
        # A moderately nested expression should still work
        depth = 8
        expr = "v"
        for _ in range(depth):
            expr = f"({expr} + 1)"
        # Should NOT raise — this is a legitimate equation
        neuron = EquationNeuron(
            equations={"v": expr},
            state={"v": 0.0},
        )
        assert neuron is not None
