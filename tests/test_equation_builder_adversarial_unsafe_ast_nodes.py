# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUnsafeASTNodes from former test_equation_builder_adversarial.py

"""Focused suite: TestUnsafeASTNodes from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403


class TestUnsafeASTNodes:
    """Attempts to use AST node types not in the whitelist."""

    def test_lambda(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node"):
            EquationNeuron(
                equations={"v": "(lambda: 0)()"},
                state={"v": 0.0},
            )

    def test_generator_expression(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node"):
            EquationNeuron(
                equations={"v": "sum(x for x in [1,2,3])"},
                state={"v": 0.0},
            )

    def test_dict_comprehension(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node|Invalid equation"):
            EquationNeuron(
                equations={"v": "{k: v for k, v in [(1,2)]}"},
                state={"v": 0.0},
            )

    def test_starred(self) -> None:
        with pytest.raises(ValueError, match="Unsafe AST node|Invalid equation"):
            EquationNeuron(
                equations={"v": "*[1,2,3]"},
                state={"v": 0.0},
            )
