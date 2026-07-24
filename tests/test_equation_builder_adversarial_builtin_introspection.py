# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBuiltinIntrospection from former test_equation_builder_adversarial.py

"""Focused suite: TestBuiltinIntrospection from former test_equation_builder_adversarial.py."""

from __future__ import annotations

from tests.equation_builder_adversarial_support import *  # noqa: F403


class TestBuiltinIntrospection:
    """Attempts to use introspection builtins."""

    @pytest.mark.parametrize(
        "func_name",
        ["getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "type"],
    )
    def test_blocked_introspection_func(self, func_name: str) -> None:
        with pytest.raises(ValueError, match="Blocked"):
            EquationNeuron(
                equations={"v": f"{func_name}(v)"},
                state={"v": 0.0},
            )
