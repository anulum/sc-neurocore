# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNames from former test_c_expr_emitter.py

"""Focused suite: TestNames from former test_c_expr_emitter.py."""

from __future__ import annotations

from tests.c_expr_emitter_support import *  # noqa: F403


class TestNames:
    """Check how the C++ emitter resolves names in an expression."""

    def test_state_var_verbatim(self) -> None:
        """Check that a state variable is emitted under its own name."""
        assert _emit("v", {"v"}) == "v"

    def test_input_current_maps_to_I_t(self) -> None:
        """Check that the input current ``I`` is emitted as ``I_t``."""
        assert _emit("I") == "I_t"

    def test_param_map(self) -> None:
        """Check that a parameter is renamed through the parameter map."""
        assert _emit("tau", set(), param_map={"tau": "P_tau"}) == "P_tau"

    def test_free_vars_recorded_in_order(self) -> None:
        """Check that free variables are recorded once each, in first-use order."""
        code, free = emit_c_expr("a + b - a", set())
        assert code == "((a + b) - a)"
        assert free == ["a", "b"]
