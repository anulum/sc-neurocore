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
    def test_state_var_verbatim(self) -> None:
        assert _emit("v", {"v"}) == "v"

    def test_input_current_maps_to_I_t(self) -> None:
        assert _emit("I") == "I_t"

    def test_param_map(self) -> None:
        assert _emit("tau", set(), param_map={"tau": "P_tau"}) == "P_tau"

    def test_free_vars_recorded_in_order(self) -> None:
        code, free = emit_c_expr("a + b - a", set())
        assert code == "((a + b) - a)"
        assert free == ["a", "b"]
