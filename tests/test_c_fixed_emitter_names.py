# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNames from former test_c_fixed_emitter.py

"""Focused suite: TestNames from former test_c_fixed_emitter.py."""

from __future__ import annotations

from tests.c_fixed_emitter_support import *  # noqa: F403


class TestNames:
    """How the emitter resolves names: state, parameters, input and free variables."""

    def test_state_variable(self) -> None:
        """A state variable becomes its struct field."""
        expr, *_ = _c("v", state={"v": "s->v"})
        assert "s->v" in expr

    def test_parameter(self) -> None:
        """A parameter becomes its encoded literal."""
        expr, *_ = _c("tau", params={"tau": 2560})
        assert "2560" in expr

    def test_input_current_sets_flag(self) -> None:
        """The input I maps to I_t and sets the input-used flag."""
        expr, _s, _t, _fv, _lut, used = _c("I")
        assert used is True and "I_t" in expr

    def test_free_variable_recorded(self) -> None:
        """An unknown name is reported as a free variable."""
        expr, _s, _t, free, *_ = _c("a + b", state={"a": "s->a"})
        assert free == ["b"]

    def test_free_variable_recorded_once(self) -> None:
        """A free variable used twice is reported once."""
        _e, _s, _t, free, *_ = _c("b + b")
        assert free == ["b"]
